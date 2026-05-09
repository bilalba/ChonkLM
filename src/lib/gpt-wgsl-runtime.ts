import { fetchCachedRange, mapWithConcurrency, requestPersistence, type LoadStepCallback, type ProgressCallback } from "./cache";
import {
  fetchGgufHeader,
  GGML_TYPE,
  ggmlTypeName,
  type GgufFile,
} from "./gguf";
import type { ModelDef } from "./registry";
import { needsSamplingReadback, sampleFromCandidateBuffer, type SamplingOptions } from "./sampling";
import { Tokenizer } from "./tokenizer";

type TensorDtype = "float32" | "float16" | "q8_0" | "q4_k" | "q5_k" | "q6_k";
type LinearKind = "gguf-q8_0" | "gguf-q4_k" | "gguf-q5_k" | "gguf-q6_k";

interface TensorSpec {
  dtype: TensorDtype;
  ggmlType: number;
  shape: number[];
  external: {
    path: string;
    offset: number;
    length: number;
  };
}

interface GptManifest {
  format: "chonklm.gpt-wgsl.gguf.v1";
  source: string;
  model: {
    hidden: number;
    intermediate: number;
    vocab: number;
    layers: number;
    heads: number;
    headDim: number;
    maxContext: number;
    normEps: number;
  };
  tensors: Record<string, TensorSpec>;
}

interface LinearWeight {
  kind: LinearKind;
  q: GPUBuffer;
  params: GPUBuffer;
  k: number;
  n: number;
  blocks: number;
  blockBytes: number;
}

interface LayerWeights {
  attnNormWeight: GPUBuffer;
  attnNormBias: GPUBuffer;
  ffnNormWeight: GPUBuffer;
  ffnNormBias: GPUBuffer;
  qkv: LinearWeight;
  qkvBias: GPUBuffer;
  attnOutput: LinearWeight;
  attnOutputBias: GPUBuffer;
  ffnUp: LinearWeight;
  ffnUpBias: GPUBuffer;
  ffnDown: LinearWeight;
  ffnDownBias: GPUBuffer;
  keyCache: GPUBuffer;
  valueCache: GPUBuffer;
}

interface Pipelines {
  embeddingQ8: GPUComputePipeline;
  embeddingQ4K: GPUComputePipeline;
  embeddingQ5K: GPUComputePipeline;
  embeddingQ6K: GPUComputePipeline;
  layerNorm: GPUComputePipeline;
  addPosition: GPUComputePipeline;
  addInPlace: GPUComputePipeline;
  sanitize: GPUComputePipeline;
  q8Matvec: GPUComputePipeline;
  q4KMatvec: GPUComputePipeline;
  q5KMatvec: GPUComputePipeline;
  q6KMatvec: GPUComputePipeline;
  storeKv: GPUComputePipeline;
  attentionScore: GPUComputePipeline;
  attentionValue: GPUComputePipeline;
  gelu: GPUComputePipeline;
  argmaxStage1: GPUComputePipeline;
  argmaxStage2: GPUComputePipeline;
}

interface LayerBindGroups {
  attnNorm: GPUBindGroup;
  qkv: GPUBindGroup;
  qkvBias: GPUBindGroup;
  storeKv: GPUBindGroup;
  attentionScore: GPUBindGroup;
  attentionValue: GPUBindGroup;
  attnOutput: GPUBindGroup;
  attnOutputBias: GPUBindGroup;
  attnResidual: GPUBindGroup;
  sanitizeHidden: GPUBindGroup;
  ffnNorm: GPUBindGroup;
  ffnUp: GPUBindGroup;
  ffnUpBias: GPUBindGroup;
  gelu: GPUBindGroup;
  ffnDown: GPUBindGroup;
  ffnDownBias: GPUBindGroup;
  ffnResidual: GPUBindGroup;
}

interface BindGroups {
  embedding: GPUBindGroup;
  addPosition: GPUBindGroup;
  finalNorm: GPUBindGroup;
  lmHead: GPUBindGroup;
  argmaxStage1: GPUBindGroup;
  argmaxStage2: GPUBindGroup;
  layers: LayerBindGroups[];
}

export interface LoadedGptWgslModel {
  runtime: "gpt-webgpu";
  def: ModelDef;
  tokenizer: Tokenizer;
  engine: GptWgslEngine;
  ep: "webgpu";
  cachedTokenIds: number[];
  cachedNextId: number | null;
}

export async function loadGptWgslModel(
  model: ModelDef,
  onProgress?: ProgressCallback,
  onStep?: LoadStepCallback,
): Promise<LoadedGptWgslModel> {
  if (model.runtime !== "gpt-webgpu") {
    throw new Error(`loadGptWgslModel: unsupported model ${model.id}`);
  }
  if (!model.gguf) throw new Error(`GPT WGSL model ${model.id} missing 'gguf' URL`);
  if (!("gpu" in navigator)) throw new Error("GPT WGSL runtime requires WebGPU");

  onStep?.({ step: "storage", detail: "checking persistent browser storage" });
  const persisted = await requestPersistence();
  onStep?.({
    step: "storage",
    detail: persisted ? "persistent storage granted for model cache" : "using best-effort browser cache",
  });

  onStep?.({ step: "runtime", detail: "requesting WebGPU adapter" });
  const gpu = (navigator as Navigator & { gpu: GPU }).gpu;
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("GPT WGSL runtime could not acquire a WebGPU adapter");

  onStep?.({ step: "runtime", detail: "requesting WebGPU device" });
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    },
  });

  onStep?.({ step: "tokenizer", detail: "loading tokenizer and GGUF header" });
  const [tokenizer, source] = await Promise.all([
    Tokenizer.load(model),
    openGgufGptSource(model, onProgress),
  ]);

  const engine = await GptWgslEngine.create(device, source.manifest, source.loader, onStep);
  source.loader.clear();
  onStep?.({ step: "ready", detail: "GPT WebGPU runtime ready" });

  return {
    runtime: "gpt-webgpu",
    def: model,
    tokenizer,
    engine,
    ep: "webgpu",
    cachedTokenIds: [],
    cachedNextId: null,
  };
}

export async function generateGptWgsl(
  loaded: LoadedGptWgslModel,
  promptIds: number[],
  opts: {
    maxNewTokens?: number;
    temperature?: number;
    topP?: number;
    topK?: number;
    repetitionPenalty?: number;
    onToken?: (info: { id: number; text: string; cumulative: string }) => void;
    shouldStop?: () => boolean;
  } = {},
): Promise<{ text: string; ids: number[]; tokensPerSec: number }> {
  const max = Math.max(0, Math.min(opts.maxNewTokens ?? 128, loaded.def.maxContext - promptIds.length));
  const eos = new Set(loaded.def.eosIds);
  const generated: number[] = [];
  let cumulative = "";
  const sampling: SamplingOptions = {
    temperature: opts.temperature ?? 0,
    topP: opts.topP ?? 1,
    topK: opts.topK ?? 0,
    repetitionPenalty: opts.repetitionPenalty ?? 1,
  };
  const t0 = performance.now();

  let prefixLen = commonPrefixLength(loaded.cachedTokenIds, promptIds);
  if (prefixLen !== loaded.cachedTokenIds.length) {
    loaded.engine.reset();
    loaded.cachedTokenIds = [];
    loaded.cachedNextId = null;
    prefixLen = 0;
  }

  let nextId = 0;
  const suffix = promptIds.slice(prefixLen);
  if (suffix.length === 0) {
    if (loaded.cachedNextId == null) throw new Error("GPT WGSL has no cached next token for an empty prompt");
    nextId = loaded.cachedNextId;
  } else {
    for (let i = 0; i < suffix.length; i++) {
      const needLogits = i === suffix.length - 1;
      nextId = await loaded.engine.runToken(suffix[i], needLogits, {
        ...sampling,
        seenIds: needLogits ? promptIds.slice(0, prefixLen + i + 1) : undefined,
      });
    }
  }

  for (let step = 0; step < max; step++) {
    if (opts.shouldStop?.()) break;
    generated.push(nextId);
    const fullText = loaded.tokenizer.decode(generated);
    const piece = fullText.slice(cumulative.length);
    cumulative = fullText;
    opts.onToken?.({ id: nextId, text: piece, cumulative });
    if (eos.has(nextId)) break;
    nextId = await loaded.engine.runToken(nextId, true, {
      ...sampling,
      seenIds: promptIds.concat(generated),
    });
  }

  loaded.cachedTokenIds = promptIds.concat(generated);
  loaded.cachedNextId = eos.has(generated[generated.length - 1] ?? -1) ? null : nextId;

  const elapsed = (performance.now() - t0) / 1000;
  return { text: cumulative, ids: generated, tokensPerSec: generated.length / Math.max(elapsed, 1e-3) };
}

export function resetGptWgslConversation(loaded: LoadedGptWgslModel): void {
  loaded.engine.reset();
  loaded.cachedTokenIds = [];
  loaded.cachedNextId = null;
}

export function disposeGptWgsl(loaded: LoadedGptWgslModel): void {
  loaded.engine.dispose();
}

interface GptSource {
  manifest: GptManifest;
  loader: TensorLoader;
}

async function openGgufGptSource(model: ModelDef, onProgress?: ProgressCallback): Promise<GptSource> {
  const url = model.gguf!.startsWith("http") || model.gguf!.startsWith("/")
    ? model.gguf!
    : `${model.base}/${model.gguf}`;
  const gguf = await fetchGgufHeader(url, onProgress);
  const manifest = manifestFromGguf(gguf, model, url);
  return { manifest, loader: new TensorLoader(model, manifest, onProgress) };
}

function manifestFromGguf(gguf: GgufFile, model: ModelDef, url: string): GptManifest {
  const arch = gguf.kv.get("general.architecture")?.value;
  if (arch !== "gpt2") {
    throw new Error(`GPT WGSL ${model.id}: expected GGUF architecture gpt2, got ${String(arch)}`);
  }

  const hidden = ggufNumber(gguf, "gpt2.embedding_length");
  const layers = ggufNumber(gguf, "gpt2.block_count");
  const intermediate = ggufNumber(gguf, "gpt2.feed_forward_length");
  const heads = ggufNumber(gguf, "gpt2.attention.head_count");
  const contextLength = ggufNumber(gguf, "gpt2.context_length");
  const normEps = Number(gguf.kv.get("gpt2.attention.layer_norm_epsilon")?.value ?? 1e-5);
  const headDim = hidden / heads;
  if (!Number.isInteger(headDim)) {
    throw new Error(`GPT WGSL ${model.id}: hidden ${hidden} is not divisible by heads ${heads}`);
  }
  if (headDim !== 64) {
    throw new Error(`GPT WGSL ${model.id}: only 64-wide GPT-2 heads are implemented, got ${headDim}`);
  }

  const tokenEmbd = gguf.tensorMap.get("token_embd.weight");
  if (!tokenEmbd || tokenEmbd.shape.length !== 2) {
    throw new Error(`GPT WGSL ${model.id}: missing token_embd.weight`);
  }
  const vocab = tokenEmbd.shape[1];

  const tensors: Record<string, TensorSpec> = {};
  for (const info of gguf.tensors) {
    tensors[info.name] = {
      dtype: dtypeForGgmlType(info.type),
      ggmlType: info.type,
      shape: info.shape,
      external: {
        path: url,
        offset: gguf.dataOffset + info.offset,
        length: info.nBytes,
      },
    };
  }

  const required = [
    "token_embd.weight",
    "position_embd.weight",
    "output_norm.weight",
    "output_norm.bias",
    "output.weight",
  ];
  for (let i = 0; i < layers; i++) {
    required.push(
      `blk.${i}.attn_norm.weight`,
      `blk.${i}.attn_norm.bias`,
      `blk.${i}.attn_qkv.weight`,
      `blk.${i}.attn_qkv.bias`,
      `blk.${i}.attn_output.weight`,
      `blk.${i}.attn_output.bias`,
      `blk.${i}.ffn_norm.weight`,
      `blk.${i}.ffn_norm.bias`,
      `blk.${i}.ffn_up.weight`,
      `blk.${i}.ffn_up.bias`,
      `blk.${i}.ffn_down.weight`,
      `blk.${i}.ffn_down.bias`,
    );
  }
  for (const name of required) {
    if (!tensors[name]) throw new Error(`GPT WGSL ${model.id}: GGUF missing tensor ${name}`);
  }

  return {
    format: "chonklm.gpt-wgsl.gguf.v1",
    source: url,
    model: {
      hidden,
      intermediate,
      vocab,
      layers,
      heads,
      headDim,
      maxContext: Math.min(model.maxContext, contextLength),
      normEps,
    },
    tensors,
  };
}

function ggufNumber(gguf: GgufFile, key: string): number {
  const v = gguf.kv.get(key)?.value;
  if (typeof v !== "number" || !Number.isFinite(v)) {
    throw new Error(`GPT WGSL GGUF missing numeric metadata ${key}`);
  }
  return v;
}

function dtypeForGgmlType(type: number): TensorDtype {
  switch (type) {
    case GGML_TYPE.F32:
      return "float32";
    case GGML_TYPE.F16:
      return "float16";
    case GGML_TYPE.Q8_0:
      return "q8_0";
    case GGML_TYPE.Q4_K:
      return "q4_k";
    case GGML_TYPE.Q5_K:
      return "q5_k";
    case GGML_TYPE.Q6_K:
      return "q6_k";
    default:
      throw new Error(`GPT WGSL unsupported GGML tensor type ${ggmlTypeName(type)}`);
  }
}

class TensorLoader {
  constructor(
    private readonly model: ModelDef,
    private readonly manifest: GptManifest,
    private readonly onProgress?: ProgressCallback,
  ) {}

  async bytes(name: string): Promise<Uint8Array> {
    const spec = this.manifest.tensors[name];
    if (!spec) throw new Error(`GPT WGSL manifest missing tensor ${name}`);
    const url = this.tensorUrl(spec.external.path);
    const buf = await fetchCachedRange(url, spec.external.offset, spec.external.length, this.onProgress);
    return new Uint8Array(buf);
  }

  async buffer(device: GPUDevice, name: string, usage: GPUBufferUsageFlags): Promise<GPUBuffer> {
    const bytes = await this.bytes(name);
    return createBufferFromBytes(device, bytes, usage, name);
  }

  async floatBuffer(device: GPUDevice, name: string): Promise<GPUBuffer> {
    const bytes = await this.floatBytes(name);
    return createBufferFromBytes(device, bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, name);
  }

  async linearWeight(
    device: GPUDevice,
    name: string,
    k: number,
    n: number,
    usage: GPUBufferUsageFlags,
  ): Promise<LinearWeight> {
    const spec = this.manifest.tensors[name];
    if (!spec) throw new Error(`GPT WGSL manifest missing tensor ${name}`);
    validateMatrixShape(name, spec, k, n);
    const kind = linearKindForSpec(spec, name);
    const q = await this.buffer(device, name, usage);
    const block = kind === "gguf-q8_0" ? 32 : 256;
    const bytes = blockBytesForKind(kind);
    return {
      kind,
      q,
      params: createUniformU32(device, [k, n, bytes, Math.min(n, 32768)], `${name}.params`),
      k,
      n,
      blocks: k / block,
      blockBytes: bytes,
    };
  }

  private async floatBytes(name: string): Promise<Uint8Array> {
    const spec = this.manifest.tensors[name];
    if (!spec) throw new Error(`GPT WGSL manifest missing tensor ${name}`);
    const bytes = await this.bytes(name);
    if (spec.dtype === "float32") return bytes;
    if (spec.dtype !== "float16") {
      throw new Error(`GPT WGSL tensor ${name} is ${spec.dtype}, expected float32/float16`);
    }
    const src = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
    const dst = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) dst[i] = f16ToF32(src[i]);
    return new Uint8Array(dst.buffer);
  }

  private tensorUrl(path: string): string {
    if (path.startsWith("http") || path.startsWith("/")) return path;
    return `${this.model.base}/${path}`;
  }

  clear(): void {
    // Range entries are owned by the Cache API; no in-memory shard cache.
  }
}

function linearKindForSpec(spec: TensorSpec, name: string): LinearKind {
  switch (spec.dtype) {
    case "q8_0":
      return "gguf-q8_0";
    case "q4_k":
      return "gguf-q4_k";
    case "q5_k":
      return "gguf-q5_k";
    case "q6_k":
      return "gguf-q6_k";
    default:
      throw new Error(`GPT WGSL tensor ${name} is ${spec.dtype}, expected a GGUF quantized matrix`);
  }
}

function blockBytesForKind(kind: LinearKind): number {
  switch (kind) {
    case "gguf-q8_0":
      return 34;
    case "gguf-q4_k":
      return 144;
    case "gguf-q5_k":
      return 176;
    case "gguf-q6_k":
      return 210;
  }
}

function validateMatrixShape(name: string, spec: TensorSpec, k: number, n: number): void {
  if (spec.shape.length !== 2 || spec.shape[0] !== k || spec.shape[1] !== n) {
    throw new Error(`GPT WGSL tensor ${name} shape [${spec.shape.join(", ")}], expected [${k}, ${n}]`);
  }
}

class GptWgslEngine {
  private readonly hidden: number;
  private readonly intermediate: number;
  private readonly vocab: number;
  private readonly layersN: number;
  private readonly heads: number;
  private readonly maxContext: number;

  private pos = 0;
  private deviceLost = false;
  private readonly pipelines: Pipelines;
  private readonly bindGroups: BindGroups;
  private readonly tokenParams: GPUBuffer;
  private readonly posParams: GPUBuffer;
  private readonly normParams: GPUBuffer;
  private readonly addHiddenParams: GPUBuffer;
  private readonly addQkvParams: GPUBuffer;
  private readonly addIntermediateParams: GPUBuffer;
  private readonly geluParams: GPUBuffer;
  private readonly argmaxParams: GPUBuffer;
  private readonly penaltyIds: GPUBuffer;
  private readonly argmaxPartial: GPUBuffer;
  private readonly argmaxResult: GPUBuffer;
  private readonly argmaxReadback: GPUBuffer;
  private readonly candidateReadback: GPUBuffer;

  private readonly embedding: LinearWeight;
  private readonly positionEmbedding: GPUBuffer;
  private readonly finalNormWeight: GPUBuffer;
  private readonly finalNormBias: GPUBuffer;
  private readonly lmHead: LinearWeight;
  private readonly layers: LayerWeights[];

  private readonly hiddenA: GPUBuffer;
  private readonly hiddenB: GPUBuffer;
  private readonly norm: GPUBuffer;
  private readonly qkv: GPUBuffer;
  private readonly attnOut: GPUBuffer;
  private readonly ff: GPUBuffer;
  private readonly scores: GPUBuffer;
  private readonly logits: GPUBuffer;

  static async create(
    device: GPUDevice,
    manifest: GptManifest,
    loader: TensorLoader,
    onStep?: LoadStepCallback,
  ): Promise<GptWgslEngine> {
    const m = manifest.model;
    onStep?.({ step: "shaders", detail: "compiling WebGPU shader pipelines" });
    const pipelines = createPipelines(device, m);
    onStep?.({ step: "weights", detail: "uploading embedding weights to GPU" });
    const usageRead = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const usageState = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const usageScratch = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

    const embedding = await loader.linearWeight(device, "token_embd.weight", m.hidden, m.vocab, usageRead);
    const positionEmbedding = await loader.floatBuffer(device, "position_embd.weight");
    const finalNormWeight = await loader.floatBuffer(device, "output_norm.weight");
    const finalNormBias = await loader.floatBuffer(device, "output_norm.bias");
    const lmHead = await loader.linearWeight(device, "output.weight", m.hidden, m.vocab, usageRead);
    const linear = (name: string, k: number, n: number) => loader.linearWeight(device, name, k, n, usageRead);

    const loadLayer = async (i: number): Promise<LayerWeights> => {
      const [
        attnNormWeight,
        attnNormBias,
        ffnNormWeight,
        ffnNormBias,
        qkv,
        qkvBias,
        attnOutput,
        attnOutputBias,
        ffnUp,
        ffnUpBias,
        ffnDown,
        ffnDownBias,
      ] = await Promise.all([
        loader.floatBuffer(device, `blk.${i}.attn_norm.weight`),
        loader.floatBuffer(device, `blk.${i}.attn_norm.bias`),
        loader.floatBuffer(device, `blk.${i}.ffn_norm.weight`),
        loader.floatBuffer(device, `blk.${i}.ffn_norm.bias`),
        linear(`blk.${i}.attn_qkv.weight`, m.hidden, m.hidden * 3),
        loader.floatBuffer(device, `blk.${i}.attn_qkv.bias`),
        linear(`blk.${i}.attn_output.weight`, m.hidden, m.hidden),
        loader.floatBuffer(device, `blk.${i}.attn_output.bias`),
        linear(`blk.${i}.ffn_up.weight`, m.hidden, m.intermediate),
        loader.floatBuffer(device, `blk.${i}.ffn_up.bias`),
        linear(`blk.${i}.ffn_down.weight`, m.intermediate, m.hidden),
        loader.floatBuffer(device, `blk.${i}.ffn_down.bias`),
      ]);
      return {
        attnNormWeight,
        attnNormBias,
        ffnNormWeight,
        ffnNormBias,
        qkv,
        qkvBias,
        attnOutput,
        attnOutputBias,
        ffnUp,
        ffnUpBias,
        ffnDown,
        ffnDownBias,
        keyCache: createEmptyBuffer(device, m.heads * m.maxContext * m.headDim * 4, usageState, `gpt.layer.${i}.k_cache`),
        valueCache: createEmptyBuffer(device, m.heads * m.maxContext * m.headDim * 4, usageState, `gpt.layer.${i}.v_cache`),
      };
    };

    let completedLayers = 0;
    const layers = await mapWithConcurrency(
      Array.from({ length: m.layers }, (_, i) => i),
      4,
      async (i) => {
        const layer = await loadLayer(i);
        completedLayers++;
        onStep?.({
          step: "weights",
          detail: `uploaded layer ${completedLayers} / ${m.layers} to GPU`,
          progress: { current: completedLayers, total: m.layers },
        });
        return layer;
      },
    );

    const argmaxGroups = Math.ceil(m.vocab / 256);
    if (argmaxGroups > 256) throw new Error(`GPT WGSL argmax supports at most 65536 vocab entries, got ${m.vocab}`);

    onStep?.({ step: "weights", detail: "uploading lm head and scratch buffers" });
    return new GptWgslEngine(device, manifest, pipelines, {
      embedding,
      positionEmbedding,
      finalNormWeight,
      finalNormBias,
      lmHead,
      layers,
      tokenParams: createEmptyBuffer(device, 64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "gpt.token.params"),
      posParams: createEmptyBuffer(device, 64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "gpt.pos.params"),
      normParams: createUniformU32(device, [m.hidden], "gpt.norm.params"),
      addHiddenParams: createUniformU32(device, [m.hidden], "gpt.add.hidden.params"),
      addQkvParams: createUniformU32(device, [m.hidden * 3], "gpt.add.qkv.params"),
      addIntermediateParams: createUniformU32(device, [m.intermediate], "gpt.add.intermediate.params"),
      geluParams: createUniformU32(device, [m.intermediate], "gpt.gelu.params"),
      argmaxParams: createUniformU32(device, [m.vocab, argmaxGroups, 0, 1000], "gpt.argmax.params"),
      penaltyIds: createEmptyBuffer(
        device,
        Math.max(4, m.maxContext * 4),
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        "gpt.penalty_ids",
      ),
      argmaxPartial: createEmptyBuffer(device, argmaxGroups * 8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, "gpt.argmax.partial"),
      argmaxResult: createEmptyBuffer(device, 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, "gpt.argmax.result"),
      argmaxReadback: createEmptyBuffer(device, 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, "gpt.argmax.readback"),
      candidateReadback: createEmptyBuffer(device, argmaxGroups * 8, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, "gpt.candidates.readback"),
      hiddenA: createEmptyBuffer(device, m.hidden * 4, usageScratch, "gpt.hidden.a"),
      hiddenB: createEmptyBuffer(device, m.hidden * 4, usageScratch, "gpt.hidden.b"),
      norm: createEmptyBuffer(device, m.hidden * 4, usageScratch, "gpt.norm"),
      qkv: createEmptyBuffer(device, m.hidden * 3 * 4, usageScratch, "gpt.qkv"),
      attnOut: createEmptyBuffer(device, m.hidden * 4, usageScratch, "gpt.attn_out"),
      ff: createEmptyBuffer(device, m.intermediate * 4, usageScratch, "gpt.ff"),
      scores: createEmptyBuffer(device, m.heads * m.maxContext * 4, usageScratch, "gpt.scores"),
      logits: createEmptyBuffer(device, m.vocab * 4, usageScratch, "gpt.logits"),
    });
  }

  private constructor(
    private readonly device: GPUDevice,
    manifest: GptManifest,
    pipelines: Pipelines,
    buffers: {
      embedding: LinearWeight;
      positionEmbedding: GPUBuffer;
      finalNormWeight: GPUBuffer;
      finalNormBias: GPUBuffer;
      lmHead: LinearWeight;
      layers: LayerWeights[];
      tokenParams: GPUBuffer;
      posParams: GPUBuffer;
      normParams: GPUBuffer;
      addHiddenParams: GPUBuffer;
      addQkvParams: GPUBuffer;
      addIntermediateParams: GPUBuffer;
      geluParams: GPUBuffer;
      argmaxParams: GPUBuffer;
      penaltyIds: GPUBuffer;
      argmaxPartial: GPUBuffer;
      argmaxResult: GPUBuffer;
      argmaxReadback: GPUBuffer;
      candidateReadback: GPUBuffer;
      hiddenA: GPUBuffer;
      hiddenB: GPUBuffer;
      norm: GPUBuffer;
      qkv: GPUBuffer;
      attnOut: GPUBuffer;
      ff: GPUBuffer;
      scores: GPUBuffer;
      logits: GPUBuffer;
    },
  ) {
    const m = manifest.model;
    this.hidden = m.hidden;
    this.intermediate = m.intermediate;
    this.vocab = m.vocab;
    this.layersN = m.layers;
    this.heads = m.heads;
    this.maxContext = m.maxContext;
    this.pipelines = pipelines;
    this.embedding = buffers.embedding;
    this.positionEmbedding = buffers.positionEmbedding;
    this.finalNormWeight = buffers.finalNormWeight;
    this.finalNormBias = buffers.finalNormBias;
    this.lmHead = buffers.lmHead;
    this.layers = buffers.layers;
    this.tokenParams = buffers.tokenParams;
    this.posParams = buffers.posParams;
    this.normParams = buffers.normParams;
    this.addHiddenParams = buffers.addHiddenParams;
    this.addQkvParams = buffers.addQkvParams;
    this.addIntermediateParams = buffers.addIntermediateParams;
    this.geluParams = buffers.geluParams;
    this.argmaxParams = buffers.argmaxParams;
    this.penaltyIds = buffers.penaltyIds;
    this.argmaxPartial = buffers.argmaxPartial;
    this.argmaxResult = buffers.argmaxResult;
    this.argmaxReadback = buffers.argmaxReadback;
    this.candidateReadback = buffers.candidateReadback;
    this.hiddenA = buffers.hiddenA;
    this.hiddenB = buffers.hiddenB;
    this.norm = buffers.norm;
    this.qkv = buffers.qkv;
    this.attnOut = buffers.attnOut;
    this.ff = buffers.ff;
    this.scores = buffers.scores;
    this.logits = buffers.logits;
    this.bindGroups = this.createBindGroups();
    this.device.lost.then((info) => {
      this.deviceLost = true;
      console.error(`GPT WGSL WebGPU device lost: ${info.message} (${info.reason})`);
    });
  }

  reset(): void {
    this.pos = 0;
    const enc = this.device.createCommandEncoder();
    for (const layer of this.layers) {
      enc.clearBuffer(layer.keyCache);
      enc.clearBuffer(layer.valueCache);
    }
    this.device.queue.submit([enc.finish()]);
  }

  get position(): number {
    return this.pos;
  }

  async runToken(tokenId: number, needLogits: boolean, sampling?: SamplingOptions): Promise<number> {
    if (this.deviceLost) throw new Error("GPT WGSL WebGPU device was lost; reload the model to continue");
    if (this.pos >= this.maxContext) {
      throw new Error(`GPT WGSL context exhausted at ${this.maxContext} tokens`);
    }

    this.device.queue.writeBuffer(this.tokenParams, 0, new Uint32Array([tokenId, this.hidden, this.embedding.blockBytes]));
    this.device.queue.writeBuffer(this.posParams, 0, new Uint32Array([this.pos, this.maxContext]));

    const encoder = this.device.createCommandEncoder();
    this.dispatch(encoder, this.pipelineForEmbedding(this.embedding), this.bindGroups.embedding, Math.ceil(this.hidden / 128));
    this.dispatch(encoder, this.pipelines.addPosition, this.bindGroups.addPosition, Math.ceil(this.hidden / 128));

    for (let i = 0; i < this.layersN; i++) {
      const layer = this.layers[i];
      const bg = this.bindGroups.layers[i];
      this.dispatch(encoder, this.pipelines.layerNorm, bg.attnNorm, 1);
      this.matvecBound(encoder, layer.qkv, bg.qkv);
      this.dispatch(encoder, this.pipelines.addInPlace, bg.qkvBias, Math.ceil((this.hidden * 3) / 128));
      this.dispatch(encoder, this.pipelines.storeKv, bg.storeKv, this.heads);
      this.dispatch(encoder, this.pipelines.attentionScore, bg.attentionScore, this.heads, this.pos + 1);
      this.dispatch(encoder, this.pipelines.attentionValue, bg.attentionValue, this.heads);
      this.matvecBound(encoder, layer.attnOutput, bg.attnOutput);
      this.dispatch(encoder, this.pipelines.addInPlace, bg.attnOutputBias, Math.ceil(this.hidden / 128));
      this.dispatch(encoder, this.pipelines.addInPlace, bg.attnResidual, Math.ceil(this.hidden / 128));
      this.dispatch(encoder, this.pipelines.sanitize, bg.sanitizeHidden, Math.ceil(this.hidden / 128));

      this.dispatch(encoder, this.pipelines.layerNorm, bg.ffnNorm, 1);
      this.matvecBound(encoder, layer.ffnUp, bg.ffnUp);
      this.dispatch(encoder, this.pipelines.addInPlace, bg.ffnUpBias, Math.ceil(this.intermediate / 128));
      this.dispatch(encoder, this.pipelines.gelu, bg.gelu, Math.ceil(this.intermediate / 128));
      this.matvecBound(encoder, layer.ffnDown, bg.ffnDown);
      this.dispatch(encoder, this.pipelines.addInPlace, bg.ffnDownBias, Math.ceil(this.hidden / 128));
      this.dispatch(encoder, this.pipelines.addInPlace, bg.ffnResidual, Math.ceil(this.hidden / 128));
      this.dispatch(encoder, this.pipelines.sanitize, bg.sanitizeHidden, Math.ceil(this.hidden / 128));
    }

    if (needLogits) {
      this.dispatch(encoder, this.pipelines.layerNorm, this.bindGroups.finalNorm, 1);
      this.matvecBound(encoder, this.lmHead, this.bindGroups.lmHead);
      const penalizedIds = sampling?.seenIds?.slice(-this.maxContext) ?? [];
      if ((sampling?.repetitionPenalty ?? 1) !== 1 && penalizedIds.length > 0) {
        this.device.queue.writeBuffer(this.penaltyIds, 0, new Uint32Array(penalizedIds));
      }
      const candidateGroups = Math.ceil(this.vocab / 256);
      const penaltyCount = (sampling?.repetitionPenalty ?? 1) !== 1 ? penalizedIds.length : 0;
      this.device.queue.writeBuffer(
        this.argmaxParams,
        0,
        new Uint32Array([this.vocab, candidateGroups, penaltyCount, Math.round((sampling?.repetitionPenalty ?? 1) * 1000)]),
      );
      this.dispatch(encoder, this.pipelines.argmaxStage1, this.bindGroups.argmaxStage1, Math.ceil(this.vocab / 256));
      if (needsSamplingReadback(sampling)) {
        encoder.copyBufferToBuffer(this.argmaxPartial, 0, this.candidateReadback, 0, candidateGroups * 8);
      } else {
        this.dispatch(encoder, this.pipelines.argmaxStage2, this.bindGroups.argmaxStage2, 1);
        encoder.copyBufferToBuffer(this.argmaxResult, 0, this.argmaxReadback, 0, 4);
      }
    }

    this.device.queue.submit([encoder.finish()]);
    this.pos += 1;
    if (!needLogits) return 0;

    if (needsSamplingReadback(sampling)) {
      const groups = Math.ceil(this.vocab / 256);
      await this.candidateReadback.mapAsync(GPUMapMode.READ);
      const best = sampleFromCandidateBuffer(this.candidateReadback.getMappedRange(), sampling, groups);
      this.candidateReadback.unmap();
      return best;
    }

    await this.argmaxReadback.mapAsync(GPUMapMode.READ);
    const best = new Uint32Array(this.argmaxReadback.getMappedRange())[0];
    this.argmaxReadback.unmap();
    return best;
  }

  dispose(): void {
    const destroyWeight = (w: LinearWeight) => {
      w.q.destroy();
      w.params.destroy();
    };
    destroyWeight(this.embedding);
    destroyWeight(this.lmHead);
    this.positionEmbedding.destroy();
    this.finalNormWeight.destroy();
    this.finalNormBias.destroy();
    for (const layer of this.layers) {
      layer.attnNormWeight.destroy();
      layer.attnNormBias.destroy();
      layer.ffnNormWeight.destroy();
      layer.ffnNormBias.destroy();
      destroyWeight(layer.qkv);
      layer.qkvBias.destroy();
      destroyWeight(layer.attnOutput);
      layer.attnOutputBias.destroy();
      destroyWeight(layer.ffnUp);
      layer.ffnUpBias.destroy();
      destroyWeight(layer.ffnDown);
      layer.ffnDownBias.destroy();
      layer.keyCache.destroy();
      layer.valueCache.destroy();
    }
    for (const b of [
      this.tokenParams,
      this.posParams,
      this.normParams,
      this.addHiddenParams,
      this.addQkvParams,
      this.addIntermediateParams,
      this.geluParams,
      this.argmaxParams,
      this.penaltyIds,
      this.argmaxPartial,
      this.argmaxResult,
      this.argmaxReadback,
      this.candidateReadback,
      this.hiddenA,
      this.hiddenB,
      this.norm,
      this.qkv,
      this.attnOut,
      this.ff,
      this.scores,
      this.logits,
    ]) {
      b.destroy();
    }
  }

  private createBindGroups(): BindGroups {
    const bind = (pipeline: GPUComputePipeline, entries: GPUBindGroupEntry[]) =>
      this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    const linear = (input: GPUBuffer, weight: LinearWeight, output: GPUBuffer) =>
      bind(this.pipelineForWeight(weight), [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: weight.q } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: weight.params } },
      ]);
    const layerNorm = (input: GPUBuffer, weight: GPUBuffer, bias: GPUBuffer, output: GPUBuffer) =>
      bind(this.pipelines.layerNorm, [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: weight } },
        { binding: 2, resource: { buffer: bias } },
        { binding: 3, resource: { buffer: output } },
        { binding: 4, resource: { buffer: this.normParams } },
      ]);
    const addInPlace = (add: GPUBuffer, io: GPUBuffer, params: GPUBuffer) =>
      bind(this.pipelines.addInPlace, [
        { binding: 0, resource: { buffer: add } },
        { binding: 1, resource: { buffer: io } },
        { binding: 2, resource: { buffer: params } },
      ]);
    const sanitize = (io: GPUBuffer, params: GPUBuffer) =>
      bind(this.pipelines.sanitize, [
        { binding: 0, resource: { buffer: io } },
        { binding: 1, resource: { buffer: params } },
      ]);

    const layers = this.layers.map((layer) => ({
      attnNorm: layerNorm(this.hiddenA, layer.attnNormWeight, layer.attnNormBias, this.norm),
      qkv: linear(this.norm, layer.qkv, this.qkv),
      qkvBias: addInPlace(layer.qkvBias, this.qkv, this.addQkvParams),
      storeKv: bind(this.pipelines.storeKv, [
        { binding: 0, resource: { buffer: this.qkv } },
        { binding: 1, resource: { buffer: layer.keyCache } },
        { binding: 2, resource: { buffer: layer.valueCache } },
        { binding: 3, resource: { buffer: this.posParams } },
      ]),
      attentionScore: bind(this.pipelines.attentionScore, [
        { binding: 0, resource: { buffer: this.qkv } },
        { binding: 1, resource: { buffer: layer.keyCache } },
        { binding: 2, resource: { buffer: this.scores } },
        { binding: 3, resource: { buffer: this.posParams } },
      ]),
      attentionValue: bind(this.pipelines.attentionValue, [
        { binding: 0, resource: { buffer: this.scores } },
        { binding: 1, resource: { buffer: layer.valueCache } },
        { binding: 2, resource: { buffer: this.attnOut } },
        { binding: 3, resource: { buffer: this.posParams } },
      ]),
      attnOutput: linear(this.attnOut, layer.attnOutput, this.hiddenB),
      attnOutputBias: addInPlace(layer.attnOutputBias, this.hiddenB, this.addHiddenParams),
      attnResidual: addInPlace(this.hiddenB, this.hiddenA, this.addHiddenParams),
      sanitizeHidden: sanitize(this.hiddenA, this.addHiddenParams),
      ffnNorm: layerNorm(this.hiddenA, layer.ffnNormWeight, layer.ffnNormBias, this.norm),
      ffnUp: linear(this.norm, layer.ffnUp, this.ff),
      ffnUpBias: addInPlace(layer.ffnUpBias, this.ff, this.addIntermediateParams),
      gelu: bind(this.pipelines.gelu, [
        { binding: 0, resource: { buffer: this.ff } },
        { binding: 1, resource: { buffer: this.geluParams } },
      ]),
      ffnDown: linear(this.ff, layer.ffnDown, this.hiddenB),
      ffnDownBias: addInPlace(layer.ffnDownBias, this.hiddenB, this.addHiddenParams),
      ffnResidual: addInPlace(this.hiddenB, this.hiddenA, this.addHiddenParams),
    }));

    return {
      embedding: bind(this.pipelineForEmbedding(this.embedding), [
        { binding: 0, resource: { buffer: this.embedding.q } },
        { binding: 1, resource: { buffer: this.hiddenA } },
        { binding: 2, resource: { buffer: this.tokenParams } },
      ]),
      addPosition: bind(this.pipelines.addPosition, [
        { binding: 0, resource: { buffer: this.hiddenA } },
        { binding: 1, resource: { buffer: this.positionEmbedding } },
        { binding: 2, resource: { buffer: this.posParams } },
      ]),
      finalNorm: layerNorm(this.hiddenA, this.finalNormWeight, this.finalNormBias, this.norm),
      lmHead: linear(this.norm, this.lmHead, this.logits),
      argmaxStage1: bind(this.pipelines.argmaxStage1, [
        { binding: 0, resource: { buffer: this.logits } },
        { binding: 1, resource: { buffer: this.argmaxPartial } },
        { binding: 2, resource: { buffer: this.argmaxParams } },
        { binding: 3, resource: { buffer: this.penaltyIds } },
      ]),
      argmaxStage2: bind(this.pipelines.argmaxStage2, [
        { binding: 0, resource: { buffer: this.argmaxPartial } },
        { binding: 1, resource: { buffer: this.argmaxResult } },
        { binding: 2, resource: { buffer: this.argmaxParams } },
      ]),
      layers,
    };
  }

  private dispatch(
    encoder: GPUCommandEncoder,
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    x: number,
    y = 1,
  ): void {
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(x, y);
    pass.end();
  }

  private matvecBound(encoder: GPUCommandEncoder, weight: LinearWeight, bindGroup: GPUBindGroup): void {
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelineForWeight(weight));
    const xGroups = Math.min(weight.n, 32768);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(xGroups, Math.ceil(weight.n / xGroups));
    pass.end();
  }

  private pipelineForEmbedding(weight: LinearWeight): GPUComputePipeline {
    switch (weight.kind) {
      case "gguf-q8_0":
        return this.pipelines.embeddingQ8;
      case "gguf-q4_k":
        return this.pipelines.embeddingQ4K;
      case "gguf-q5_k":
        return this.pipelines.embeddingQ5K;
      case "gguf-q6_k":
        return this.pipelines.embeddingQ6K;
    }
  }

  private pipelineForWeight(weight: LinearWeight): GPUComputePipeline {
    switch (weight.kind) {
      case "gguf-q8_0":
        return this.pipelines.q8Matvec;
      case "gguf-q4_k":
        return this.pipelines.q4KMatvec;
      case "gguf-q5_k":
        return this.pipelines.q5KMatvec;
      case "gguf-q6_k":
        return this.pipelines.q6KMatvec;
    }
  }
}

function createPipelines(device: GPUDevice, m: GptManifest["model"]): Pipelines {
  const constants = shaderConstants(m);
  const module = (label: string, code: string) => device.createShaderModule({ label, code: `${constants}\n${code}` });
  const pipeline = (label: string, code: string) =>
    device.createComputePipeline({
      label,
      layout: "auto",
      compute: { module: module(label, code), entryPoint: "main" },
    });

  return {
    embeddingQ8: pipeline("gpt.embedding.q8_0", GGUF_Q8_0_EMBEDDING_WGSL),
    embeddingQ4K: pipeline("gpt.embedding.q4_k", GGUF_Q4_K_EMBEDDING_WGSL),
    embeddingQ5K: pipeline("gpt.embedding.q5_k", GGUF_Q5_K_EMBEDDING_WGSL),
    embeddingQ6K: pipeline("gpt.embedding.q6_k", GGUF_Q6_K_EMBEDDING_WGSL),
    layerNorm: pipeline("gpt.layer_norm", LAYER_NORM_WGSL),
    addPosition: pipeline("gpt.add_position", ADD_POSITION_WGSL),
    addInPlace: pipeline("gpt.add_in_place", ADD_IN_PLACE_WGSL),
    sanitize: pipeline("gpt.sanitize", SANITIZE_WGSL),
    q8Matvec: pipeline("gpt.q8_matvec", GGUF_Q8_0_MATVEC_WGSL),
    q4KMatvec: pipeline("gpt.q4_k_matvec", GGUF_Q4_K_MATVEC_WGSL),
    q5KMatvec: pipeline("gpt.q5_k_matvec", GGUF_Q5_K_MATVEC_WGSL),
    q6KMatvec: pipeline("gpt.q6_k_matvec", GGUF_Q6_K_MATVEC_WGSL),
    storeKv: pipeline("gpt.store_kv", STORE_KV_WGSL),
    attentionScore: pipeline("gpt.attention_score", ATTENTION_SCORE_WGSL),
    attentionValue: pipeline("gpt.attention_value", ATTENTION_VALUE_WGSL),
    gelu: pipeline("gpt.gelu", GELU_WGSL),
    argmaxStage1: pipeline("gpt.argmax_stage1", ARGMAX_STAGE1_WGSL),
    argmaxStage2: pipeline("gpt.argmax_stage2", ARGMAX_STAGE2_WGSL),
  };
}

function shaderConstants(m: GptManifest["model"]): string {
  return /* wgsl */ `
const HIDDEN: u32 = ${m.hidden}u;
const INTERMEDIATE: u32 = ${m.intermediate}u;
const VOCAB: u32 = ${m.vocab}u;
const HEADS: u32 = ${m.heads}u;
const HEAD_DIM: u32 = ${m.headDim}u;
const MAX_CONTEXT: u32 = ${m.maxContext}u;
const INV_SQRT_HEAD_DIM: f32 = ${1 / Math.sqrt(m.headDim)};
const NORM_EPS: f32 = ${m.normEps};
`;
}

function createEmptyBuffer(
  device: GPUDevice,
  size: number,
  usage: GPUBufferUsageFlags,
  label: string,
): GPUBuffer {
  return device.createBuffer({ label, size: Math.max(4, align4(size)), usage });
}

function createBufferFromBytes(
  device: GPUDevice,
  bytes: Uint8Array,
  usage: GPUBufferUsageFlags,
  label: string,
): GPUBuffer {
  const padded = new Uint8Array(align4(bytes.byteLength));
  padded.set(bytes);
  const buffer = createEmptyBuffer(device, padded.byteLength, usage, label);
  device.queue.writeBuffer(buffer, 0, padded);
  return buffer;
}

function createUniformU32(device: GPUDevice, values: number[], label: string): GPUBuffer {
  const data = new Uint32Array(16);
  data.set(values);
  return createBufferFromBytes(device, new Uint8Array(data.buffer), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label);
}

function align4(n: number): number {
  return (n + 3) & ~3;
}

function f16ToF32(h: number): number {
  const sign = (h & 0x8000) ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x03ff;
  if (exp === 0) return sign * Math.pow(2, -14) * (frac / 1024);
  if (exp === 0x1f) return frac ? NaN : sign * Infinity;
  return sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

function commonPrefixLength(a: readonly number[], b: readonly number[]): number {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a[i] === b[i]) i++;
  return i;
}

const GGUF_Q8_0_MATVEC_WGSL = /* wgsl */ `
struct Params {
  k: u32,
  n: u32,
  block_bytes: u32,
  x_groups: u32,
};

@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partial: array<f32, 128>;

fn byte_at(index: u32) -> u32 {
  let word = data[index >> 2u];
  let shift = (index & 3u) * 8u;
  return (word >> shift) & 0xffu;
}

fn i8_at(index: u32) -> i32 {
  let b = byte_at(index);
  return select(i32(b), i32(b) - 256, b >= 128u);
}

fn f16_at(index: u32) -> f32 {
  return unpack2x16float(byte_at(index) | (byte_at(index + 1u) << 8u)).x;
}

fn q8_0_weight(row: u32, col: u32) -> f32 {
  let blocks = params.k / 32u;
  let block = col >> 5u;
  let within = col & 31u;
  let base = row * blocks * params.block_bytes + block * params.block_bytes;
  return f32(i8_at(base + 2u + within)) * f16_at(base);
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = wg.x + wg.y * params.x_groups;
  if (n >= params.n) {
    return;
  }

  var sum = 0.0;
  for (var k = lid.x; k < params.k; k = k + 128u) {
    sum = sum + input[k] * q8_0_weight(n, k);
  }

  partial[lid.x] = sum;
  workgroupBarrier();

  var stride = 64u;
  loop {
    if (lid.x < stride) {
      partial[lid.x] = partial[lid.x] + partial[lid.x + stride];
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride = stride >> 1u;
  }

  if (lid.x == 0u) {
    output[n] = partial[0];
  }
}
`;

const GGUF_Q4_K_MATVEC_WGSL = /* wgsl */ `
struct Params {
  k: u32,
  n: u32,
  block_bytes: u32,
  x_groups: u32,
};

@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partial: array<f32, 128>;

fn byte_at(index: u32) -> u32 {
  let word = data[index >> 2u];
  let shift = (index & 3u) * 8u;
  return (word >> shift) & 0xffu;
}

fn f16_at(index: u32) -> f32 {
  return unpack2x16float(byte_at(index) | (byte_at(index + 1u) << 8u)).x;
}

fn scale_min_k4(j: u32, base: u32) -> vec2<u32> {
  if (j < 4u) {
    return vec2<u32>(byte_at(base + j) & 63u, byte_at(base + j + 4u) & 63u);
  }
  let d = (byte_at(base + j + 4u) & 15u) | ((byte_at(base + j - 4u) >> 6u) << 4u);
  let m = (byte_at(base + j + 4u) >> 4u) | ((byte_at(base + j) >> 6u) << 4u);
  return vec2<u32>(d, m);
}

fn q4_k_weight(row: u32, col: u32) -> f32 {
  let blocks = params.k / 256u;
  let super_block = col >> 8u;
  let within = col & 255u;
  let group64 = within >> 6u;
  let lane = within & 31u;
  let high = (within & 32u) != 0u;
  let base = row * blocks * params.block_bytes + super_block * params.block_bytes;
  let d = f16_at(base);
  let dmin = f16_at(base + 2u);
  let sm = scale_min_k4(group64 * 2u + select(0u, 1u, high), base + 4u);
  let q_byte = byte_at(base + 16u + group64 * 32u + lane);
  let qv = select(q_byte & 15u, q_byte >> 4u, high);
  return d * f32(sm.x) * f32(qv) - dmin * f32(sm.y);
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = wg.x + wg.y * params.x_groups;
  if (n >= params.n) {
    return;
  }

  var sum = 0.0;
  for (var k = lid.x; k < params.k; k = k + 128u) {
    sum = sum + input[k] * q4_k_weight(n, k);
  }

  partial[lid.x] = sum;
  workgroupBarrier();

  var stride = 64u;
  loop {
    if (lid.x < stride) {
      partial[lid.x] = partial[lid.x] + partial[lid.x + stride];
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride = stride >> 1u;
  }

  if (lid.x == 0u) {
    output[n] = partial[0];
  }
}
`;

const GGUF_Q5_K_MATVEC_WGSL = /* wgsl */ `
struct Params {
  k: u32,
  n: u32,
  block_bytes: u32,
  x_groups: u32,
};

@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partial: array<f32, 128>;

fn byte_at(index: u32) -> u32 {
  let word = data[index >> 2u];
  let shift = (index & 3u) * 8u;
  return (word >> shift) & 0xffu;
}

fn f16_at(index: u32) -> f32 {
  return unpack2x16float(byte_at(index) | (byte_at(index + 1u) << 8u)).x;
}

fn scale_min_k4(j: u32, base: u32) -> vec2<u32> {
  if (j < 4u) {
    return vec2<u32>(byte_at(base + j) & 63u, byte_at(base + j + 4u) & 63u);
  }
  let d = (byte_at(base + j + 4u) & 15u) | ((byte_at(base + j - 4u) >> 6u) << 4u);
  let m = (byte_at(base + j + 4u) >> 4u) | ((byte_at(base + j) >> 6u) << 4u);
  return vec2<u32>(d, m);
}

fn q5_k_weight(row: u32, col: u32) -> f32 {
  let blocks = params.k / 256u;
  let super_block = col >> 8u;
  let within = col & 255u;
  let group64 = within >> 6u;
  let lane = within & 31u;
  let high = (within & 32u) != 0u;
  let base = row * blocks * params.block_bytes + super_block * params.block_bytes;
  let d = f16_at(base);
  let dmin = f16_at(base + 2u);
  let sm = scale_min_k4(group64 * 2u + select(0u, 1u, high), base + 4u);
  let q_byte = byte_at(base + 48u + group64 * 32u + lane);
  let low4 = select(q_byte & 15u, q_byte >> 4u, high);
  let qh_mask = select(1u, 2u, high) << (group64 * 2u);
  let high_bit = select(0u, 16u, (byte_at(base + 16u + lane) & qh_mask) != 0u);
  return d * f32(sm.x) * f32(low4 + high_bit) - dmin * f32(sm.y);
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = wg.x + wg.y * params.x_groups;
  if (n >= params.n) {
    return;
  }

  var sum = 0.0;
  for (var k = lid.x; k < params.k; k = k + 128u) {
    sum = sum + input[k] * q5_k_weight(n, k);
  }

  partial[lid.x] = sum;
  workgroupBarrier();

  var stride = 64u;
  loop {
    if (lid.x < stride) {
      partial[lid.x] = partial[lid.x] + partial[lid.x + stride];
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride = stride >> 1u;
  }

  if (lid.x == 0u) {
    output[n] = partial[0];
  }
}
`;

const GGUF_Q6_K_MATVEC_WGSL = /* wgsl */ `
struct Params {
  k: u32,
  n: u32,
  block_bytes: u32,
  x_groups: u32,
};

@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partial: array<f32, 128>;

fn byte_at(index: u32) -> u32 {
  let word = data[index >> 2u];
  let shift = (index & 3u) * 8u;
  return (word >> shift) & 0xffu;
}

fn i8_at(index: u32) -> i32 {
  let b = byte_at(index);
  return select(i32(b), i32(b) - 256, b >= 128u);
}

fn f16_at(index: u32) -> f32 {
  return unpack2x16float(byte_at(index) | (byte_at(index + 1u) << 8u)).x;
}

fn q6_k_weight(row: u32, col: u32) -> f32 {
  let blocks = params.k / 256u;
  let super_block = col >> 8u;
  let within = col & 255u;
  let half = within >> 7u;
  let local = within & 127u;
  let lane = local & 31u;
  let quarter = local >> 5u;
  let base = row * blocks * params.block_bytes + super_block * params.block_bytes;
  let ql_base = base + half * 64u;
  let qh_base = base + 128u + half * 32u;
  let sc_base = base + 192u + half * 8u;
  let ql_index = ql_base + lane + select(0u, 32u, (quarter & 1u) == 1u);
  let ql = byte_at(ql_index);
  let low = (quarter < 2u);
  let lo = select(ql >> 4u, ql & 15u, low);
  let qh = (byte_at(qh_base + lane) >> (quarter * 2u)) & 3u;
  let q = i32(lo | (qh << 4u)) - 32;
  let scale = i8_at(sc_base + (lane >> 4u) + quarter * 2u);
  return f16_at(base + 208u) * f32(scale) * f32(q);
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = wg.x + wg.y * params.x_groups;
  if (n >= params.n) {
    return;
  }

  var sum = 0.0;
  for (var k = lid.x; k < params.k; k = k + 128u) {
    sum = sum + input[k] * q6_k_weight(n, k);
  }

  partial[lid.x] = sum;
  workgroupBarrier();

  var stride = 64u;
  loop {
    if (lid.x < stride) {
      partial[lid.x] = partial[lid.x] + partial[lid.x + stride];
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride = stride >> 1u;
  }

  if (lid.x == 0u) {
    output[n] = partial[0];
  }
}
`;

const GGUF_Q8_0_EMBEDDING_WGSL = /* wgsl */ `
struct Params {
  token: u32,
  k: u32,
  block_bytes: u32,
};

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn byte_at(index: u32) -> u32 {
  let word = data[index >> 2u];
  let shift = (index & 3u) * 8u;
  return (word >> shift) & 0xffu;
}

fn i8_at(index: u32) -> i32 {
  let b = byte_at(index);
  return select(i32(b), i32(b) - 256, b >= 128u);
}

fn f16_at(index: u32) -> f32 {
  return unpack2x16float(byte_at(index) | (byte_at(index + 1u) << 8u)).x;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= params.k) {
    return;
  }
  let blocks = params.k / 32u;
  let block = k >> 5u;
  let within = k & 31u;
  let base = params.token * blocks * params.block_bytes + block * params.block_bytes;
  output[k] = f32(i8_at(base + 2u + within)) * f16_at(base);
}
`;

const GGUF_Q4_K_EMBEDDING_WGSL = /* wgsl */ `
struct Params {
  token: u32,
  k: u32,
  block_bytes: u32,
};

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn byte_at(index: u32) -> u32 {
  let word = data[index >> 2u];
  let shift = (index & 3u) * 8u;
  return (word >> shift) & 0xffu;
}

fn f16_at(index: u32) -> f32 {
  return unpack2x16float(byte_at(index) | (byte_at(index + 1u) << 8u)).x;
}

fn scale_min_k4(j: u32, base: u32) -> vec2<u32> {
  if (j < 4u) {
    return vec2<u32>(byte_at(base + j) & 63u, byte_at(base + j + 4u) & 63u);
  }
  let d = (byte_at(base + j + 4u) & 15u) | ((byte_at(base + j - 4u) >> 6u) << 4u);
  let m = (byte_at(base + j + 4u) >> 4u) | ((byte_at(base + j) >> 6u) << 4u);
  return vec2<u32>(d, m);
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= params.k) {
    return;
  }
  let blocks = params.k / 256u;
  let super_block = k >> 8u;
  let within = k & 255u;
  let group64 = within >> 6u;
  let lane = within & 31u;
  let high = (within & 32u) != 0u;
  let base = params.token * blocks * params.block_bytes + super_block * params.block_bytes;
  let sm = scale_min_k4(group64 * 2u + select(0u, 1u, high), base + 4u);
  let q_byte = byte_at(base + 16u + group64 * 32u + lane);
  let qv = select(q_byte & 15u, q_byte >> 4u, high);
  output[k] = f16_at(base) * f32(sm.x) * f32(qv) - f16_at(base + 2u) * f32(sm.y);
}
`;

const GGUF_Q5_K_EMBEDDING_WGSL = /* wgsl */ `
struct Params {
  token: u32,
  k: u32,
  block_bytes: u32,
};

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn byte_at(index: u32) -> u32 {
  let word = data[index >> 2u];
  let shift = (index & 3u) * 8u;
  return (word >> shift) & 0xffu;
}

fn f16_at(index: u32) -> f32 {
  return unpack2x16float(byte_at(index) | (byte_at(index + 1u) << 8u)).x;
}

fn scale_min_k4(j: u32, base: u32) -> vec2<u32> {
  if (j < 4u) {
    return vec2<u32>(byte_at(base + j) & 63u, byte_at(base + j + 4u) & 63u);
  }
  let d = (byte_at(base + j + 4u) & 15u) | ((byte_at(base + j - 4u) >> 6u) << 4u);
  let m = (byte_at(base + j + 4u) >> 4u) | ((byte_at(base + j) >> 6u) << 4u);
  return vec2<u32>(d, m);
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= params.k) {
    return;
  }
  let blocks = params.k / 256u;
  let super_block = k >> 8u;
  let within = k & 255u;
  let group64 = within >> 6u;
  let lane = within & 31u;
  let high = (within & 32u) != 0u;
  let base = params.token * blocks * params.block_bytes + super_block * params.block_bytes;
  let sm = scale_min_k4(group64 * 2u + select(0u, 1u, high), base + 4u);
  let q_byte = byte_at(base + 48u + group64 * 32u + lane);
  let low4 = select(q_byte & 15u, q_byte >> 4u, high);
  let qh_mask = select(1u, 2u, high) << (group64 * 2u);
  let high_bit = select(0u, 16u, (byte_at(base + 16u + lane) & qh_mask) != 0u);
  output[k] = f16_at(base) * f32(sm.x) * f32(low4 + high_bit) - f16_at(base + 2u) * f32(sm.y);
}
`;

const GGUF_Q6_K_EMBEDDING_WGSL = /* wgsl */ `
struct Params {
  token: u32,
  k: u32,
  block_bytes: u32,
};

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn byte_at(index: u32) -> u32 {
  let word = data[index >> 2u];
  let shift = (index & 3u) * 8u;
  return (word >> shift) & 0xffu;
}

fn i8_at(index: u32) -> i32 {
  let b = byte_at(index);
  return select(i32(b), i32(b) - 256, b >= 128u);
}

fn f16_at(index: u32) -> f32 {
  return unpack2x16float(byte_at(index) | (byte_at(index + 1u) << 8u)).x;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= params.k) {
    return;
  }
  let blocks = params.k / 256u;
  let super_block = k >> 8u;
  let within = k & 255u;
  let half = within >> 7u;
  let local = within & 127u;
  let lane = local & 31u;
  let quarter = local >> 5u;
  let base = params.token * blocks * params.block_bytes + super_block * params.block_bytes;
  let ql_base = base + half * 64u;
  let qh_base = base + 128u + half * 32u;
  let sc_base = base + 192u + half * 8u;
  let ql_index = ql_base + lane + select(0u, 32u, (quarter & 1u) == 1u);
  let ql = byte_at(ql_index);
  let low = (quarter < 2u);
  let lo = select(ql >> 4u, ql & 15u, low);
  let qh = (byte_at(qh_base + lane) >> (quarter * 2u)) & 3u;
  let q = i32(lo | (qh << 4u)) - 32;
  let scale = i8_at(sc_base + (lane >> 4u) + quarter * 2u);
  output[k] = f16_at(base + 208u) * f32(scale) * f32(q);
}
`;

const ADD_POSITION_WGSL = /* wgsl */ `
struct Params {
  pos: u32,
  max_context: u32,
};

@group(0) @binding(0) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> position_embedding: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h = gid.x;
  if (h < HIDDEN) {
    hidden[h] = hidden[h] + position_embedding[params.pos * HIDDEN + h];
  }
}
`;

const LAYER_NORM_WGSL = /* wgsl */ `
struct Params {
  n: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> sum_partial: array<f32, 256>;
var<workgroup> sq_partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  var sum = 0.0;
  var sq = 0.0;
  for (var i = lid.x; i < params.n; i = i + 256u) {
    let v = input[i];
    sum = sum + v;
    sq = sq + v * v;
  }
  sum_partial[lid.x] = sum;
  sq_partial[lid.x] = sq;
  workgroupBarrier();

  var stride = 128u;
  loop {
    if (lid.x < stride) {
      sum_partial[lid.x] = sum_partial[lid.x] + sum_partial[lid.x + stride];
      sq_partial[lid.x] = sq_partial[lid.x] + sq_partial[lid.x + stride];
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride = stride >> 1u;
  }

  let mean = sum_partial[0] / f32(params.n);
  let variance = max(sq_partial[0] / f32(params.n) - mean * mean, 0.0);
  let inv_std = inverseSqrt(variance + NORM_EPS);
  for (var i = lid.x; i < params.n; i = i + 256u) {
    let v = (input[i] - mean) * inv_std;
    output[i] = v * weight[i] + bias[i];
  }
}
`;

const ADD_IN_PLACE_WGSL = /* wgsl */ `
struct Params {
  n: u32,
};

@group(0) @binding(0) var<storage, read> add: array<f32>;
@group(0) @binding(1) var<storage, read_write> io: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < params.n) {
    io[i] = io[i] + add[i];
  }
}
`;

const SANITIZE_WGSL = /* wgsl */ `
struct Params {
  n: u32,
};

@group(0) @binding(0) var<storage, read_write> io: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < params.n) {
    let v = io[i];
    if (v != v) {
      io[i] = 0.0;
    } else {
      io[i] = clamp(v, -10000.0, 10000.0);
    }
  }
}
`;

const STORE_KV_WGSL = /* wgsl */ `
struct Params {
  pos: u32,
  max_context: u32,
};

@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> key_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> value_cache: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let head = wg.x;
  let d = lid.x;
  if (head >= HEADS || d >= HEAD_DIM) {
    return;
  }
  let idx = head * HEAD_DIM + d;
  let cache_idx = (head * MAX_CONTEXT + params.pos) * HEAD_DIM + d;
  key_cache[cache_idx] = qkv[HIDDEN + idx];
  value_cache[cache_idx] = qkv[2u * HIDDEN + idx];
}
`;

const ATTENTION_SCORE_WGSL = /* wgsl */ `
struct Params {
  pos: u32,
  max_context: u32,
};

@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> key_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let head = wg.x;
  let t = wg.y;
  if (head >= HEADS || t > params.pos) {
    return;
  }
  let d = lid.x;
  let q = qkv[head * HEAD_DIM + d];
  let k = key_cache[(head * MAX_CONTEXT + t) * HEAD_DIM + d];
  partial[d] = q * k;
  workgroupBarrier();

  var stride = 32u;
  loop {
    if (d < stride) {
      partial[d] = partial[d] + partial[d + stride];
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride = stride >> 1u;
  }
  if (d == 0u) {
    scores[head * MAX_CONTEXT + t] = partial[0] * INV_SQRT_HEAD_DIM;
  }
}
`;

const ATTENTION_VALUE_WGSL = /* wgsl */ `
struct Params {
  pos: u32,
  max_context: u32,
};

@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(1) var<storage, read> value_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let head = wg.x;
  let d = lid.x;
  if (head >= HEADS || d >= HEAD_DIM) {
    return;
  }

  var max_score = -3.402823e38;
  for (var t = 0u; t <= params.pos; t = t + 1u) {
    max_score = max(max_score, scores[head * MAX_CONTEXT + t]);
  }

  var denom = 0.0;
  var acc = 0.0;
  for (var t = 0u; t <= params.pos; t = t + 1u) {
    let p = exp(scores[head * MAX_CONTEXT + t] - max_score);
    denom = denom + p;
    acc = acc + p * value_cache[(head * MAX_CONTEXT + t) * HEAD_DIM + d];
  }
  output[head * HEAD_DIM + d] = acc / max(denom, 1e-20);
}
`;

const GELU_WGSL = /* wgsl */ `
struct Params {
  n: u32,
};

@group(0) @binding(0) var<storage, read_write> io: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

fn gelu(x: f32) -> f32 {
  if (x != x) {
    return 0.0;
  }
  if (x > 10.0) {
    return x;
  }
  if (x < -10.0) {
    return 0.0;
  }
  let c = 0.7978845608028654;
  return 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < params.n) {
    io[i] = gelu(io[i]);
  }
}
`;

const ARGMAX_STAGE1_WGSL = /* wgsl */ `
struct Params {
  n: u32,
  groups: u32,
  penalty_count: u32,
  repetition_penalty_milli: u32,
};

struct Pair {
  value: f32,
  index: u32,
};

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> partial_pairs: array<Pair>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> penalty_ids: array<u32>;

var<workgroup> values: array<f32, 256>;
var<workgroup> indices: array<u32, 256>;

fn is_penalized(id: u32) -> bool {
  for (var j = 0u; j < params.penalty_count; j = j + 1u) {
    if (penalty_ids[j] == id) {
      return true;
    }
  }
  return false;
}

fn repetition_penalized(id: u32, value: f32) -> f32 {
  if (params.penalty_count == 0u || !is_penalized(id)) {
    return value;
  }
  let penalty = f32(params.repetition_penalty_milli) / 1000.0;
  return select(value * penalty, value / penalty, value > 0.0);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let i = wg.x * 256u + lid.x;
  if (i < params.n) {
    values[lid.x] = repetition_penalized(i, logits[i]);
    indices[lid.x] = i;
  } else {
    values[lid.x] = -3.402823e38;
    indices[lid.x] = 0u;
  }
  workgroupBarrier();

  var stride = 128u;
  loop {
    if (lid.x < stride) {
      let other = lid.x + stride;
      if (values[other] > values[lid.x]) {
        values[lid.x] = values[other];
        indices[lid.x] = indices[other];
      }
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride = stride >> 1u;
  }

  if (lid.x == 0u) {
    partial_pairs[wg.x].value = values[0];
    partial_pairs[wg.x].index = indices[0];
  }
}
`;

const ARGMAX_STAGE2_WGSL = /* wgsl */ `
struct Params {
  n: u32,
  groups: u32,
};

struct Pair {
  value: f32,
  index: u32,
};

@group(0) @binding(0) var<storage, read> partial_pairs: array<Pair>;
@group(0) @binding(1) var<storage, read_write> result: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> values: array<f32, 256>;
var<workgroup> indices: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  if (lid.x < params.groups) {
    values[lid.x] = partial_pairs[lid.x].value;
    indices[lid.x] = partial_pairs[lid.x].index;
  } else {
    values[lid.x] = -3.402823e38;
    indices[lid.x] = 0u;
  }
  workgroupBarrier();

  var stride = 128u;
  loop {
    if (lid.x < stride) {
      let other = lid.x + stride;
      if (values[other] > values[lid.x]) {
        values[lid.x] = values[other];
        indices[lid.x] = indices[other];
      }
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride = stride >> 1u;
  }

  if (lid.x == 0u) {
    result[0] = indices[0];
  }
}
`;
