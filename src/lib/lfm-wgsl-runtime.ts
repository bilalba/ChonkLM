// Experimental hand-written WebGPU runtime for LiquidAI/LFM2.5-350M.
//
// This is intentionally narrow: it implements the official GGUF Q8_0 /
// Q4_K_M artifacts through range fetches.

import { fetchCachedRange, mapWithConcurrency, requestPersistence, type LoadStepCallback, type ProgressCallback } from "./cache";
import {
  fetchGgufHeader,
  GGML_TYPE,
  ggmlTypeName,
  type GgufFile,
  type GgufTensorInfo,
} from "./gguf";
import type { ModelDef } from "./registry";
import { needsCandidateReadback, sampleFromCandidateBuffer, type SamplingOptions } from "./sampling";
import { Tokenizer } from "./tokenizer";

type TensorDtype = "float32" | "float16" | "uint8" | "int64";

interface TensorSpec {
  dtype: TensorDtype;
  shape: number[];
  external?: { path: string; offset: number; length: number };
  inlineBase64?: string;
}

interface LfmManifest {
  format: "chonklm.lfm2-wgsl.v1";
  source?: string;
  model: {
    hidden: number;
    intermediate: number;
    vocab: number;
    layers: number;
    heads: number;
    kvHeads: number;
    headDim: number;
    maxContext: number;
    layerTypes: ("conv" | "attention")[];
  };
  tensors: Record<string, TensorSpec>;
}

type LinearKind = "gguf-q8_0" | "gguf-q4_k" | "gguf-q6_k";

interface LinearWeight {
  kind: LinearKind;
  q: GPUBuffer;
  scales?: GPUBuffer;
  zp?: GPUBuffer;
  params: GPUBuffer;
  k: number;
  n: number;
  blocks: number;
}

interface EmbeddingWeight {
  kind: LinearKind;
  q: GPUBuffer;
  params: GPUBuffer;
}

interface MlpWeights {
  gate: LinearWeight;
  up: LinearWeight;
  down: LinearWeight;
}

interface ConvLayer {
  kind: "conv";
  operatorNorm: GPUBuffer;
  ffnNorm: GPUBuffer;
  convWeight: GPUBuffer;
  convIn: LinearWeight;
  convOut: LinearWeight;
  mlp: MlpWeights;
  state: GPUBuffer;
}

interface AttentionLayer {
  kind: "attention";
  operatorNorm: GPUBuffer;
  ffnNorm: GPUBuffer;
  qNorm: GPUBuffer;
  kNorm: GPUBuffer;
  qProj: LinearWeight;
  kProj: LinearWeight;
  vProj: LinearWeight;
  oProj: LinearWeight;
  mlp: MlpWeights;
  keyCache: GPUBuffer;
  valueCache: GPUBuffer;
}

type Layer = ConvLayer | AttentionLayer;

interface Pipelines {
  embedding: GPUComputePipeline;
  embeddingQ8: GPUComputePipeline;
  embeddingQ4K: GPUComputePipeline;
  embeddingQ6K: GPUComputePipeline;
  rmsNorm: GPUComputePipeline;
  addRmsNorm: GPUComputePipeline;
  q4Matvec: GPUComputePipeline;
  q8Matvec: GPUComputePipeline;
  q4KMatvec: GPUComputePipeline;
  q6KMatvec: GPUComputePipeline;
  convBlock: GPUComputePipeline;
  add: GPUComputePipeline;
  siluMul: GPUComputePipeline;
  qkNormRopeStore: GPUComputePipeline;
  attentionScore: GPUComputePipeline;
  attentionValue: GPUComputePipeline;
  argmax: GPUComputePipeline;
  topk256: GPUComputePipeline;
}

interface CommonLayerBindGroups {
  operatorNorm: GPUBindGroup;
  opAddNorm: GPUBindGroup;
  mlpGate: GPUBindGroup;
  mlpUp: GPUBindGroup;
  silu: GPUBindGroup;
  mlpDown: GPUBindGroup;
  ffnAdd: GPUBindGroup;
}

interface ConvLayerBindGroups extends CommonLayerBindGroups {
  kind: "conv";
  convIn: GPUBindGroup;
  convBlock: GPUBindGroup;
  convOut: GPUBindGroup;
  opAdd: GPUBindGroup;
}

interface AttentionLayerBindGroups extends CommonLayerBindGroups {
  kind: "attention";
  qProj: GPUBindGroup;
  kProj: GPUBindGroup;
  vProj: GPUBindGroup;
  qkNormRopeStore: GPUBindGroup;
  attentionScore: GPUBindGroup;
  attentionValue: GPUBindGroup;
  oProj: GPUBindGroup;
  opAdd: GPUBindGroup;
}

type LayerBindGroups = ConvLayerBindGroups | AttentionLayerBindGroups;

interface BindGroups {
  embedding: GPUBindGroup;
  finalNorm: GPUBindGroup;
  lmHead: GPUBindGroup;
  argmax: GPUBindGroup;
  topk256: GPUBindGroup;
  layers: LayerBindGroups[];
}

export interface LoadedLfmWgslModel {
  runtime: "lfm2-webgpu";
  def: ModelDef;
  tokenizer: Tokenizer;
  engine: LfmWgslEngine;
  ep: "webgpu";
  cachedTokenIds: number[];
  cachedNextId: number | null;
}

export async function loadLfmWgslModel(
  model: ModelDef,
  onProgress?: ProgressCallback,
  onStep?: LoadStepCallback,
): Promise<LoadedLfmWgslModel> {
  if (model.runtime !== "lfm2-webgpu") {
    throw new Error(`loadLfmWgslModel: unsupported model ${model.id}`);
  }
  if (!("gpu" in navigator)) {
    throw new Error("LFM WGSL runtime requires WebGPU");
  }

  onStep?.({ step: "storage", detail: "checking persistent browser storage" });
  const persisted = await requestPersistence();
  onStep?.({
    step: "storage",
    detail: persisted ? "persistent storage granted for model cache" : "using best-effort browser cache",
  });

  onStep?.({ step: "runtime", detail: "requesting WebGPU adapter" });
  const gpu = (navigator as Navigator & { gpu: GPU }).gpu;
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("LFM WGSL runtime could not acquire a WebGPU adapter");
  onStep?.({ step: "runtime", detail: "requesting WebGPU device" });
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    },
  });

  onStep?.({ step: "tokenizer", detail: "loading tokenizer and GGUF header" });
  const tokenizerPromise = Tokenizer.load(model);
  const [manifest, loader] = await openGgufLfmSource(model, onProgress);
  const tokenizer = await tokenizerPromise;

  const engine = await LfmWgslEngine.create(device, manifest, loader, onStep);
  loader.clear();
  onStep?.({ step: "ready", detail: "LFM WebGPU runtime ready" });
  return {
    runtime: "lfm2-webgpu",
    def: model,
    tokenizer,
    engine,
    ep: "webgpu",
    cachedTokenIds: [],
    cachedNextId: null,
  };
}

export async function generateLfmWgsl(
  loaded: LoadedLfmWgslModel,
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
  const max = Math.min(opts.maxNewTokens ?? 128, loaded.def.maxContext - promptIds.length);
  const eos = new Set(loaded.def.eosIds);
  const generated: number[] = [];
  let cumulative = "";
  const sampling = {
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
  if (suffix.length === 0 && loaded.cachedNextId != null) {
    nextId = loaded.cachedNextId;
  } else {
    for (let i = 0; i < suffix.length; i++) {
      const seenIds = promptIds.slice(0, prefixLen + i + 1);
      nextId = await loaded.engine.runToken(suffix[i], i === suffix.length - 1, {
        ...sampling,
        seenIds,
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

export function disposeLfmWgsl(loaded: LoadedLfmWgslModel): void {
  loaded.engine.dispose();
}

export function resetLfmWgslConversation(loaded: LoadedLfmWgslModel): void {
  loaded.engine.reset();
  loaded.cachedTokenIds = [];
  loaded.cachedNextId = null;
}

interface LfmTensorSource {
  floatBuffer(device: GPUDevice, name: string): Promise<GPUBuffer>;
  linearWeight(
    device: GPUDevice,
    name: string,
    k: number,
    n: number,
    usage: GPUBufferUsageFlags,
  ): Promise<LinearWeight>;
  embeddingWeight(device: GPUDevice, usage: GPUBufferUsageFlags): Promise<EmbeddingWeight>;
  clear(): void;
}

async function openGgufLfmSource(
  model: ModelDef,
  onProgress?: ProgressCallback,
): Promise<[LfmManifest, LfmTensorSource]> {
  if (!model.gguf) throw new Error(`LFM GGUF model ${model.id} missing gguf path`);
  const url = model.gguf.startsWith("http") ? model.gguf : `${model.base}/${model.gguf}`;
  const gguf = await fetchGgufHeader(url, onProgress);
  const arch = gguf.kv.get("general.architecture")?.value;
  if (arch !== "lfm2") {
    throw new Error(`unsupported LFM GGUF architecture: ${String(arch)}`);
  }
  const layerTypes = model.layerTypes;
  if (!layerTypes) throw new Error(`LFM GGUF model ${model.id} missing registry layerTypes`);
  const manifest: LfmManifest = {
    format: "chonklm.lfm2-wgsl.v1",
    source: model.gguf,
    model: {
      hidden: model.headDim * model.kvHeads * 2,
      intermediate: 4608,
      vocab: model.vocab,
      layers: model.layers,
      heads: 16,
      kvHeads: model.kvHeads,
      headDim: model.headDim,
      maxContext: model.maxContext,
      layerTypes: layerTypes.filter((t): t is "conv" | "attention" => t === "conv" || t === "attention"),
    },
    tensors: {},
  };
  return [manifest, new GgufTensorLoader(url, gguf, onProgress)];
}

class GgufTensorLoader implements LfmTensorSource {
  constructor(
    private url: string,
    private gguf: GgufFile,
    private onProgress?: ProgressCallback,
  ) {}

  async bytes(name: string): Promise<Uint8Array> {
    const tensor = this.tensor(name);
    const buf = await fetchCachedRange(
      this.url,
      this.gguf.dataOffset + tensor.offset,
      tensor.nBytes,
      this.onProgress,
    );
    return new Uint8Array(buf);
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
    const tensor = this.tensor(name);
    const kind = ggufLinearKind(tensor);
    const raw = await this.bytes(name);
    const bytes = kind === "gguf-q8_0" ? packQ8_0ForGpu(raw, k * n) : raw;
    const block = kind === "gguf-q8_0" ? 32 : 256;
    return {
      kind,
      q: createBufferFromBytes(device, bytes, usage, `lfm.${name}`),
      params: createUniformU32(device, [k, n, k / block, Math.min(n, 32768)], `${name}.params`),
      k,
      n,
      blocks: k / block,
    };
  }

  async embeddingWeight(device: GPUDevice, usage: GPUBufferUsageFlags): Promise<EmbeddingWeight> {
    const name = "token_embd.weight";
    const tensor = this.tensor(name);
    const kind = ggufLinearKind(tensor);
    const raw = await this.bytes(name);
    const bytes = kind === "gguf-q8_0" ? packQ8_0ForGpu(raw, tensor.nElements) : raw;
    const block = kind === "gguf-q8_0" ? 32 : 256;
    return {
      kind,
      q: createBufferFromBytes(device, bytes, usage, "lfm.token_embd.weight"),
      params: createUniformU32(device, [0, 1024, 1024 / block], "lfm.embedding.params"),
    };
  }

  clear(): void {}

  private tensor(name: string): GgufTensorInfo {
    const tensor = this.gguf.tensorMap.get(name);
    if (!tensor) throw new Error(`LFM GGUF missing tensor ${name}`);
    return tensor;
  }

  private async floatBytes(name: string): Promise<Uint8Array> {
    const tensor = this.tensor(name);
    const bytes = await this.bytes(name);
    if (tensor.type === GGML_TYPE.F32) return bytes;
    if (tensor.type !== GGML_TYPE.F16) {
      throw new Error(`LFM GGUF tensor ${name} is ${ggmlTypeName(tensor.type)}, expected F32/F16`);
    }
    const src = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
    const dst = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) dst[i] = f16ToF32(src[i]);
    return new Uint8Array(dst.buffer);
  }
}

function ggufLinearKind(tensor: GgufTensorInfo): LinearKind & EmbeddingWeight["kind"] {
  if (tensor.type === GGML_TYPE.Q8_0) return "gguf-q8_0";
  if (tensor.type === GGML_TYPE.Q4_K) return "gguf-q4_k";
  if (tensor.type === GGML_TYPE.Q6_K) return "gguf-q6_k";
  throw new Error(`unsupported LFM GGUF linear tensor ${tensor.name}: ${ggmlTypeName(tensor.type)}`);
}

function packQ8_0ForGpu(bytes: Uint8Array, nElements: number): Uint8Array {
  if (nElements % 32 !== 0) throw new Error(`Q8_0 tensor element count is not divisible by 32: ${nElements}`);
  const blocks = nElements / 32;
  if (bytes.byteLength !== blocks * 34) {
    throw new Error(`Q8_0 tensor byte length mismatch: expected ${blocks * 34}, got ${bytes.byteLength}`);
  }
  const out = new ArrayBuffer(blocks * 36);
  const words = new Uint32Array(out);
  const floats = new Float32Array(out);
  let src = 0;
  for (let b = 0; b < blocks; b++) {
    const half = bytes[src] | (bytes[src + 1] << 8);
    floats[b * 9] = f16ToF32(half);
    src += 2;
    for (let i = 0; i < 8; i++) {
      let packed = 0;
      for (let j = 0; j < 4; j++) packed |= bytes[src + i * 4 + j] << (j * 8);
      words[b * 9 + 1 + i] = packed >>> 0;
    }
    src += 32;
  }
  return new Uint8Array(out);
}

class LfmWgslEngine {
  private readonly hidden = 1024;
  private readonly intermediate = 4608;
  private readonly heads = 16;
  private readonly maxContext = 4096;

  private pos = 0;
  private deviceLost = false;
  private readonly pipelines: Pipelines;
  private readonly bindGroups: BindGroups;
  private readonly tokenParams: GPUBuffer;
  private readonly posParams: GPUBuffer;
  private readonly normHiddenParams: GPUBuffer;
  private readonly addHiddenParams: GPUBuffer;
  private readonly siluParams: GPUBuffer;
  private readonly argmaxSizeParams: GPUBuffer;
  private readonly argmaxResult: GPUBuffer;
  private readonly argmaxReadback: GPUBuffer;
  private readonly topkResult: GPUBuffer;
  private readonly topkReadback: GPUBuffer;

  private readonly embeddingWeight: EmbeddingWeight;
  private readonly finalNorm: GPUBuffer;
  private readonly lmHead: LinearWeight;
  private readonly layers: Layer[];

  private readonly hiddenA: GPUBuffer;
  private readonly hiddenB: GPUBuffer;
  private readonly norm: GPUBuffer;
  private readonly proj3072: GPUBuffer;
  private readonly convMid: GPUBuffer;
  private readonly gate: GPUBuffer;
  private readonly up: GPUBuffer;
  private readonly ff: GPUBuffer;
  private readonly q: GPUBuffer;
  private readonly k: GPUBuffer;
  private readonly v: GPUBuffer;
  private readonly qNormed: GPUBuffer;
  private readonly attnOut: GPUBuffer;
  private readonly scores: GPUBuffer;
  private readonly logits: GPUBuffer;

  static async create(
    device: GPUDevice,
    manifest: LfmManifest,
    loader: LfmTensorSource,
    onStep?: LoadStepCallback,
  ): Promise<LfmWgslEngine> {
    onStep?.({ step: "shaders", detail: "compiling WebGPU shader pipelines" });
    const pipelines = createPipelines(device);
    onStep?.({ step: "weights", detail: "uploading embedding weights to GPU" });
    const usageRead = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const usageState = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const usageScratch = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

    const embeddingWeight = await loader.embeddingWeight(device, usageRead);
    const finalNorm = await loader.floatBuffer(device, "token_embd_norm.weight");
    const lmHead: LinearWeight = {
      ...embeddingWeight,
      params: createUniformU32(device, [1024, 65536, embeddingWeight.kind === "gguf-q8_0" ? 32 : 4, 32768], "lfm.lm_head.params"),
      k: 1024,
      n: 65536,
      blocks: embeddingWeight.kind === "gguf-q8_0" ? 32 : 4,
    };

    const linear = (name: string, k: number, n: number): Promise<LinearWeight> =>
      loader.linearWeight(device, name, k, n, usageRead);
    const name = {
      operatorNorm: (i: number) => `blk.${i}.attn_norm.weight`,
      ffnNorm: (i: number) => `blk.${i}.ffn_norm.weight`,
      convWeight: (i: number) => `blk.${i}.shortconv.conv.weight`,
      qNorm: (i: number) => `blk.${i}.attn_q_norm.weight`,
      kNorm: (i: number) => `blk.${i}.attn_k_norm.weight`,
      mlp: (i: number, proj: "gate" | "up" | "down") => `blk.${i}.ffn_${proj}.weight`,
      conv: (i: number, proj: "in" | "out") => `blk.${i}.shortconv.${proj}_proj.weight`,
      attn: (i: number, proj: "q" | "k" | "v" | "o") =>
        `blk.${i}.${proj === "o" ? "attn_output" : `attn_${proj}`}.weight`,
    };
    const mlp = async (i: number): Promise<MlpWeights> => {
      const [gate, up, down] = await Promise.all([
        linear(name.mlp(i, "gate"), 1024, 4608),
        linear(name.mlp(i, "up"), 1024, 4608),
        linear(name.mlp(i, "down"), 4608, 1024),
      ]);
      return { gate, up, down };
    };

    const loadLayer = async (i: number): Promise<Layer> => {
      const [operatorNorm, ffnNorm, mlpWeights] = await Promise.all([
        loader.floatBuffer(device, name.operatorNorm(i)),
        loader.floatBuffer(device, name.ffnNorm(i)),
        mlp(i),
      ]);
      const common = { operatorNorm, ffnNorm, mlp: mlpWeights };
      if (manifest.model.layerTypes[i] === "conv") {
        const [convWeight, convIn, convOut] = await Promise.all([
          loader.floatBuffer(device, name.convWeight(i)),
          linear(name.conv(i, "in"), 1024, 3072),
          linear(name.conv(i, "out"), 1024, 1024),
        ]);
        return {
          kind: "conv",
          ...common,
          convWeight,
          convIn,
          convOut,
          state: createEmptyBuffer(device, 1024 * 3 * 4, usageState, `lfm.layer.${i}.conv_state`),
        };
      }
      const [qNorm, kNorm, qProj, kProj, vProj, oProj] = await Promise.all([
        loader.floatBuffer(device, name.qNorm(i)),
        loader.floatBuffer(device, name.kNorm(i)),
        linear(name.attn(i, "q"), 1024, 1024),
        linear(name.attn(i, "k"), 1024, 512),
        linear(name.attn(i, "v"), 1024, 512),
        linear(name.attn(i, "o"), 1024, 1024),
      ]);
      return {
        kind: "attention",
        ...common,
        qNorm,
        kNorm,
        qProj,
        kProj,
        vProj,
        oProj,
        keyCache: createEmptyBuffer(device, 4096 * 8 * 64 * 4, usageState, `lfm.layer.${i}.k_cache`),
        valueCache: createEmptyBuffer(device, 4096 * 8 * 64 * 4, usageState, `lfm.layer.${i}.v_cache`),
      };
    };

    let completedLayers = 0;
    const layers = await mapWithConcurrency(
      Array.from({ length: manifest.model.layers }, (_, i) => i),
      4,
      async (i) => {
        const layer = await loadLayer(i);
        completedLayers++;
        onStep?.({
          step: "weights",
          detail: `uploaded layer ${completedLayers} / ${manifest.model.layers} to GPU`,
          progress: { current: completedLayers, total: manifest.model.layers },
        });
        return layer;
      },
    );

    onStep?.({ step: "weights", detail: "uploading lm head and scratch buffers" });
    return new LfmWgslEngine(device, pipelines, {
      embeddingWeight,
      finalNorm,
      lmHead,
      layers,
      tokenParams: createEmptyBuffer(device, 64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "lfm.token.params"),
      posParams: createEmptyBuffer(device, 64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "lfm.pos.params"),
      normHiddenParams: createUniformU32(device, [1024], "lfm.norm.hidden.params"),
      addHiddenParams: createUniformU32(device, [1024], "lfm.add.hidden.params"),
      siluParams: createUniformU32(device, [4608], "lfm.silu.params"),
      argmaxSizeParams: createUniformU32(device, [65536], "lfm.argmax.size"),
      argmaxResult: createEmptyBuffer(device, 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, "lfm.argmax.result"),
      argmaxReadback: createEmptyBuffer(device, 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, "lfm.argmax.readback"),
      topkResult: createEmptyBuffer(device, 256 * 2 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, "lfm.topk.result"),
      topkReadback: createEmptyBuffer(device, 256 * 2 * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, "lfm.topk.readback"),
      hiddenA: createEmptyBuffer(device, 1024 * 4, usageScratch, "lfm.hidden.a"),
      hiddenB: createEmptyBuffer(device, 1024 * 4, usageScratch, "lfm.hidden.b"),
      norm: createEmptyBuffer(device, 1024 * 4, usageScratch, "lfm.norm"),
      proj3072: createEmptyBuffer(device, 3072 * 4, usageScratch, "lfm.proj3072"),
      convMid: createEmptyBuffer(device, 1024 * 4, usageScratch, "lfm.conv_mid"),
      gate: createEmptyBuffer(device, 4608 * 4, usageScratch, "lfm.gate"),
      up: createEmptyBuffer(device, 4608 * 4, usageScratch, "lfm.up"),
      ff: createEmptyBuffer(device, 4608 * 4, usageScratch, "lfm.ff"),
      q: createEmptyBuffer(device, 1024 * 4, usageScratch, "lfm.q"),
      k: createEmptyBuffer(device, 512 * 4, usageScratch, "lfm.k"),
      v: createEmptyBuffer(device, 512 * 4, usageScratch, "lfm.v"),
      qNormed: createEmptyBuffer(device, 1024 * 4, usageScratch, "lfm.q_normed"),
      attnOut: createEmptyBuffer(device, 1024 * 4, usageScratch, "lfm.attn_out"),
      scores: createEmptyBuffer(device, 16 * 4096 * 4, usageScratch, "lfm.scores"),
      logits: createEmptyBuffer(device, 65536 * 4, usageScratch, "lfm.logits"),
    });
  }

  private constructor(
    private device: GPUDevice,
    pipelines: Pipelines,
    buffers: {
      embeddingWeight: EmbeddingWeight;
      finalNorm: GPUBuffer;
      lmHead: LinearWeight;
      layers: Layer[];
      tokenParams: GPUBuffer;
      posParams: GPUBuffer;
      normHiddenParams: GPUBuffer;
      addHiddenParams: GPUBuffer;
      siluParams: GPUBuffer;
      argmaxSizeParams: GPUBuffer;
      argmaxResult: GPUBuffer;
      argmaxReadback: GPUBuffer;
      topkResult: GPUBuffer;
      topkReadback: GPUBuffer;
      hiddenA: GPUBuffer;
      hiddenB: GPUBuffer;
      norm: GPUBuffer;
      proj3072: GPUBuffer;
      convMid: GPUBuffer;
      gate: GPUBuffer;
      up: GPUBuffer;
      ff: GPUBuffer;
      q: GPUBuffer;
      k: GPUBuffer;
      v: GPUBuffer;
      qNormed: GPUBuffer;
      attnOut: GPUBuffer;
      scores: GPUBuffer;
      logits: GPUBuffer;
    },
  ) {
    this.pipelines = pipelines;
    this.embeddingWeight = buffers.embeddingWeight;
    this.finalNorm = buffers.finalNorm;
    this.lmHead = buffers.lmHead;
    this.layers = buffers.layers;
    this.tokenParams = buffers.tokenParams;
    this.posParams = buffers.posParams;
    this.normHiddenParams = buffers.normHiddenParams;
    this.addHiddenParams = buffers.addHiddenParams;
    this.siluParams = buffers.siluParams;
    this.argmaxSizeParams = buffers.argmaxSizeParams;
    this.argmaxResult = buffers.argmaxResult;
    this.argmaxReadback = buffers.argmaxReadback;
    this.topkResult = buffers.topkResult;
    this.topkReadback = buffers.topkReadback;
    this.hiddenA = buffers.hiddenA;
    this.hiddenB = buffers.hiddenB;
    this.norm = buffers.norm;
    this.proj3072 = buffers.proj3072;
    this.convMid = buffers.convMid;
    this.gate = buffers.gate;
    this.up = buffers.up;
    this.ff = buffers.ff;
    this.q = buffers.q;
    this.k = buffers.k;
    this.v = buffers.v;
    this.qNormed = buffers.qNormed;
    this.attnOut = buffers.attnOut;
    this.scores = buffers.scores;
    this.logits = buffers.logits;
    this.bindGroups = this.createBindGroups();
    this.device.lost.then((info) => {
      this.deviceLost = true;
      console.error(`LFM WGSL WebGPU device lost: ${info.message} (${info.reason})`);
    });
  }

  reset(): void {
    this.pos = 0;
    const enc = this.device.createCommandEncoder();
    for (const layer of this.layers) {
      if (layer.kind === "conv") {
        enc.clearBuffer(layer.state);
      } else {
        enc.clearBuffer(layer.keyCache);
        enc.clearBuffer(layer.valueCache);
      }
    }
    this.device.queue.submit([enc.finish()]);
  }

  get position(): number {
    return this.pos;
  }

  async runToken(tokenId: number, needLogits: boolean, sampling?: SamplingOptions): Promise<number> {
    if (this.deviceLost) {
      throw new Error("LFM WGSL WebGPU device was lost; reload the model to continue");
    }
    if (this.pos >= this.maxContext) {
      throw new Error(`LFM WGSL context exhausted at ${this.maxContext} tokens`);
    }

    this.device.queue.writeBuffer(this.tokenParams, 0, new Uint32Array([tokenId, 32]));
    this.device.queue.writeBuffer(this.embeddingWeight.params, 0, new Uint32Array([tokenId]));
    this.device.queue.writeBuffer(this.posParams, 0, new Uint32Array([this.pos, this.maxContext]));

    const encoder = this.device.createCommandEncoder();
    this.dispatch(
      encoder,
      embeddingPipeline(this.pipelines, this.embeddingWeight.kind),
      this.bindGroups.embedding,
      Math.ceil(this.hidden / 128),
    );

    for (let i = 0; i < this.layers.length; i++) {
      const layer = this.layers[i];
      const bg = this.bindGroups.layers[i];
      this.dispatch(encoder, this.pipelines.rmsNorm, bg.operatorNorm, 1);
      if (layer.kind === "conv" && bg.kind === "conv") {
        this.q4MatvecBound(encoder, layer.convIn, bg.convIn);
        this.dispatch(encoder, this.pipelines.convBlock, bg.convBlock, Math.ceil(this.hidden / 128));
        this.q4MatvecBound(encoder, layer.convOut, bg.convOut);
        this.dispatch(encoder, this.pipelines.addRmsNorm, bg.opAddNorm, 1);
      } else {
        const attnLayer = layer as AttentionLayer;
        const attnBg = bg as AttentionLayerBindGroups;
        this.q4MatvecBatch(encoder, [
          { weight: attnLayer.qProj, bindGroup: attnBg.qProj },
          { weight: attnLayer.kProj, bindGroup: attnBg.kProj },
          { weight: attnLayer.vProj, bindGroup: attnBg.vProj },
        ]);
        this.dispatch(encoder, this.pipelines.qkNormRopeStore, attnBg.qkNormRopeStore, this.heads);
        this.dispatch(encoder, this.pipelines.attentionScore, attnBg.attentionScore, this.heads, this.pos + 1);
        this.dispatch(encoder, this.pipelines.attentionValue, attnBg.attentionValue, this.heads);
        this.q4MatvecBound(encoder, attnLayer.oProj, attnBg.oProj);
        this.dispatch(encoder, this.pipelines.addRmsNorm, attnBg.opAddNorm, 1);
      }

      this.q4MatvecBatch(encoder, [
        { weight: layer.mlp.gate, bindGroup: bg.mlpGate },
        { weight: layer.mlp.up, bindGroup: bg.mlpUp },
      ]);
      this.dispatch(encoder, this.pipelines.siluMul, bg.silu, Math.ceil(this.intermediate / 128));
      this.q4MatvecBound(encoder, layer.mlp.down, bg.mlpDown);
      this.dispatch(encoder, this.pipelines.add, bg.ffnAdd, Math.ceil(this.hidden / 128));
    }

    let best = 0;
    if (needLogits) {
      this.dispatch(encoder, this.pipelines.rmsNorm, this.bindGroups.finalNorm, 1);
      this.q4MatvecBound(encoder, this.lmHead, this.bindGroups.lmHead);
      if (needsCandidateReadback(sampling)) {
        this.dispatch(encoder, this.pipelines.topk256, this.bindGroups.topk256, 1);
        encoder.copyBufferToBuffer(this.topkResult, 0, this.topkReadback, 0, 256 * 2 * 4);
      } else {
        this.dispatch(encoder, this.pipelines.argmax, this.bindGroups.argmax, 1);
        encoder.copyBufferToBuffer(this.argmaxResult, 0, this.argmaxReadback, 0, 4);
      }
    }

    this.device.queue.submit([encoder.finish()]);
    this.pos += 1;
    if (!needLogits) return best;

    if (needsCandidateReadback(sampling)) {
      await this.topkReadback.mapAsync(GPUMapMode.READ);
      best = sampleFromCandidateBuffer(this.topkReadback.getMappedRange(), sampling, 256);
      this.topkReadback.unmap();
      return best;
    }

    await this.argmaxReadback.mapAsync(GPUMapMode.READ);
    best = new Uint32Array(this.argmaxReadback.getMappedRange())[0];
    this.argmaxReadback.unmap();
    return best;
  }

  dispose(): void {
    const destroyWeight = (w: LinearWeight | EmbeddingWeight) => {
      w.q.destroy();
      if ("scales" in w) w.scales?.destroy();
      if ("zp" in w) w.zp?.destroy();
      w.params.destroy();
    };
    destroyWeight(this.embeddingWeight);
    this.lmHead.params.destroy();
    this.finalNorm.destroy();
    for (const layer of this.layers) {
      layer.operatorNorm.destroy();
      layer.ffnNorm.destroy();
      destroyWeight(layer.mlp.gate);
      destroyWeight(layer.mlp.up);
      destroyWeight(layer.mlp.down);
      if (layer.kind === "conv") {
        layer.convWeight.destroy();
        destroyWeight(layer.convIn);
        destroyWeight(layer.convOut);
        layer.state.destroy();
      } else {
        layer.qNorm.destroy();
        layer.kNorm.destroy();
        destroyWeight(layer.qProj);
        destroyWeight(layer.kProj);
        destroyWeight(layer.vProj);
        destroyWeight(layer.oProj);
        layer.keyCache.destroy();
        layer.valueCache.destroy();
      }
    }
    for (const b of [
      this.tokenParams,
      this.posParams,
      this.normHiddenParams,
      this.addHiddenParams,
      this.siluParams,
      this.argmaxSizeParams,
      this.argmaxResult,
      this.argmaxReadback,
      this.topkResult,
      this.topkReadback,
      this.hiddenA,
      this.hiddenB,
      this.norm,
      this.proj3072,
      this.convMid,
      this.gate,
      this.up,
      this.ff,
      this.q,
      this.k,
      this.v,
      this.qNormed,
      this.attnOut,
      this.scores,
      this.logits,
    ]) {
      b.destroy();
    }
  }

  private createBindGroups(): BindGroups {
    const bind = (pipeline: GPUComputePipeline, entries: GPUBindGroupEntry[]) =>
      this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    const rms = (input: GPUBuffer, weight: GPUBuffer, output: GPUBuffer, params: GPUBuffer) =>
      bind(this.pipelines.rmsNorm, [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: weight } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: params } },
      ]);
    const linear = (input: GPUBuffer, weight: LinearWeight, output: GPUBuffer) => {
      return bind(linearPipeline(this.pipelines, weight.kind), [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: weight.q } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: weight.params } },
      ]);
    };
    const add = (a: GPUBuffer, bOut: GPUBuffer) =>
      bind(this.pipelines.add, [
        { binding: 0, resource: { buffer: a } },
        { binding: 1, resource: { buffer: bOut } },
        { binding: 2, resource: { buffer: this.addHiddenParams } },
      ]);
    const addNorm = (a: GPUBuffer, bOut: GPUBuffer, weight: GPUBuffer, output: GPUBuffer) =>
      bind(this.pipelines.addRmsNorm, [
        { binding: 0, resource: { buffer: a } },
        { binding: 1, resource: { buffer: bOut } },
        { binding: 2, resource: { buffer: weight } },
        { binding: 3, resource: { buffer: output } },
        { binding: 4, resource: { buffer: this.normHiddenParams } },
      ]);
    const silu = bind(this.pipelines.siluMul, [
      { binding: 0, resource: { buffer: this.gate } },
      { binding: 1, resource: { buffer: this.up } },
      { binding: 2, resource: { buffer: this.ff } },
      { binding: 3, resource: { buffer: this.siluParams } },
    ]);

    const layers: LayerBindGroups[] = this.layers.map((layer) => {
      const common = {
        operatorNorm: rms(this.hiddenA, layer.operatorNorm, this.norm, this.normHiddenParams),
        opAddNorm: addNorm(this.hiddenA, this.hiddenB, layer.ffnNorm, this.norm),
        mlpGate: linear(this.norm, layer.mlp.gate, this.gate),
        mlpUp: linear(this.norm, layer.mlp.up, this.up),
        silu,
        mlpDown: linear(this.ff, layer.mlp.down, this.hiddenA),
        ffnAdd: add(this.hiddenB, this.hiddenA),
      };
      if (layer.kind === "conv") {
        return {
          kind: "conv",
          ...common,
          convIn: linear(this.norm, layer.convIn, this.proj3072),
          convBlock: bind(this.pipelines.convBlock, [
            { binding: 0, resource: { buffer: this.proj3072 } },
            { binding: 1, resource: { buffer: layer.state } },
            { binding: 2, resource: { buffer: layer.convWeight } },
            { binding: 3, resource: { buffer: this.convMid } },
          ]),
          convOut: linear(this.convMid, layer.convOut, this.hiddenB),
          opAdd: add(this.hiddenA, this.hiddenB),
        };
      }
      return {
        kind: "attention",
        ...common,
        qProj: linear(this.norm, layer.qProj, this.q),
        kProj: linear(this.norm, layer.kProj, this.k),
        vProj: linear(this.norm, layer.vProj, this.v),
        qkNormRopeStore: bind(this.pipelines.qkNormRopeStore, [
          { binding: 0, resource: { buffer: this.q } },
          { binding: 1, resource: { buffer: this.k } },
          { binding: 2, resource: { buffer: this.v } },
          { binding: 3, resource: { buffer: layer.qNorm } },
          { binding: 4, resource: { buffer: layer.kNorm } },
          { binding: 5, resource: { buffer: this.qNormed } },
          { binding: 6, resource: { buffer: layer.keyCache } },
          { binding: 7, resource: { buffer: layer.valueCache } },
          { binding: 8, resource: { buffer: this.posParams } },
        ]),
        attentionScore: bind(this.pipelines.attentionScore, [
          { binding: 0, resource: { buffer: this.qNormed } },
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
        oProj: linear(this.attnOut, layer.oProj, this.hiddenB),
        opAdd: add(this.hiddenA, this.hiddenB),
      };
    });

    const embedding = (() => {
      const w = this.embeddingWeight;
      return bind(embeddingPipeline(this.pipelines, w.kind), [
        { binding: 0, resource: { buffer: w.q } },
        { binding: 1, resource: { buffer: this.hiddenA } },
        { binding: 2, resource: { buffer: w.params } },
      ]);
    })();

    return {
      embedding,
      finalNorm: rms(this.hiddenA, this.finalNorm, this.norm, this.normHiddenParams),
      lmHead: linear(this.norm, this.lmHead, this.logits),
      argmax: bind(this.pipelines.argmax, [
        { binding: 0, resource: { buffer: this.logits } },
        { binding: 1, resource: { buffer: this.argmaxResult } },
        { binding: 2, resource: { buffer: this.argmaxSizeParams } },
      ]),
      topk256: bind(this.pipelines.topk256, [
        { binding: 0, resource: { buffer: this.logits } },
        { binding: 1, resource: { buffer: this.topkResult } },
        { binding: 2, resource: { buffer: this.argmaxSizeParams } },
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
    z = 1,
  ): void {
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(x, y, z);
    pass.end();
  }

  private q4MatvecBound(encoder: GPUCommandEncoder, weight: LinearWeight, bindGroup: GPUBindGroup): void {
    this.q4MatvecBatch(encoder, [{ weight, bindGroup }]);
  }

  private q4MatvecBatch(
    encoder: GPUCommandEncoder,
    items: Array<{ weight: LinearWeight; bindGroup: GPUBindGroup }>,
  ): void {
    if (items.length === 0) return;
    for (const { weight, bindGroup } of items) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(linearPipeline(this.pipelines, weight.kind));
      const xGroups = Math.min(weight.n, 32768);
      const yGroups = Math.ceil(weight.n / xGroups);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(xGroups, yGroups);
      pass.end();
    }
  }
}

function commonPrefixLength(a: readonly number[], b: readonly number[]): number {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a[i] === b[i]) i++;
  return i;
}

function f16ToF32(h: number): number {
  const sign = (h & 0x8000) ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x03ff;
  if (exp === 0) return sign * Math.pow(2, -14) * (frac / 1024);
  if (exp === 0x1f) return frac ? NaN : sign * Infinity;
  return sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

function createPipelines(device: GPUDevice): Pipelines {
  const module = (label: string, code: string) => device.createShaderModule({ label, code });
  const pipeline = (label: string, code: string) =>
    device.createComputePipeline({
      label,
      layout: "auto",
      compute: { module: module(label, code), entryPoint: "main" },
    });

  return {
    embedding: pipeline("lfm.embedding", EMBEDDING_WGSL),
    embeddingQ8: pipeline("lfm.embedding.q8_0", EMBEDDING_Q8_WGSL),
    embeddingQ4K: pipeline("lfm.embedding.q4_k", EMBEDDING_Q4K_WGSL),
    embeddingQ6K: pipeline("lfm.embedding.q6_k", EMBEDDING_Q6K_WGSL),
    rmsNorm: pipeline("lfm.rms_norm", RMS_NORM_WGSL),
    addRmsNorm: pipeline("lfm.add_rms_norm", ADD_RMS_NORM_WGSL),
    q4Matvec: pipeline("lfm.q4_matvec", Q4_MATVEC_WGSL),
    q8Matvec: pipeline("lfm.q8_matvec", Q8_MATVEC_WGSL),
    q4KMatvec: pipeline("lfm.q4k_matvec", Q4K_MATVEC_WGSL),
    q6KMatvec: pipeline("lfm.q6k_matvec", Q6K_MATVEC_WGSL),
    convBlock: pipeline("lfm.conv_block", CONV_BLOCK_WGSL),
    add: pipeline("lfm.add", ADD_WGSL),
    siluMul: pipeline("lfm.silu_mul", SILU_MUL_WGSL),
    qkNormRopeStore: pipeline("lfm.qk_norm_rope_store", QK_NORM_ROPE_STORE_WGSL),
    attentionScore: pipeline("lfm.attention_score", ATTENTION_SCORE_WGSL),
    attentionValue: pipeline("lfm.attention_value", ATTENTION_VALUE_WGSL),
    argmax: pipeline("lfm.argmax", ARGMAX_WGSL),
    topk256: pipeline("lfm.topk256", TOPK256_WGSL),
  };
}

function linearPipeline(pipelines: Pipelines, kind: LinearKind): GPUComputePipeline {
  if (kind === "gguf-q8_0") return pipelines.q8Matvec;
  if (kind === "gguf-q4_k") return pipelines.q4KMatvec;
  return pipelines.q6KMatvec;
}

function embeddingPipeline(pipelines: Pipelines, kind: EmbeddingWeight["kind"]): GPUComputePipeline {
  if (kind === "gguf-q8_0") return pipelines.embeddingQ8;
  if (kind === "gguf-q4_k") return pipelines.embeddingQ4K;
  return pipelines.embeddingQ6K;
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

const Q4_MATVEC_WGSL = /* wgsl */ `
struct Params {
  k: u32,
  n: u32,
  blocks: u32,
  x_groups: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> q: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> zp: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> partial: array<f32, 128>;

fn q_byte_at(index: u32) -> u32 {
  let word = q[index >> 2u];
  let shift = (index & 3u) * 8u;
  return (word >> shift) & 0xffu;
}

fn zp_byte_at(index: u32) -> u32 {
  let word = zp[index >> 2u];
  let shift = (index & 3u) * 8u;
  return (word >> shift) & 0xffu;
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = wg.x + wg.y * params.x_groups;
  if (n >= params.n) {
    return;
  }

  var sum = 0.0;
  for (var k = lid.x; k < params.k; k = k + 128u) {
    let block = k >> 5u;
    let within = k & 31u;
    let q_index = n * params.blocks * 16u + block * 16u + (within >> 1u);
    let q_byte = q_byte_at(q_index);
    let qv = select(q_byte >> 4u, q_byte & 15u, (within & 1u) == 0u);

    let zp_index = n * (params.blocks >> 1u) + (block >> 1u);
    let zp_byte = zp_byte_at(zp_index);
    let zv = select(zp_byte >> 4u, zp_byte & 15u, (block & 1u) == 0u);
    let w = f32(i32(qv) - i32(zv)) * scales[n * params.blocks + block];
    sum = sum + input[k] * w;
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

const EMBEDDING_WGSL = /* wgsl */ `
struct Params {
  token: u32,
  blocks: u32,
};

@group(0) @binding(0) var<storage, read> q: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> zp: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

fn q_byte_at(index: u32) -> u32 {
  let word = q[index >> 2u];
  let shift = (index & 3u) * 8u;
  return (word >> shift) & 0xffu;
}

fn zp_byte_at(index: u32) -> u32 {
  let word = zp[index >> 2u];
  let shift = (index & 3u) * 8u;
  return (word >> shift) & 0xffu;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= 1024u) {
    return;
  }
  let block = k >> 5u;
  let within = k & 31u;
  let q_index = params.token * params.blocks * 16u + block * 16u + (within >> 1u);
  let q_byte = q_byte_at(q_index);
  let qv = select(q_byte >> 4u, q_byte & 15u, (within & 1u) == 0u);
  let zp_index = params.token * (params.blocks >> 1u) + (block >> 1u);
  let zp_byte = zp_byte_at(zp_index);
  let zv = select(zp_byte >> 4u, zp_byte & 15u, (block & 1u) == 0u);
  output[k] = f32(i32(qv) - i32(zv)) * scales[params.token * params.blocks + block];
}
`;

const Q8_MATVEC_WGSL = /* wgsl */ `
struct Params {
  k: u32,
  n: u32,
  blocks: u32,
  x_groups: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> q: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partial: array<f32, 128>;

fn q8_at(base: u32, index: u32) -> f32 {
  let word = q[base + 1u + (index >> 2u)];
  let shift = (index & 3u) * 8u;
  return f32(extractBits(bitcast<i32>(word), shift, 8u));
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = wg.x + wg.y * params.x_groups;
  if (n >= params.n) {
    return;
  }

  var sum = 0.0;
  for (var k = lid.x; k < params.k; k = k + 128u) {
    let block = k >> 5u;
    let within = k & 31u;
    let base = n * params.blocks * 9u + block * 9u;
    sum = sum + input[k] * q8_at(base, within) * bitcast<f32>(q[base]);
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

const EMBEDDING_Q8_WGSL = /* wgsl */ `
struct Params {
  token: u32,
  k: u32,
  blocks: u32,
};

@group(0) @binding(0) var<storage, read> q: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn q8_at(base: u32, index: u32) -> f32 {
  let word = q[base + 1u + (index >> 2u)];
  let shift = (index & 3u) * 8u;
  return f32(extractBits(bitcast<i32>(word), shift, 8u));
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= params.k) {
    return;
  }
  let block = k >> 5u;
  let within = k & 31u;
  let base = params.token * params.blocks * 9u + block * 9u;
  output[k] = q8_at(base, within) * bitcast<f32>(q[base]);
}
`;

const GGUF_K_QUANT_WGSL = /* wgsl */ `
fn byte_at(index: u32) -> u32 {
  let word = q[index >> 2u];
  let shift = (index & 3u) * 8u;
  return (word >> shift) & 0xffu;
}

fn sbyte_at(index: u32) -> i32 {
  let b = byte_at(index);
  if (b >= 128u) {
    return i32(b) - 256;
  }
  return i32(b);
}

fn f16_at(index: u32) -> f32 {
  let h = byte_at(index) | (byte_at(index + 1u) << 8u);
  let sign = select(1.0, -1.0, (h & 0x8000u) != 0u);
  let exp = (h >> 10u) & 0x1fu;
  let frac = h & 0x03ffu;
  if (exp == 0u) {
    if (frac == 0u) {
      return select(0.0, -0.0, sign < 0.0);
    }
    return sign * exp2(-14.0) * (f32(frac) / 1024.0);
  }
  if (exp == 31u) {
    return sign * 3.402823e38;
  }
  return sign * exp2(f32(exp) - 15.0) * (1.0 + f32(frac) / 1024.0);
}

fn q4k_scale(scales_base: u32, group: u32) -> u32 {
  if (group < 4u) {
    return byte_at(scales_base + group) & 0x3fu;
  }
  let j = group - 4u;
  return (byte_at(scales_base + 8u + j) & 0x0fu) | ((byte_at(scales_base + j) >> 2u) & 0x30u);
}

fn q4k_min(scales_base: u32, group: u32) -> u32 {
  if (group < 4u) {
    return byte_at(scales_base + 4u + group) & 0x3fu;
  }
  let j = group - 4u;
  return (byte_at(scales_base + 8u + j) >> 4u) | ((byte_at(scales_base + 4u + j) >> 2u) & 0x30u);
}

fn dequant_q4k(block_base: u32, within: u32) -> f32 {
  let d = f16_at(block_base);
  let dmin = f16_at(block_base + 2u);
  let group = within >> 5u;
  let elem = within & 31u;
  let scales_base = block_base + 4u;
  let qs_base = block_base + 16u;
  let qb = byte_at(qs_base + (group >> 1u) * 32u + elem);
  let qv = select(qb >> 4u, qb & 15u, (group & 1u) == 0u);
  return d * f32(q4k_scale(scales_base, group)) * f32(qv) -
    dmin * f32(q4k_min(scales_base, group));
}

fn dequant_q6k(block_base: u32, within: u32) -> f32 {
  let low = (byte_at(block_base + (within >> 7u) * 64u + (within & 63u)) >> (((within >> 6u) & 1u) * 4u)) & 15u;
  let high = (byte_at(block_base + 128u + (within >> 7u) * 32u + (within & 31u)) >> (((within >> 5u) & 3u) * 2u)) & 3u;
  let qv = i32(low | (high << 4u)) - 32;
  let scale = sbyte_at(block_base + 192u + (within >> 4u));
  let d = f16_at(block_base + 208u);
  return d * f32(scale) * f32(qv);
}
`;

const Q4K_MATVEC_WGSL = /* wgsl */ `
struct Params {
  k: u32,
  n: u32,
  blocks: u32,
  x_groups: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> q: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partial: array<f32, 128>;

${GGUF_K_QUANT_WGSL}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = wg.x + wg.y * params.x_groups;
  if (n >= params.n) {
    return;
  }

  var sum = 0.0;
  for (var k = lid.x; k < params.k; k = k + 128u) {
    let block = k >> 8u;
    let within = k & 255u;
    let block_base = n * params.blocks * 144u + block * 144u;
    sum = sum + input[k] * dequant_q4k(block_base, within);
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

const Q6K_MATVEC_WGSL = /* wgsl */ `
struct Params {
  k: u32,
  n: u32,
  blocks: u32,
  x_groups: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> q: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partial: array<f32, 128>;

${GGUF_K_QUANT_WGSL}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = wg.x + wg.y * params.x_groups;
  if (n >= params.n) {
    return;
  }

  var sum = 0.0;
  for (var k = lid.x; k < params.k; k = k + 128u) {
    let block = k >> 8u;
    let within = k & 255u;
    let block_base = n * params.blocks * 210u + block * 210u;
    sum = sum + input[k] * dequant_q6k(block_base, within);
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

const EMBEDDING_Q4K_WGSL = /* wgsl */ `
struct Params {
  token: u32,
  k: u32,
  blocks: u32,
};

@group(0) @binding(0) var<storage, read> q: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

${GGUF_K_QUANT_WGSL}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= params.k) {
    return;
  }
  let block = k >> 8u;
  let within = k & 255u;
  let block_base = params.token * params.blocks * 144u + block * 144u;
  output[k] = dequant_q4k(block_base, within);
}
`;

const EMBEDDING_Q6K_WGSL = /* wgsl */ `
struct Params {
  token: u32,
  k: u32,
  blocks: u32,
};

@group(0) @binding(0) var<storage, read> q: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

${GGUF_K_QUANT_WGSL}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= params.k) {
    return;
  }
  let block = k >> 8u;
  let within = k & 255u;
  let block_base = params.token * params.blocks * 210u + block * 210u;
  output[k] = dequant_q6k(block_base, within);
}
`;

const RMS_NORM_WGSL = /* wgsl */ `
struct Params {
  group_size: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let base = wg.x * params.group_size;
  var sum = 0.0;
  for (var i = lid.x; i < params.group_size; i = i + 256u) {
    let v = input[base + i];
    sum = sum + v * v;
  }
  partial[lid.x] = sum;
  workgroupBarrier();

  var stride = 128u;
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

  let inv = inverseSqrt(partial[0] / f32(params.group_size) + 0.00001);
  for (var i = lid.x; i < params.group_size; i = i + 256u) {
    output[base + i] = input[base + i] * inv * weight[i];
  }
}
`;

const ADD_RMS_NORM_WGSL = /* wgsl */ `
struct Params {
  group_size: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b_out: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  var sum = 0.0;
  for (var i = lid.x; i < params.group_size; i = i + 256u) {
    let v = a[i] + b_out[i];
    b_out[i] = v;
    sum = sum + v * v;
  }
  partial[lid.x] = sum;
  workgroupBarrier();

  var stride = 128u;
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

  let inv = inverseSqrt(partial[0] / f32(params.group_size) + 0.00001);
  for (var i = lid.x; i < params.group_size; i = i + 256u) {
    output[i] = b_out[i] * inv * weight[i];
  }
}
`;

const CONV_BLOCK_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> proj: array<f32>;
@group(0) @binding(1) var<storage, read_write> state: array<f32>;
@group(0) @binding(2) var<storage, read> conv_weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h = gid.x;
  if (h >= 1024u) {
    return;
  }
  let a = proj[h];
  let b = proj[1024u + h];
  let c = proj[2048u + h];
  let cur = a * c;

  let s = h * 3u;
  let p1 = state[s + 1u];
  let p2 = state[s + 2u];
  let w = h * 3u;
  let conv = p1 * conv_weight[w] + p2 * conv_weight[w + 1u] + cur * conv_weight[w + 2u];

  state[s] = p1;
  state[s + 1u] = p2;
  state[s + 2u] = cur;
  output[h] = b * conv;
}
`;

const ADD_WGSL = /* wgsl */ `
struct Params {
  n: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b_out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < params.n) {
    b_out[i] = a[i] + b_out[i];
  }
}
`;

const SILU_MUL_WGSL = /* wgsl */ `
struct Params {
  n: u32,
};

@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < params.n) {
    let g = gate[i];
    output[i] = (g / (1.0 + exp(-g))) * up[i];
  }
}
`;

const QK_NORM_ROPE_STORE_WGSL = /* wgsl */ `
struct Params {
  pos: u32,
  max_context: u32,
};

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> q_weight: array<f32>;
@group(0) @binding(4) var<storage, read> k_weight: array<f32>;
@group(0) @binding(5) var<storage, read_write> q_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> key_cache: array<f32>;
@group(0) @binding(7) var<storage, read_write> value_cache: array<f32>;
@group(0) @binding(8) var<uniform> params: Params;

var<workgroup> q_partial: array<f32, 64>;
var<workgroup> k_partial: array<f32, 64>;

fn rope_angle(d: u32) -> f32 {
  let inv = pow(1000000.0, -2.0 * f32(d) / 64.0);
  return f32(params.pos) * inv;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let q_head = wg.x;
  let kv_head = q_head >> 1u;
  let d = lid.x;
  let q_base = q_head * 64u;
  let k_base = kv_head * 64u;

  let qv = q[q_base + d];
  q_partial[d] = qv * qv;
  if ((q_head & 1u) == 0u) {
    let kv = k[k_base + d];
    k_partial[d] = kv * kv;
  } else {
    k_partial[d] = 0.0;
  }
  workgroupBarrier();

  var stride = 32u;
  loop {
    if (d < stride) {
      q_partial[d] = q_partial[d] + q_partial[d + stride];
      k_partial[d] = k_partial[d] + k_partial[d + stride];
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride = stride >> 1u;
  }

  let q_inv = inverseSqrt(q_partial[0] / 64.0 + 0.00001);
  if (d < 32u) {
    let a = q[q_base + d] * q_inv * q_weight[d];
    let b = q[q_base + 32u + d] * q_inv * q_weight[32u + d];
    let angle = rope_angle(d);
    let co = cos(angle);
    let si = sin(angle);
    q_out[q_base + d] = a * co - b * si;
    q_out[q_base + 32u + d] = a * si + b * co;
  }

  if ((q_head & 1u) == 0u) {
    let k_inv = inverseSqrt(k_partial[0] / 64.0 + 0.00001);
    if (d < 32u) {
      let a = k[k_base + d] * k_inv * k_weight[d];
      let b = k[k_base + 32u + d] * k_inv * k_weight[32u + d];
      let angle = rope_angle(d);
      let co = cos(angle);
      let si = sin(angle);
      let cache_base = (kv_head * params.max_context + params.pos) * 64u;
      key_cache[cache_base + d] = a * co - b * si;
      key_cache[cache_base + 32u + d] = a * si + b * co;
    }
    let cache_base = (kv_head * params.max_context + params.pos) * 64u;
    value_cache[cache_base + d] = v[k_base + d];
  }
}
`;

const ATTENTION_SCORE_WGSL = /* wgsl */ `
struct Params {
  pos: u32,
  max_context: u32,
};

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> key_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let head = wg.x;
  let t = wg.y;
  let kv_head = head >> 1u;
  let q_base = head * 64u;
  let k_base = (kv_head * params.max_context + t) * 64u;
  let d = lid.x;
  partial[d] = q[q_base + d] * key_cache[k_base + d];
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
    scores[head * params.max_context + t] = partial[0] * 0.125;
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

var<workgroup> max_score: f32;
var<workgroup> denom: f32;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let head = wg.x;
  let d = lid.x;
  let kv_head = head >> 1u;

  if (d == 0u) {
    var m = -3.402823e38;
    for (var t = 0u; t <= params.pos; t = t + 1u) {
      m = max(m, scores[head * params.max_context + t]);
    }
    max_score = m;

    var s = 0.0;
    for (var t = 0u; t <= params.pos; t = t + 1u) {
      s = s + exp(scores[head * params.max_context + t] - m);
    }
    denom = s;
  }
  workgroupBarrier();

  var acc = 0.0;
  for (var t = 0u; t <= params.pos; t = t + 1u) {
    let p = exp(scores[head * params.max_context + t] - max_score) / denom;
    let v_base = (kv_head * params.max_context + t) * 64u;
    acc = acc + p * value_cache[v_base + d];
  }
  output[head * 64u + d] = acc;
}
`;

const ARGMAX_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<u32>;
@group(0) @binding(2) var<uniform> size: u32;

var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_idx: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  var local_max = -3.402823e38;
  var local_idx = 0u;
  var i = tid;
  while (i < size) {
    let v = logits[i];
    if (v > local_max) {
      local_max = v;
      local_idx = i;
    }
    i = i + 256u;
  }
  shared_max[tid] = local_max;
  shared_idx[tid] = local_idx;
  workgroupBarrier();

  var stride = 128u;
  loop {
    if (tid < stride && shared_max[tid + stride] > shared_max[tid]) {
      shared_max[tid] = shared_max[tid + stride];
      shared_idx[tid] = shared_idx[tid + stride];
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride = stride >> 1u;
  }

  if (tid == 0u) {
    result[0] = shared_idx[0];
  }
}
`;

const TOPK256_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<uniform> size: u32;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  var local_max = -3.402823e38;
  var local_idx = 0u;
  var i = tid;
  while (i < size) {
    let v = logits[i];
    if (v > local_max) {
      local_max = v;
      local_idx = i;
    }
    i = i + 256u;
  }
  result[tid * 2u] = local_max;
  result[tid * 2u + 1u] = bitcast<f32>(local_idx);
}
`;
