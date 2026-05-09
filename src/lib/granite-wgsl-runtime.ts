// Experimental hand-written WebGPU runtime for IBM Granite 4.0 H 350M.
//
// This runtime is intentionally narrow: it implements the official GGUF
// Q8_0 and Q4_K_M artifacts in models/granite-4.0-h-350m/gguf. All token
// execution is done through WGSL kernels.

import { fetchCachedRange, mapWithConcurrency, requestPersistence, type LoadStepCallback, type ProgressCallback } from "./cache";
import {
  fetchGgufHeader,
  GGML_TYPE,
  ggmlTypeName,
  type GgufFile,
} from "./gguf";
import type { ModelDef } from "./registry";
import { needsCandidateReadback, sampleFromCandidateBuffer, type SamplingOptions } from "./sampling";
import { Tokenizer } from "./tokenizer";

type TensorDtype = "float32" | "float16" | "uint8" | "int64" | "q8_0" | "q4_k" | "q6_k";

interface TensorSpec {
  dtype: TensorDtype;
  shape: number[];
  ggmlType?: number;
  external?: { path: string; offset: number; length: number };
  inlineBase64?: string;
}

interface GraniteManifest {
  format: "chonklm.granite-wgsl.v1";
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
    layerTypes: ("mamba" | "attention")[];
  };
  tensors: Record<string, TensorSpec>;
}

type LinearKind = "gguf-q8_0" | "gguf-q4_k" | "gguf-q6_k";

interface LinearWeight {
  kind: LinearKind;
  q: GPUBuffer;
  params: GPUBuffer;
  k: number;
  n: number;
  blocks: number;
}

interface MlpWeights {
  gate: LinearWeight;
  up: LinearWeight;
  down: LinearWeight;
}

interface CommonLayer {
  inputNorm: GPUBuffer;
  postNorm: GPUBuffer;
  mlp: MlpWeights;
}

interface MambaLayer extends CommonLayer {
  kind: "mamba";
  inProj: LinearWeight;
  outProj: LinearWeight;
  convWeight: GPUBuffer;
  convBias: GPUBuffer;
  dtBias: GPUBuffer;
  aNeg: GPUBuffer;
  d: GPUBuffer;
  mambaNorm: GPUBuffer;
  convState: GPUBuffer;
  ssmState: GPUBuffer;
}

interface AttentionLayer extends CommonLayer {
  kind: "attention";
  qProj: LinearWeight;
  kProj: LinearWeight;
  vProj: LinearWeight;
  oProj: LinearWeight;
  keyCache: GPUBuffer;
  valueCache: GPUBuffer;
}

type Layer = MambaLayer | AttentionLayer;

interface Pipelines {
  embeddingQ8: GPUComputePipeline;
  embeddingQ4K: GPUComputePipeline;
  embeddingQ6K: GPUComputePipeline;
  rmsNorm: GPUComputePipeline;
  addScaledRmsNorm: GPUComputePipeline;
  addScaled: GPUComputePipeline;
  q8Matvec: GPUComputePipeline;
  q4KMatvec: GPUComputePipeline;
  q6KMatvec: GPUComputePipeline;
  mambaConv: GPUComputePipeline;
  mambaSsm: GPUComputePipeline;
  mambaGateNorm: GPUComputePipeline;
  storeKv: GPUComputePipeline;
  attentionScore: GPUComputePipeline;
  attentionValue: GPUComputePipeline;
  siluMul: GPUComputePipeline;
  scaleLogits: GPUComputePipeline;
  argmax: GPUComputePipeline;
  topk256: GPUComputePipeline;
}

interface CommonLayerBindGroups {
  inputNorm: GPUBindGroup;
  postAddNorm: GPUBindGroup;
  mlpGate: GPUBindGroup;
  mlpUp: GPUBindGroup;
  silu: GPUBindGroup;
  mlpDown: GPUBindGroup;
  mlpAdd: GPUBindGroup;
}

interface MambaLayerBindGroups extends CommonLayerBindGroups {
  kind: "mamba";
  inProj: GPUBindGroup;
  conv: GPUBindGroup;
  ssm: GPUBindGroup;
  gateNorm: GPUBindGroup;
  outProj: GPUBindGroup;
}

interface AttentionLayerBindGroups extends CommonLayerBindGroups {
  kind: "attention";
  qProj: GPUBindGroup;
  kProj: GPUBindGroup;
  vProj: GPUBindGroup;
  storeKv: GPUBindGroup;
  attentionScore: GPUBindGroup;
  attentionValue: GPUBindGroup;
  oProj: GPUBindGroup;
}

type LayerBindGroups = MambaLayerBindGroups | AttentionLayerBindGroups;

interface BindGroups {
  embedding: GPUBindGroup;
  finalNorm: GPUBindGroup;
  lmHead: GPUBindGroup;
  scaleLogits: GPUBindGroup;
  argmax: GPUBindGroup;
  topk256: GPUBindGroup;
  layers: LayerBindGroups[];
}

export interface LoadedGraniteWgslModel {
  runtime: "granite-webgpu";
  def: ModelDef;
  tokenizer: Tokenizer;
  engine: GraniteWgslEngine;
  ep: "webgpu";
  cachedTokenIds: number[];
  cachedNextId: number | null;
}

export async function loadGraniteWgslModel(
  model: ModelDef,
  onProgress?: ProgressCallback,
  onStep?: LoadStepCallback,
): Promise<LoadedGraniteWgslModel> {
  if (model.runtime !== "granite-webgpu") {
    throw new Error(`loadGraniteWgslModel: unsupported model ${model.id}`);
  }
  if (!model.gguf) {
    throw new Error(`Granite WGSL model ${model.id} missing 'gguf' URL`);
  }
  if (!("gpu" in navigator)) {
    throw new Error("Granite WGSL runtime requires WebGPU");
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
  if (!adapter) throw new Error("Granite WGSL runtime could not acquire a WebGPU adapter");
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
    openGgufGraniteSource(model, onProgress),
  ]);

  const engine = await GraniteWgslEngine.create(device, source.manifest, source.loader, onStep);
  source.loader.clear();
  onStep?.({ step: "ready", detail: "Granite WebGPU runtime ready" });

  return {
    runtime: "granite-webgpu",
    def: model,
    tokenizer,
    engine,
    ep: "webgpu",
    cachedTokenIds: [],
    cachedNextId: null,
  };
}

export async function generateGraniteWgsl(
  loaded: LoadedGraniteWgslModel,
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

export function disposeGraniteWgsl(loaded: LoadedGraniteWgslModel): void {
  loaded.engine.dispose();
}

export function resetGraniteWgslConversation(loaded: LoadedGraniteWgslModel): void {
  loaded.engine.reset();
  loaded.cachedTokenIds = [];
  loaded.cachedNextId = null;
}

interface GraniteSource {
  manifest: GraniteManifest;
  loader: TensorLoader;
}

async function openGgufGraniteSource(model: ModelDef, onProgress?: ProgressCallback): Promise<GraniteSource> {
  if (!model.gguf) throw new Error(`Granite WGSL model ${model.id} missing 'gguf' URL`);
  const url = model.gguf.startsWith("http") || model.gguf.startsWith("/")
    ? model.gguf
    : `${model.base}/${model.gguf}`;
  const gguf = await fetchGgufHeader(url, onProgress);
  validateGraniteGguf(gguf, model.id);

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

  const layerTypes = model.layerTypes?.map((t) => t === "attention" ? "attention" : "mamba")
    ?? GRANITE_LAYER_TYPES;
  const manifest: GraniteManifest = {
    format: "chonklm.granite-wgsl.v1",
    source: url,
    model: {
      hidden: 768,
      intermediate: 2048,
      vocab: 100352,
      layers: 32,
      heads: 12,
      kvHeads: 4,
      headDim: 64,
      maxContext: model.maxContext,
      layerTypes,
    },
    tensors,
  };
  return { manifest, loader: new TensorLoader(model, manifest, onProgress) };
}

const GRANITE_LAYER_TYPES: ("mamba" | "attention")[] = [
  "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba",
  "mamba", "mamba", "attention", "mamba", "mamba", "attention", "mamba", "mamba",
  "mamba", "attention", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba",
  "mamba", "mamba", "mamba", "attention", "mamba", "mamba", "mamba", "mamba",
];

function validateGraniteGguf(gguf: GgufFile, modelId: string): void {
  const arch = gguf.kv.get("general.architecture")?.value;
  if (arch !== "granitehybrid") {
    throw new Error(`Granite WGSL ${modelId}: expected GGUF architecture granitehybrid, got ${String(arch)}`);
  }
  const fileType = Number(gguf.kv.get("general.file_type")?.value ?? -1);
  if (fileType !== 7 && fileType !== 15) {
    throw new Error(`Granite WGSL ${modelId}: expected GGUF Q8_0 or Q4_K_M file_type, got ${fileType}`);
  }
  for (const name of ["token_embd.weight", "output_norm.weight", "blk.0.attn_norm.weight"]) {
    if (!gguf.tensorMap.has(name)) throw new Error(`Granite WGSL ${modelId}: GGUF missing tensor ${name}`);
  }
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
    case GGML_TYPE.Q6_K:
      return "q6_k";
    default:
      throw new Error(`Granite WGSL unsupported GGML tensor type ${ggmlTypeName(type)}`);
  }
}

class TensorLoader {
  constructor(
    private model: ModelDef,
    private manifest: GraniteManifest,
    private onProgress?: ProgressCallback,
  ) {}

  async bytes(name: string): Promise<Uint8Array> {
    const spec = this.manifest.tensors[name];
    if (!spec) throw new Error(`Granite WGSL manifest missing tensor ${name}`);
    if (spec.inlineBase64) return base64Bytes(spec.inlineBase64);
    if (!spec.external) throw new Error(`Granite WGSL tensor ${name} has no payload`);
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

  async floatStorageBuffer(device: GPUDevice, name: string, usage: GPUBufferUsageFlags): Promise<GPUBuffer> {
    const bytes = await this.floatBytes(name);
    return createBufferFromBytes(device, bytes, usage, name);
  }

  async linearWeight(
    device: GPUDevice,
    name: string,
    k: number,
    n: number,
    usage: GPUBufferUsageFlags,
  ): Promise<LinearWeight> {
    const spec = this.manifest.tensors[name];
    if (!spec) throw new Error(`Granite WGSL manifest missing tensor ${name}`);
    const kind = linearKindForSpec(spec, name);
    validateMatrixShape(name, spec, k, n);
    const q = await this.buffer(device, name, usage);
    const block = kind === "gguf-q8_0" ? 32 : 256;
    return {
      kind,
      q,
      params: createUniformU32(device, [k, n, k / block, Math.min(n, 32768)], `${name}.params`),
      k,
      n,
      blocks: k / block,
    };
  }

  private async floatBytes(name: string): Promise<Uint8Array> {
    const spec = this.manifest.tensors[name];
    if (!spec) throw new Error(`Granite WGSL manifest missing tensor ${name}`);
    const bytes = await this.bytes(name);
    if (spec.dtype === "float32") return bytes;
    if (spec.dtype !== "float16") {
      throw new Error(`Granite WGSL tensor ${name} is ${spec.dtype}, expected float32/float16`);
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
    case "q6_k":
      return "gguf-q6_k";
    default:
      throw new Error(`Granite WGSL tensor ${name} is ${spec.dtype}, expected a GGUF quantized matrix`);
  }
}

function validateMatrixShape(name: string, spec: TensorSpec, k: number, n: number): void {
  if (spec.shape.length !== 2 || spec.shape[0] !== k || spec.shape[1] !== n) {
    throw new Error(`Granite WGSL tensor ${name} shape [${spec.shape.join(", ")}], expected [${k}, ${n}]`);
  }
}

class GraniteWgslEngine {
  private readonly hidden = 768;
  private readonly intermediate = 2048;
  private readonly heads = 12;
  private readonly kvHeads = 4;
  private readonly maxContext = 4096;
  private readonly mambaConvDim = 1792;
  private readonly mambaDt = 48;
  private readonly mambaDHead = 32;

  private pos = 0;
  private deviceLost = false;
  private readonly pipelines: Pipelines;
  private readonly bindGroups: BindGroups;
  private readonly tokenParams: GPUBuffer;
  private readonly posParams: GPUBuffer;
  private readonly normHiddenParams: GPUBuffer;
  private readonly normMambaParams: GPUBuffer;
  private readonly siluParams: GPUBuffer;
  private readonly addHiddenParams: GPUBuffer;
  private readonly argmaxSizeParams: GPUBuffer;
  private readonly argmaxResult: GPUBuffer;
  private readonly argmaxReadback: GPUBuffer;
  private readonly topkResult: GPUBuffer;
  private readonly topkReadback: GPUBuffer;

  private readonly embedding: LinearWeight;
  private readonly finalNorm: GPUBuffer;
  private readonly lmHead: LinearWeight;
  private readonly layers: Layer[];

  private readonly hiddenA: GPUBuffer;
  private readonly hiddenB: GPUBuffer;
  private readonly norm: GPUBuffer;
  private readonly mambaProj: GPUBuffer;
  private readonly mambaConvAct: GPUBuffer;
  private readonly mambaSsmY: GPUBuffer;
  private readonly mambaNormed: GPUBuffer;
  private readonly gate: GPUBuffer;
  private readonly up: GPUBuffer;
  private readonly ff: GPUBuffer;
  private readonly q: GPUBuffer;
  private readonly k: GPUBuffer;
  private readonly v: GPUBuffer;
  private readonly attnOut: GPUBuffer;
  private readonly scores: GPUBuffer;
  private readonly logits: GPUBuffer;

  static async create(
    device: GPUDevice,
    manifest: GraniteManifest,
    loader: TensorLoader,
    onStep?: LoadStepCallback,
  ): Promise<GraniteWgslEngine> {
    onStep?.({ step: "shaders", detail: "compiling WebGPU shader pipelines" });
    const pipelines = createPipelines(device);
    onStep?.({ step: "weights", detail: "uploading embedding weights to GPU" });
    const usageRead = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const usageState = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const usageScratch = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

    const embedding = await loader.linearWeight(device, "token_embd.weight", 768, 100352, usageRead);
    const finalNorm = await loader.floatBuffer(device, "output_norm.weight");
    const lmHead: LinearWeight = {
      kind: embedding.kind,
      q: embedding.q,
      params: createUniformU32(device, [768, 100352, embedding.blocks, 32768], "granite.lm_head.params"),
      k: 768,
      n: 100352,
      blocks: embedding.blocks,
    };

    const linear = (name: string, k: number, n: number): Promise<LinearWeight> =>
      loader.linearWeight(device, name, k, n, usageRead);
    const mlp = async (i: number): Promise<MlpWeights> => {
      const [gate, up, down] = await Promise.all([
        linear(`blk.${i}.ffn_gate.weight`, 768, 2048),
        linear(`blk.${i}.ffn_up.weight`, 768, 2048),
        linear(`blk.${i}.ffn_down.weight`, 2048, 768),
      ]);
      return { gate, up, down };
    };

    const loadLayer = async (i: number): Promise<Layer> => {
      const [inputNorm, postNorm, mlpWeights] = await Promise.all([
        loader.floatBuffer(device, `blk.${i}.attn_norm.weight`),
        loader.floatBuffer(device, `blk.${i}.ffn_norm.weight`),
        mlp(i),
      ]);
      const common = { inputNorm, postNorm, mlp: mlpWeights };
      if (manifest.model.layerTypes[i] === "mamba") {
        const [inProj, outProj, convWeight, convBias, dtBias, aNeg, d, mambaNorm] = await Promise.all([
          linear(`blk.${i}.ssm_in.weight`, 768, 3376),
          linear(`blk.${i}.ssm_out.weight`, 1536, 768),
          loader.floatBuffer(device, `blk.${i}.ssm_conv1d.weight`),
          loader.floatBuffer(device, `blk.${i}.ssm_conv1d.bias`),
          loader.floatBuffer(device, `blk.${i}.ssm_dt.bias`),
          loader.floatBuffer(device, `blk.${i}.ssm_a`),
          loader.floatBuffer(device, `blk.${i}.ssm_d`),
          loader.floatBuffer(device, `blk.${i}.ssm_norm.weight`),
        ]);
        return {
          kind: "mamba",
          ...common,
          inProj,
          outProj,
          convWeight,
          convBias,
          dtBias,
          aNeg,
          d,
          mambaNorm,
          convState: createEmptyBuffer(device, 1792 * 4 * 4, usageState, `granite.layer.${i}.conv_state`),
          ssmState: createEmptyBuffer(device, 48 * 32 * 128 * 4, usageState, `granite.layer.${i}.ssm_state`),
        };
      }
      const [qProj, kProj, vProj, oProj] = await Promise.all([
        linear(`blk.${i}.attn_q.weight`, 768, 768),
        linear(`blk.${i}.attn_k.weight`, 768, 256),
        linear(`blk.${i}.attn_v.weight`, 768, 256),
        linear(`blk.${i}.attn_output.weight`, 768, 768),
      ]);
      return {
        kind: "attention",
        ...common,
        qProj,
        kProj,
        vProj,
        oProj,
        keyCache: createEmptyBuffer(device, 4096 * 4 * 64 * 4, usageState, `granite.layer.${i}.k_cache`),
        valueCache: createEmptyBuffer(device, 4096 * 4 * 64 * 4, usageState, `granite.layer.${i}.v_cache`),
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
    return new GraniteWgslEngine(device, pipelines, {
      embedding,
      finalNorm,
      lmHead,
      layers,
      tokenParams: createEmptyBuffer(device, 64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "granite.token.params"),
      posParams: createEmptyBuffer(device, 64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "granite.pos.params"),
      normHiddenParams: createUniformU32(device, [768], "granite.norm.hidden.params"),
      normMambaParams: createUniformU32(device, [1536], "granite.norm.mamba.params"),
      siluParams: createUniformU32(device, [2048], "granite.silu.params"),
      addHiddenParams: createUniformU32(device, [768], "granite.add.hidden.params"),
      argmaxSizeParams: createUniformU32(device, [100352], "granite.argmax.size"),
      argmaxResult: createEmptyBuffer(device, 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, "granite.argmax.result"),
      argmaxReadback: createEmptyBuffer(device, 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, "granite.argmax.readback"),
      topkResult: createEmptyBuffer(device, 256 * 2 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, "granite.topk.result"),
      topkReadback: createEmptyBuffer(device, 256 * 2 * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, "granite.topk.readback"),
      hiddenA: createEmptyBuffer(device, 768 * 4, usageScratch, "granite.hidden.a"),
      hiddenB: createEmptyBuffer(device, 768 * 4, usageScratch, "granite.hidden.b"),
      norm: createEmptyBuffer(device, 2048 * 4, usageScratch, "granite.norm"),
      mambaProj: createEmptyBuffer(device, 3376 * 4, usageScratch, "granite.mamba_proj"),
      mambaConvAct: createEmptyBuffer(device, 1792 * 4, usageScratch, "granite.mamba_conv_act"),
      mambaSsmY: createEmptyBuffer(device, 1536 * 4, usageScratch, "granite.mamba_ssm_y"),
      mambaNormed: createEmptyBuffer(device, 1536 * 4, usageScratch, "granite.mamba_normed"),
      gate: createEmptyBuffer(device, 2048 * 4, usageScratch, "granite.gate"),
      up: createEmptyBuffer(device, 2048 * 4, usageScratch, "granite.up"),
      ff: createEmptyBuffer(device, 2048 * 4, usageScratch, "granite.ff"),
      q: createEmptyBuffer(device, 768 * 4, usageScratch, "granite.q"),
      k: createEmptyBuffer(device, 256 * 4, usageScratch, "granite.k"),
      v: createEmptyBuffer(device, 256 * 4, usageScratch, "granite.v"),
      attnOut: createEmptyBuffer(device, 768 * 4, usageScratch, "granite.attn_out"),
      scores: createEmptyBuffer(device, 12 * 4096 * 4, usageScratch, "granite.scores"),
      logits: createEmptyBuffer(device, 100352 * 4, usageScratch, "granite.logits"),
    });
  }

  private constructor(
    private device: GPUDevice,
    pipelines: Pipelines,
    buffers: {
      embedding: LinearWeight;
      finalNorm: GPUBuffer;
      lmHead: LinearWeight;
      layers: Layer[];
      tokenParams: GPUBuffer;
      posParams: GPUBuffer;
      normHiddenParams: GPUBuffer;
      normMambaParams: GPUBuffer;
      siluParams: GPUBuffer;
      addHiddenParams: GPUBuffer;
      argmaxSizeParams: GPUBuffer;
      argmaxResult: GPUBuffer;
      argmaxReadback: GPUBuffer;
      topkResult: GPUBuffer;
      topkReadback: GPUBuffer;
      hiddenA: GPUBuffer;
      hiddenB: GPUBuffer;
      norm: GPUBuffer;
      mambaProj: GPUBuffer;
      mambaConvAct: GPUBuffer;
      mambaSsmY: GPUBuffer;
      mambaNormed: GPUBuffer;
      gate: GPUBuffer;
      up: GPUBuffer;
      ff: GPUBuffer;
      q: GPUBuffer;
      k: GPUBuffer;
      v: GPUBuffer;
      attnOut: GPUBuffer;
      scores: GPUBuffer;
      logits: GPUBuffer;
    },
  ) {
    this.pipelines = pipelines;
    this.embedding = buffers.embedding;
    this.finalNorm = buffers.finalNorm;
    this.lmHead = buffers.lmHead;
    this.layers = buffers.layers;
    this.tokenParams = buffers.tokenParams;
    this.posParams = buffers.posParams;
    this.normHiddenParams = buffers.normHiddenParams;
    this.normMambaParams = buffers.normMambaParams;
    this.siluParams = buffers.siluParams;
    this.addHiddenParams = buffers.addHiddenParams;
    this.argmaxSizeParams = buffers.argmaxSizeParams;
    this.argmaxResult = buffers.argmaxResult;
    this.argmaxReadback = buffers.argmaxReadback;
    this.topkResult = buffers.topkResult;
    this.topkReadback = buffers.topkReadback;
    this.hiddenA = buffers.hiddenA;
    this.hiddenB = buffers.hiddenB;
    this.norm = buffers.norm;
    this.mambaProj = buffers.mambaProj;
    this.mambaConvAct = buffers.mambaConvAct;
    this.mambaSsmY = buffers.mambaSsmY;
    this.mambaNormed = buffers.mambaNormed;
    this.gate = buffers.gate;
    this.up = buffers.up;
    this.ff = buffers.ff;
    this.q = buffers.q;
    this.k = buffers.k;
    this.v = buffers.v;
    this.attnOut = buffers.attnOut;
    this.scores = buffers.scores;
    this.logits = buffers.logits;
    this.bindGroups = this.createBindGroups();
    this.device.lost.then((info) => {
      this.deviceLost = true;
      console.error(`Granite WGSL WebGPU device lost: ${info.message} (${info.reason})`);
    });
  }

  reset(): void {
    this.pos = 0;
    const enc = this.device.createCommandEncoder();
    for (const layer of this.layers) {
      if (layer.kind === "mamba") {
        enc.clearBuffer(layer.convState);
        enc.clearBuffer(layer.ssmState);
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
      throw new Error("Granite WGSL WebGPU device was lost; reload the model to continue");
    }
    if (this.pos >= this.maxContext) {
      throw new Error(`Granite WGSL context exhausted at ${this.maxContext} tokens`);
    }

    this.device.queue.writeBuffer(this.tokenParams, 0, new Uint32Array([tokenId, this.embedding.blocks]));
    this.device.queue.writeBuffer(this.posParams, 0, new Uint32Array([this.pos, this.maxContext]));

    const encoder = this.device.createCommandEncoder();
    this.dispatch(encoder, this.pipelineForEmbedding(this.embedding), this.bindGroups.embedding, Math.ceil(this.hidden / 128));

    for (let i = 0; i < this.layers.length; i++) {
      const layer = this.layers[i];
      const bg = this.bindGroups.layers[i];
      this.dispatch(encoder, this.pipelines.rmsNorm, bg.inputNorm, 1);

      if (layer.kind === "mamba" && bg.kind === "mamba") {
        this.q4MatvecBound(encoder, layer.inProj, bg.inProj);
        this.dispatch(encoder, this.pipelines.mambaConv, bg.conv, Math.ceil(this.mambaConvDim / 128));
        this.dispatch(encoder, this.pipelines.mambaSsm, bg.ssm, this.mambaDt, this.mambaDHead);
        this.dispatch(encoder, this.pipelines.mambaGateNorm, bg.gateNorm, 1);
        this.q4MatvecBound(encoder, layer.outProj, bg.outProj);
      } else {
        const attnLayer = layer as AttentionLayer;
        const attnBg = bg as AttentionLayerBindGroups;
        this.q4MatvecBatch(encoder, [
          { weight: attnLayer.qProj, bindGroup: attnBg.qProj },
          { weight: attnLayer.kProj, bindGroup: attnBg.kProj },
          { weight: attnLayer.vProj, bindGroup: attnBg.vProj },
        ]);
        this.dispatch(encoder, this.pipelines.storeKv, attnBg.storeKv, this.kvHeads);
        this.dispatch(encoder, this.pipelines.attentionScore, attnBg.attentionScore, this.heads, this.pos + 1);
        this.dispatch(encoder, this.pipelines.attentionValue, attnBg.attentionValue, this.heads);
        this.q4MatvecBound(encoder, attnLayer.oProj, attnBg.oProj);
      }

      this.dispatch(encoder, this.pipelines.addScaledRmsNorm, bg.postAddNorm, 1);
      this.q4MatvecBatch(encoder, [
        { weight: layer.mlp.gate, bindGroup: bg.mlpGate },
        { weight: layer.mlp.up, bindGroup: bg.mlpUp },
      ]);
      this.dispatch(encoder, this.pipelines.siluMul, bg.silu, Math.ceil(this.intermediate / 128));
      this.q4MatvecBound(encoder, layer.mlp.down, bg.mlpDown);
      this.dispatch(encoder, this.pipelines.addScaled, bg.mlpAdd, Math.ceil(this.hidden / 128));
    }

    let best = 0;
    if (needLogits) {
      this.dispatch(encoder, this.pipelines.rmsNorm, this.bindGroups.finalNorm, 1);
      this.q4MatvecBound(encoder, this.lmHead, this.bindGroups.lmHead);
      this.dispatch(encoder, this.pipelines.scaleLogits, this.bindGroups.scaleLogits, Math.ceil(100352 / 128));
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
    const destroyWeight = (w: LinearWeight) => {
      w.q.destroy();
      w.params.destroy();
    };
    destroyWeight(this.embedding);
    this.lmHead.params.destroy();
    this.finalNorm.destroy();
    for (const layer of this.layers) {
      layer.inputNorm.destroy();
      layer.postNorm.destroy();
      destroyWeight(layer.mlp.gate);
      destroyWeight(layer.mlp.up);
      destroyWeight(layer.mlp.down);
      if (layer.kind === "mamba") {
        destroyWeight(layer.inProj);
        destroyWeight(layer.outProj);
        layer.convWeight.destroy();
        layer.convBias.destroy();
        layer.dtBias.destroy();
        layer.aNeg.destroy();
        layer.d.destroy();
        layer.mambaNorm.destroy();
        layer.convState.destroy();
        layer.ssmState.destroy();
      } else {
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
      this.normMambaParams,
      this.siluParams,
      this.addHiddenParams,
      this.argmaxSizeParams,
      this.argmaxResult,
      this.argmaxReadback,
      this.topkResult,
      this.topkReadback,
      this.hiddenA,
      this.hiddenB,
      this.norm,
      this.mambaProj,
      this.mambaConvAct,
      this.mambaSsmY,
      this.mambaNormed,
      this.gate,
      this.up,
      this.ff,
      this.q,
      this.k,
      this.v,
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
    const addNorm = (residual: GPUBuffer, add: GPUBuffer, weight: GPUBuffer, output: GPUBuffer) =>
      bind(this.pipelines.addScaledRmsNorm, [
        { binding: 0, resource: { buffer: residual } },
        { binding: 1, resource: { buffer: add } },
        { binding: 2, resource: { buffer: weight } },
        { binding: 3, resource: { buffer: output } },
        { binding: 4, resource: { buffer: this.addHiddenParams } },
      ]);
    const addScaled = (residual: GPUBuffer, add: GPUBuffer) =>
      bind(this.pipelines.addScaled, [
        { binding: 0, resource: { buffer: residual } },
        { binding: 1, resource: { buffer: add } },
        { binding: 2, resource: { buffer: this.addHiddenParams } },
      ]);
    const linear = (input: GPUBuffer, weight: LinearWeight, output: GPUBuffer) =>
      bind(this.pipelineForWeight(weight), [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: weight.q } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: weight.params } },
      ]);
    const silu = bind(this.pipelines.siluMul, [
      { binding: 0, resource: { buffer: this.gate } },
      { binding: 1, resource: { buffer: this.up } },
      { binding: 2, resource: { buffer: this.ff } },
      { binding: 3, resource: { buffer: this.siluParams } },
    ]);

    const layers: LayerBindGroups[] = this.layers.map((layer) => {
      const common = {
        inputNorm: rms(this.hiddenA, layer.inputNorm, this.norm, this.normHiddenParams),
        postAddNorm: addNorm(this.hiddenA, this.hiddenB, layer.postNorm, this.norm),
        mlpGate: linear(this.norm, layer.mlp.gate, this.gate),
        mlpUp: linear(this.norm, layer.mlp.up, this.up),
        silu,
        mlpDown: linear(this.ff, layer.mlp.down, this.hiddenB),
        mlpAdd: addScaled(this.hiddenA, this.hiddenB),
      };
      if (layer.kind === "mamba") {
        return {
          kind: "mamba",
          ...common,
          inProj: linear(this.norm, layer.inProj, this.mambaProj),
          conv: bind(this.pipelines.mambaConv, [
            { binding: 0, resource: { buffer: this.mambaProj } },
            { binding: 1, resource: { buffer: layer.convState } },
            { binding: 2, resource: { buffer: layer.convWeight } },
            { binding: 3, resource: { buffer: layer.convBias } },
            { binding: 4, resource: { buffer: this.mambaConvAct } },
          ]),
          ssm: bind(this.pipelines.mambaSsm, [
            { binding: 0, resource: { buffer: this.mambaProj } },
            { binding: 1, resource: { buffer: this.mambaConvAct } },
            { binding: 2, resource: { buffer: layer.dtBias } },
            { binding: 3, resource: { buffer: layer.aNeg } },
            { binding: 4, resource: { buffer: layer.d } },
            { binding: 5, resource: { buffer: layer.ssmState } },
            { binding: 6, resource: { buffer: this.mambaSsmY } },
          ]),
          gateNorm: bind(this.pipelines.mambaGateNorm, [
            { binding: 0, resource: { buffer: this.mambaProj } },
            { binding: 1, resource: { buffer: this.mambaSsmY } },
            { binding: 2, resource: { buffer: layer.mambaNorm } },
            { binding: 3, resource: { buffer: this.mambaNormed } },
            { binding: 4, resource: { buffer: this.normMambaParams } },
          ]),
          outProj: linear(this.mambaNormed, layer.outProj, this.hiddenB),
        };
      }
      return {
        kind: "attention",
        ...common,
        qProj: linear(this.norm, layer.qProj, this.q),
        kProj: linear(this.norm, layer.kProj, this.k),
        vProj: linear(this.norm, layer.vProj, this.v),
        storeKv: bind(this.pipelines.storeKv, [
          { binding: 0, resource: { buffer: this.k } },
          { binding: 1, resource: { buffer: this.v } },
          { binding: 2, resource: { buffer: layer.keyCache } },
          { binding: 3, resource: { buffer: layer.valueCache } },
          { binding: 4, resource: { buffer: this.posParams } },
        ]),
        attentionScore: bind(this.pipelines.attentionScore, [
          { binding: 0, resource: { buffer: this.q } },
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
      };
    });

    return {
      embedding: bind(this.pipelineForEmbedding(this.embedding), [
        { binding: 0, resource: { buffer: this.embedding.q } },
        { binding: 1, resource: { buffer: this.hiddenA } },
        { binding: 2, resource: { buffer: this.tokenParams } },
      ]),
      finalNorm: rms(this.hiddenA, this.finalNorm, this.norm, this.normHiddenParams),
      lmHead: linear(this.norm, this.lmHead, this.logits),
      scaleLogits: bind(this.pipelines.scaleLogits, [
        { binding: 0, resource: { buffer: this.logits } },
        { binding: 1, resource: { buffer: this.argmaxSizeParams } },
      ]),
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

  private pipelineForEmbedding(weight: LinearWeight): GPUComputePipeline {
    switch (weight.kind) {
      case "gguf-q8_0":
        return this.pipelines.embeddingQ8;
      case "gguf-q4_k":
        return this.pipelines.embeddingQ4K;
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
      case "gguf-q6_k":
        return this.pipelines.q6KMatvec;
    }
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
      pass.setPipeline(this.pipelineForWeight(weight));
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
    embeddingQ8: pipeline("granite.embedding_q8", EMBEDDING_Q8_WGSL),
    embeddingQ4K: pipeline("granite.embedding_q4_k", EMBEDDING_Q4_K_WGSL),
    embeddingQ6K: pipeline("granite.embedding_q6_k", EMBEDDING_Q6_K_WGSL),
    rmsNorm: pipeline("granite.rms_norm", RMS_NORM_WGSL),
    addScaledRmsNorm: pipeline("granite.add_scaled_rms_norm", ADD_SCALED_RMS_NORM_WGSL),
    addScaled: pipeline("granite.add_scaled", ADD_SCALED_WGSL),
    q8Matvec: pipeline("granite.q8_matvec", Q8_MATVEC_WGSL),
    q4KMatvec: pipeline("granite.q4_k_matvec", Q4_K_MATVEC_WGSL),
    q6KMatvec: pipeline("granite.q6_k_matvec", Q6_K_MATVEC_WGSL),
    mambaConv: pipeline("granite.mamba_conv", MAMBA_CONV_WGSL),
    mambaSsm: pipeline("granite.mamba_ssm", MAMBA_SSM_WGSL),
    mambaGateNorm: pipeline("granite.mamba_gate_norm", MAMBA_GATE_NORM_WGSL),
    storeKv: pipeline("granite.store_kv", STORE_KV_WGSL),
    attentionScore: pipeline("granite.attention_score", ATTENTION_SCORE_WGSL),
    attentionValue: pipeline("granite.attention_value", ATTENTION_VALUE_WGSL),
    siluMul: pipeline("granite.silu_mul", SILU_MUL_WGSL),
    scaleLogits: pipeline("granite.scale_logits", SCALE_LOGITS_WGSL),
    argmax: pipeline("granite.argmax", ARGMAX_WGSL),
    topk256: pipeline("granite.topk256", TOPK256_WGSL),
  };
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

function base64Bytes(s: string): Uint8Array {
  const binary = atob(s);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}

const GGUF_DEQUANT_WGSL = /* wgsl */ `
fn byte_at(index: u32) -> u32 {
  let word = q[index >> 2u];
  let shift = (index & 3u) * 8u;
  return (word >> shift) & 0xffu;
}

fn u16_at(index: u32) -> u32 {
  return byte_at(index) | (byte_at(index + 1u) << 8u);
}

fn f16_at(index: u32) -> f32 {
  let h = u16_at(index);
  let sign = select(1.0, -1.0, (h & 0x8000u) != 0u);
  let exp = (h >> 10u) & 0x1fu;
  let frac = h & 0x03ffu;
  if (exp == 0u) {
    return sign * exp2(-14.0) * (f32(frac) / 1024.0);
  }
  if (exp == 31u) {
    return sign * 65504.0;
  }
  return sign * exp2(f32(exp) - 15.0) * (1.0 + f32(frac) / 1024.0);
}

fn i8_at(index: u32) -> i32 {
  let b = byte_at(index);
  return select(i32(b), i32(b) - 256, b >= 128u);
}

fn q8_0_weight(row: u32, k: u32, blocks: u32) -> f32 {
  let block = k >> 5u;
  let within = k & 31u;
  let base = row * blocks * 34u + block * 34u;
  return f16_at(base) * f32(i8_at(base + 2u + within));
}

fn q4_k_scale(block_base: u32, group: u32) -> u32 {
  let i = group & 3u;
  if (group < 4u) {
    return byte_at(block_base + 4u + i) & 0x3fu;
  }
  return (byte_at(block_base + 12u + i) & 0x0fu) |
    ((byte_at(block_base + 4u + i) >> 2u) & 0x30u);
}

fn q4_k_min(block_base: u32, group: u32) -> u32 {
  let i = group & 3u;
  if (group < 4u) {
    return byte_at(block_base + 8u + i) & 0x3fu;
  }
  return (byte_at(block_base + 12u + i) >> 4u) |
    ((byte_at(block_base + 8u + i) >> 2u) & 0x30u);
}

fn q4_k_weight(row: u32, k: u32, blocks: u32) -> f32 {
  let block = k >> 8u;
  let rem = k & 255u;
  let group = rem >> 5u;
  let within = rem & 31u;
  let base = row * blocks * 144u + block * 144u;
  let q_byte = byte_at(base + 16u + (group >> 1u) * 32u + within);
  let qv = select(q_byte >> 4u, q_byte & 15u, (group & 1u) == 0u);
  let d = f16_at(base);
  let dmin = f16_at(base + 2u);
  return d * f32(q4_k_scale(base, group)) * f32(qv) -
    dmin * f32(q4_k_min(base, group));
}

fn q6_k_weight(row: u32, k: u32, blocks: u32) -> f32 {
  let block = k >> 8u;
  let rem = k & 255u;
  let scale_group = rem >> 4u;
  let within16 = rem & 15u;
  let group32 = scale_group >> 1u;
  let within32 = (scale_group & 1u) * 16u + within16;
  let base = row * blocks * 210u + block * 210u;

  let ql_outer = group32 >> 2u;
  let ql_rem = group32 & 3u;
  let ql_shift = (ql_rem >> 1u) * 4u;
  let ql_index = base + ql_outer * 64u + (ql_rem & 1u) * 32u + within32;
  let ql = (byte_at(ql_index) >> ql_shift) & 15u;

  let qh_shift = (group32 & 3u) * 2u;
  let qh_index = base + 128u + ql_outer * 32u + within32;
  let qh = (byte_at(qh_index) >> qh_shift) & 3u;

  let qv = i32(ql | (qh << 4u)) - 32;
  let scale = i8_at(base + 192u + scale_group);
  return f16_at(base + 208u) * f32(scale) * f32(qv);
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

${GGUF_DEQUANT_WGSL}

var<workgroup> partial: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = wg.x + wg.y * params.x_groups;
  if (n >= params.n) {
    return;
  }

  var sum = 0.0;
  for (var k = lid.x; k < params.k; k = k + 128u) {
    sum = sum + input[k] * q8_0_weight(n, k, params.blocks);
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

const Q4_K_MATVEC_WGSL = Q8_MATVEC_WGSL.replace(
  "input[k] * q8_0_weight(n, k, params.blocks)",
  "input[k] * q4_k_weight(n, k, params.blocks)",
);
const Q6_K_MATVEC_WGSL = Q8_MATVEC_WGSL.replace(
  "input[k] * q8_0_weight(n, k, params.blocks)",
  "input[k] * q6_k_weight(n, k, params.blocks)",
);

const EMBEDDING_Q8_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> q: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: vec2<u32>;

${GGUF_DEQUANT_WGSL}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= 768u) {
    return;
  }
  output[k] = q8_0_weight(params.x, k, params.y) * 12.0;
}
`;

const EMBEDDING_Q4_K_WGSL = EMBEDDING_Q8_WGSL.replace(
  "output[k] = q8_0_weight(params.x, k, params.y)",
  "output[k] = q4_k_weight(params.x, k, params.y)",
);
const EMBEDDING_Q6_K_WGSL = EMBEDDING_Q8_WGSL.replace(
  "output[k] = q8_0_weight(params.x, k, params.y)",
  "output[k] = q6_k_weight(params.x, k, params.y)",
);

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
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  var sum = 0.0;
  for (var i = lid.x; i < params.group_size; i = i + 256u) {
    let v = input[i];
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
    output[i] = input[i] * inv * weight[i];
  }
}
`;

const ADD_SCALED_RMS_NORM_WGSL = /* wgsl */ `
struct Params {
  group_size: u32,
};

@group(0) @binding(0) var<storage, read_write> residual: array<f32>;
@group(0) @binding(1) var<storage, read> add: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  var sum = 0.0;
  for (var i = lid.x; i < params.group_size; i = i + 256u) {
    let v = residual[i] + add[i] * 0.246;
    residual[i] = v;
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
    output[i] = residual[i] * inv * weight[i];
  }
}
`;

const ADD_SCALED_WGSL = /* wgsl */ `
struct Params {
  n: u32,
};

@group(0) @binding(0) var<storage, read_write> residual: array<f32>;
@group(0) @binding(1) var<storage, read> add: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < params.n) {
    residual[i] = residual[i] + add[i] * 0.246;
  }
}
`;

const MAMBA_CONV_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> proj: array<f32>;
@group(0) @binding(1) var<storage, read_write> state: array<f32>;
@group(0) @binding(2) var<storage, read> conv_weight: array<f32>;
@group(0) @binding(3) var<storage, read> conv_bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

fn silu(x: f32) -> f32 {
  return x / (1.0 + exp(-x));
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (c >= 1792u) {
    return;
  }
  let new_value = proj[1536u + c];
  let s = c * 4u;
  let p1 = state[s + 1u];
  let p2 = state[s + 2u];
  let p3 = state[s + 3u];
  state[s] = p1;
  state[s + 1u] = p2;
  state[s + 2u] = p3;
  state[s + 3u] = new_value;

  let w = c * 4u;
  let y = p1 * conv_weight[w] +
    p2 * conv_weight[w + 1u] +
    p3 * conv_weight[w + 2u] +
    new_value * conv_weight[w + 3u] +
    conv_bias[c];
  output[c] = silu(y);
}
`;

const MAMBA_SSM_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> proj: array<f32>;
@group(0) @binding(1) var<storage, read> conv: array<f32>;
@group(0) @binding(2) var<storage, read> dt_bias: array<f32>;
@group(0) @binding(3) var<storage, read> a_neg: array<f32>;
@group(0) @binding(4) var<storage, read> d_weight: array<f32>;
@group(0) @binding(5) var<storage, read_write> state: array<f32>;
@group(0) @binding(6) var<storage, read_write> output: array<f32>;

var<workgroup> partial: array<f32, 128>;

fn softplus(x: f32) -> f32 {
  return select(log(1.0 + exp(x)), x + log(1.0 + exp(-x)), x > 20.0);
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = wg.x;
  let h = wg.y;
  let s = lid.x;
  let nh = n * 32u + h;
  let hs = conv[nh];
  let dt_raw = proj[3328u + n];
  let dt = softplus(dt_raw + dt_bias[n]);
  let decay = exp(dt * a_neg[n]);
  let b = conv[1536u + s];
  let c = conv[1664u + s];
  let idx = (nh * 128u) + s;
  let next_state = state[idx] * decay + dt * b * hs;
  state[idx] = next_state;
  partial[s] = next_state * c;
  workgroupBarrier();

  var stride = 64u;
  loop {
    if (s < stride) {
      partial[s] = partial[s] + partial[s + stride];
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride = stride >> 1u;
  }

  if (s == 0u) {
    output[nh] = partial[0] + d_weight[n] * hs;
  }
}
`;

const MAMBA_GATE_NORM_WGSL = /* wgsl */ `
struct Params {
  group_size: u32,
};

@group(0) @binding(0) var<storage, read> proj: array<f32>;
@group(0) @binding(1) var<storage, read> ssm_y: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> partial: array<f32, 256>;

fn silu(x: f32) -> f32 {
  return x / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  var sum = 0.0;
  for (var i = lid.x; i < params.group_size; i = i + 256u) {
    let v = ssm_y[i] * silu(proj[i]);
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
    output[i] = ssm_y[i] * silu(proj[i]) * inv * weight[i];
  }
}
`;

const STORE_KV_WGSL = /* wgsl */ `
struct Params {
  pos: u32,
  max_context: u32,
};

@group(0) @binding(0) var<storage, read> k: array<f32>;
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read_write> key_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> value_cache: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let kv_head = wg.x;
  let d = lid.x;
  let src = kv_head * 64u + d;
  let dst = (kv_head * params.max_context + params.pos) * 64u + d;
  key_cache[dst] = k[src];
  value_cache[dst] = v[src];
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
  let kv_head = head / 3u;
  let d = lid.x;
  let q_base = head * 64u;
  let k_base = (kv_head * params.max_context + t) * 64u;
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
    scores[head * params.max_context + t] = partial[0] * 0.015625;
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
  let kv_head = head / 3u;

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

const SCALE_LOGITS_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<uniform> size: u32;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < size) {
    logits[i] = logits[i] * 0.3333333333333333;
  }
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
