// Hand-written WebGPU runtime for Llama-style decoder models backed by GGUF
// Q8_0 and K-quant artifacts.

import { fetchCachedRange, mapWithConcurrency, requestPersistence, type LoadStepCallback, type ProgressCallback } from "./cache";
import {
  fetchGgufHeader,
  GGML_TYPE,
  ggmlTypeName,
  type GgufFile,
  type GgufTensorInfo,
  type GgufValue,
} from "./gguf";
import type { ModelDef } from "./registry";
import { needsSamplingReadback, sampleFromCandidateBuffer, type SamplingOptions } from "./sampling";
import { Tokenizer } from "./tokenizer";

type TensorDtype =
  | "float32"
  | "float16"
  | "uint8"
  | "int64"
  | "gguf-q5_0"
  | "gguf-q5_1"
  | "gguf-q8_0"
  | "gguf-q4_k"
  | "gguf-q5_k"
  | "gguf-q6_k";

interface TensorSpec {
  dtype: TensorDtype;
  shape: number[];
  external?: { path: string; offset: number; length: number };
  inlineBase64?: string;
}

interface LlamaManifest {
  format: "chonklm.pleias-wgsl.v1" | "chonklm.llama-wgsl.v1";
  model: {
    id: string;
    hidden: number;
    intermediate: number;
    vocab: number;
    layers: number;
    heads: number;
    kvHeads: number;
    headDim: number;
    maxContext: number;
    ropeTheta: number;
    normEps: number;
    qkNorm?: boolean;
    qkNormEps?: number;
    architecture?: "llama" | "gemma3" | "openelm";
    ropeLayout?: "interleaved" | "split";
    embedding?: "float" | "q4" | "gguf";
    embeddingScale?: number;
    lmHead?: "embedding" | "q4" | "q4-projected" | "q4-chunked";
    embeddingDim?: number;
    layerHeads?: number[];
    layerKvHeads?: number[];
    layerIntermediates?: number[];
    layerRopeTheta?: number[];
    layerAttentionWindow?: number[];
    attention?: "split" | "qkv";
    mlp?: "split" | "swiglu-packed";
    activation?: "silu" | "gelu-tanh";
    blockSize: number;
    embeddingChunks: string[];
    lmHeadChunks?: string[];
  };
  aliases?: Record<string, string>;
  tensors: Record<string, TensorSpec>;
}

interface Fp32Chunk {
  buffer: GPUBuffer;
  rows: number;
  offset: number;
}

interface Q4Chunk {
  q: GPUBuffer;
  rows: number;
  offset: number;
}

interface Q4Embedding {
  chunks: Q4Chunk[];
  scales: GPUBuffer;
  zp: GPUBuffer;
  blocks: number;
  zpBlocks: number;
}

interface Q4Weight {
  kind: "gguf-q5_0" | "gguf-q5_1" | "gguf-q8_0" | "gguf-q4_k" | "gguf-q5_k" | "gguf-q6_k";
  q: GPUBuffer;
  k: number;
  n: number;
  blocks: number;
  blockBytes: number;
}

interface LayerWeights {
  heads: number;
  kvHeads: number;
  kvDim: number;
  qDim: number;
  intermediate: number;
  ropeTheta: number;
  attentionWindow: number;
  inputNorm: GPUBuffer;
  postNorm: GPUBuffer;
  preFfnNorm?: GPUBuffer;
  postFfnNorm?: GPUBuffer;
  qNorm?: GPUBuffer;
  kNorm?: GPUBuffer;
  qProj?: Q4Weight;
  kProj?: Q4Weight;
  vProj?: Q4Weight;
  qkvProj?: Q4Weight;
  oProj: Q4Weight;
  gate?: Q4Weight;
  up?: Q4Weight;
  gateUp?: Q4Weight;
  down: Q4Weight;
  keyCache: GPUBuffer;
  valueCache: GPUBuffer;
}

interface Pipelines {
  embedding: GPUComputePipeline;
  rmsNorm: GPUComputePipeline;
  addRmsNorm: GPUComputePipeline;
  q4Matvec: GPUComputePipeline;
  q4MatvecZp: GPUComputePipeline;
  q5Matvec: GPUComputePipeline;
  q5_1Matvec: GPUComputePipeline;
  q8Matvec: GPUComputePipeline;
  q4KMatvec: GPUComputePipeline;
  q5KMatvec: GPUComputePipeline;
  q6KMatvec: GPUComputePipeline;
  ropeStore: GPUComputePipeline;
  qkNormRopeStore: GPUComputePipeline;
  qkvNormRopeStore: GPUComputePipeline;
  ropeStoreInterleaved: GPUComputePipeline;
  qkNormRopeStoreInterleaved: GPUComputePipeline;
  qkvNormRopeStoreInterleaved: GPUComputePipeline;
  attentionScore: GPUComputePipeline;
  attentionValue: GPUComputePipeline;
  add: GPUComputePipeline;
  addClamp: GPUComputePipeline;
  siluMul: GPUComputePipeline;
  siluSplitMul: GPUComputePipeline;
  geluTanhMul: GPUComputePipeline;
  q4Embedding: GPUComputePipeline;
  q5Embedding: GPUComputePipeline;
  q5_1Embedding: GPUComputePipeline;
  q8Embedding: GPUComputePipeline;
  q4KEmbedding: GPUComputePipeline;
  q5KEmbedding: GPUComputePipeline;
  q6KEmbedding: GPUComputePipeline;
  q4MatvecZpOffset: GPUComputePipeline;
  lmHead: GPUComputePipeline;
  argmaxStage1: GPUComputePipeline;
  argmaxStage2: GPUComputePipeline;
}

interface ActivePass {
  encoder: GPUCommandEncoder;
  pass: GPUComputePassEncoder;
  paramCursor: number;
}

export interface LoadedLlamaWgslModel {
  runtime: "llama-webgpu";
  def: ModelDef;
  tokenizer: Tokenizer;
  engine: LlamaWgslEngine;
  ep: "webgpu";
  cachedTokenIds: number[];
  cachedNextId: number | null;
}

export async function loadLlamaWgslModel(
  model: ModelDef,
  onProgress?: ProgressCallback,
  onStep?: LoadStepCallback,
): Promise<LoadedLlamaWgslModel> {
  if (model.runtime !== "llama-webgpu") {
    throw new Error(`loadLlamaWgslModel: ${model.id} is not configured for the Llama WGSL runtime`);
  }
  if (!("gpu" in navigator)) {
    throw new Error("Llama WGSL runtime requires WebGPU");
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
  if (!adapter) throw new Error("Llama WGSL runtime could not acquire a WebGPU adapter");
  onStep?.({ step: "runtime", detail: "requesting WebGPU device" });
  const requiredLimits: Record<string, number> = {};
  if (adapter.limits.maxStorageBufferBindingSize > 128 * 1024 * 1024) {
    requiredLimits.maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize;
  }
  const device = await adapter.requestDevice(
    Object.keys(requiredLimits).length ? { requiredLimits } : undefined,
  );
  void device.lost.then((info) => {
    console.warn(`[chonklm] Llama WGSL WebGPU device lost: ${info.reason} ${info.message}`);
  });

  onStep?.({
    step: "tokenizer",
    detail: "loading tokenizer and GGUF header",
  });
  const tokenizerPromise = Tokenizer.load(model);
  const manifest = await buildGgufLlamaManifest(model, onProgress);
  const tokenizer = await tokenizerPromise;

  const contextLength = Math.min(model.maxContext, manifest.model.maxContext);
  const loader = new TensorLoader(model, manifest, onProgress);
  const engine = await LlamaWgslEngine.create(device, manifest, loader, contextLength, onStep);
  onStep?.({ step: "ready", detail: "Llama WebGPU runtime ready" });
  return {
    runtime: "llama-webgpu",
    def: model,
    tokenizer,
    engine,
    ep: "webgpu",
    cachedTokenIds: [],
    cachedNextId: null,
  };
}

export async function generateLlamaWgsl(
  loaded: LoadedLlamaWgslModel,
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
  const sampling: SamplingOptions = {
    temperature: opts.temperature ?? 0,
    topP: opts.topP ?? 1,
    topK: opts.topK ?? 0,
    repetitionPenalty: opts.repetitionPenalty ?? loaded.def.defaultRepetitionPenalty ?? 1,
  };
  const generated: number[] = [];
  let cumulative = "";

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
      const needLogits = i === suffix.length - 1;
      nextId = await loaded.engine.runToken(suffix[i], needLogits, {
        ...sampling,
        penalizedTokenIds: needLogits ? promptIds.slice(0, prefixLen + i + 1) : undefined,
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
    if (step + 1 >= max) break;
    nextId = await loaded.engine.runToken(nextId, true, {
      ...sampling,
      penalizedTokenIds: promptIds.concat(generated),
      seenIds: promptIds.concat(generated),
    });
  }

  loaded.cachedTokenIds = promptIds.concat(generated);
  loaded.cachedNextId = eos.has(generated[generated.length - 1] ?? -1) ? null : nextId;

  const elapsed = (performance.now() - t0) / 1000;
  return { text: cumulative, ids: generated, tokensPerSec: generated.length / Math.max(elapsed, 1e-3) };
}

export function disposeLlamaWgsl(loaded: LoadedLlamaWgslModel): void {
  loaded.engine.dispose();
}

export function resetLlamaWgslConversation(loaded: LoadedLlamaWgslModel): void {
  loaded.engine.reset();
  loaded.cachedTokenIds = [];
  loaded.cachedNextId = null;
}

async function buildGgufLlamaManifest(model: ModelDef, onProgress?: ProgressCallback): Promise<LlamaManifest> {
  if (!model.gguf) throw new Error(`Llama WGSL ${model.id} is missing gguf path`);
  const ggufPath = model.gguf;
  const gguf = await fetchGgufHeader(modelFileUrl(model, ggufPath), onProgress);
  const architecture = ggufString(gguf, "general.architecture");
  if (architecture !== "qwen3" && architecture !== "llama" && architecture !== "gemma3" && architecture !== "openelm") {
    throw new Error(`Llama GGUF WGSL does not support architecture ${architecture}`);
  }
  const isGemma3 = architecture === "gemma3";
  const isOpenElm = architecture === "openelm";

  const tensors: Record<string, TensorSpec> = {};
  for (const tensor of gguf.tensors) {
    tensors[tensor.name] = tensorSpecFromGguf(gguf, ggufPath, tensor);
  }

  const aliases: Record<string, string> = {
    "model.embed_tokens.weight": "token_embd.weight",
    [`model.layers.${model.layers}.final_norm_layernorm.weight`]: "output_norm.weight",
    "lm_head.MatMul.weight": gguf.tensorMap.has("output.weight") ? "output.weight" : "token_embd.weight",
  };
  for (let i = 0; i < model.layers; i++) {
    aliases[`model.layers.${i}.input_layernorm.weight`] = `blk.${i}.attn_norm.weight`;
    aliases[`model.layers.${i}.post_attention_layernorm.weight`] = isGemma3
      ? `blk.${i}.post_attention_norm.weight`
      : `blk.${i}.ffn_norm.weight`;
    if (isGemma3) {
      aliases[`model.layers.${i}.pre_feedforward_layernorm.weight`] = `blk.${i}.ffn_norm.weight`;
      aliases[`model.layers.${i}.post_feedforward_layernorm.weight`] = `blk.${i}.post_ffw_norm.weight`;
    }
    aliases[`model.layers.${i}.attn.q_norm.layernorm.weight`] = `blk.${i}.attn_q_norm.weight`;
    aliases[`model.layers.${i}.attn.k_norm.layernorm.weight`] = `blk.${i}.attn_k_norm.weight`;
    if (isOpenElm) {
      aliases[`model.layers.${i}.attn.qkv_proj.MatMul.weight`] = `blk.${i}.attn_qkv.weight`;
    } else {
      aliases[`model.layers.${i}.attn.q_proj.MatMul.weight`] = `blk.${i}.attn_q.weight`;
      aliases[`model.layers.${i}.attn.k_proj.MatMul.weight`] = `blk.${i}.attn_k.weight`;
      aliases[`model.layers.${i}.attn.v_proj.MatMul.weight`] = `blk.${i}.attn_v.weight`;
    }
    aliases[`model.layers.${i}.attn.o_proj.MatMul.weight`] = `blk.${i}.attn_output.weight`;
    aliases[`model.layers.${i}.mlp.gate_proj.MatMul.weight`] = `blk.${i}.ffn_gate.weight`;
    aliases[`model.layers.${i}.mlp.up_proj.MatMul.weight`] = `blk.${i}.ffn_up.weight`;
    aliases[`model.layers.${i}.mlp.down_proj.MatMul.weight`] = `blk.${i}.ffn_down.weight`;
  }

  const layers = ggufNumber(gguf, `${architecture}.block_count`);
  const hidden = ggufNumber(gguf, `${architecture}.embedding_length`);
  const layerHeads = isOpenElm ? ggufNumberArray(gguf, `${architecture}.attention.head_count`) : undefined;
  const layerKvHeads = isOpenElm ? ggufNumberArray(gguf, `${architecture}.attention.head_count_kv`) : undefined;
  const layerIntermediates = isOpenElm ? ggufNumberArray(gguf, `${architecture}.feed_forward_length`) : undefined;
  const heads = layerHeads ? Math.max(...layerHeads) : ggufNumber(gguf, `${architecture}.attention.head_count`);
  const kvHeads = layerKvHeads ? Math.max(...layerKvHeads) : ggufNumber(gguf, `${architecture}.attention.head_count_kv`);
  const intermediate = layerIntermediates
    ? Math.max(...layerIntermediates)
    : ggufNumber(gguf, `${architecture}.feed_forward_length`);
  const headDim = ggufNumber(gguf, `${architecture}.attention.key_length`, Math.floor(hidden / heads));
  const gemmaSlidingWindow = isGemma3 ? ggufNumber(gguf, "gemma3.attention.sliding_window", 0) : 0;
  const gemmaGlobalTheta = isGemma3 ? ggufNumber(gguf, "gemma3.rope.freq_base", 1000000) : 0;
  const gemmaLocalTheta = 10000;
  return {
    format: "chonklm.llama-wgsl.v1",
    model: {
      id: model.id,
      hidden,
      intermediate,
      vocab: model.vocab,
      layers,
      heads,
      kvHeads,
      headDim,
      maxContext: Math.min(model.maxContext, ggufNumber(gguf, `${architecture}.context_length`, model.maxContext)),
      ropeTheta: isGemma3 ? gemmaLocalTheta : ggufNumber(gguf, `${architecture}.rope.freq_base`, 10000),
      normEps: ggufNumber(gguf, `${architecture}.attention.layer_norm_rms_epsilon`, 1e-5),
      qkNorm: gguf.tensorMap.has("blk.0.attn_q_norm.weight") && gguf.tensorMap.has("blk.0.attn_k_norm.weight"),
      qkNormEps: ggufNumber(gguf, `${architecture}.attention.layer_norm_rms_epsilon`, 1e-5),
      architecture: isGemma3 ? "gemma3" : isOpenElm ? "openelm" : "llama",
      ropeLayout: architecture === "llama" ? "interleaved" : "split",
      embedding: "gguf",
      embeddingDim: hidden,
      embeddingScale: isGemma3 ? Math.sqrt(hidden) : 1,
      lmHead: "q4",
      layerHeads,
      layerKvHeads,
      layerIntermediates,
      layerRopeTheta: isGemma3
        ? Array.from({ length: layers }, (_, i) => i % 6 === 5 ? gemmaGlobalTheta : gemmaLocalTheta)
        : undefined,
      layerAttentionWindow: isGemma3
        ? Array.from({ length: layers }, (_, i) => i % 6 === 5 ? 0 : gemmaSlidingWindow)
        : undefined,
      attention: isOpenElm ? "qkv" : "split",
      mlp: "split",
      activation: isGemma3 ? "gelu-tanh" : "silu",
      blockSize: 32,
      embeddingChunks: ["model.embed_tokens.weight"],
    },
    aliases,
    tensors,
  };
}

function modelFileUrl(model: ModelDef, path: string): string {
  return path.startsWith("http") ? path : `${model.base}/${path}`;
}

function tensorSpecFromGguf(gguf: GgufFile, ggufPath: string, tensor: GgufTensorInfo): TensorSpec {
  return {
    dtype: tensorDtypeFromGgml(tensor.type),
    shape: [...tensor.shape].reverse(),
    external: {
      path: ggufPath,
      offset: gguf.dataOffset + tensor.offset,
      length: tensor.nBytes,
    },
  };
}

function tensorDtypeFromGgml(type: number): TensorDtype {
  switch (type) {
    case GGML_TYPE.F32:
      return "float32";
    case GGML_TYPE.F16:
      return "float16";
    case GGML_TYPE.Q5_0:
      return "gguf-q5_0";
    case GGML_TYPE.Q5_1:
      return "gguf-q5_1";
    case GGML_TYPE.Q8_0:
      return "gguf-q8_0";
    case GGML_TYPE.Q4_K:
      return "gguf-q4_k";
    case GGML_TYPE.Q5_K:
      return "gguf-q5_k";
    case GGML_TYPE.Q6_K:
      return "gguf-q6_k";
    default:
      throw new Error(`unsupported GGUF tensor type ${ggmlTypeName(type)}`);
  }
}

function ggufNumber(gguf: GgufFile, key: string, fallback?: number): number {
  const value = gguf.kv.get(key)?.value;
  const number = ggufMetadataNumber(value);
  if (number !== undefined) return number;
  if (fallback !== undefined) return fallback;
  throw new Error(`GGUF metadata ${key} is missing or not numeric`);
}

function ggufNumberArray(gguf: GgufFile, key: string): number[] {
  const value = gguf.kv.get(key)?.value;
  if (!Array.isArray(value)) {
    throw new Error(`GGUF metadata ${key} is missing or not a numeric array`);
  }
  return value.map((item, i) => {
    const number = ggufMetadataNumber(item);
    if (number === undefined) {
      throw new Error(`GGUF metadata ${key}[${i}] is not numeric`);
    }
    return number;
  });
}

function ggufMetadataNumber(value: unknown): number | undefined {
  if (typeof value === "number") return value;
  if (typeof value === "bigint") return Number(value);
  if (value && typeof value === "object" && "value" in value) {
    return ggufMetadataNumber((value as GgufValue).value);
  }
  return undefined;
}

function ggufString(gguf: GgufFile, key: string): string {
  const value = gguf.kv.get(key)?.value;
  if (typeof value === "string") return value;
  throw new Error(`GGUF metadata ${key} is missing or not a string`);
}

function isGgufQuantDtype(dtype: TensorDtype): boolean {
  return dtype === "gguf-q5_0" || dtype === "gguf-q5_1" || dtype === "gguf-q8_0" || dtype === "gguf-q4_k" || dtype === "gguf-q5_k" || dtype === "gguf-q6_k";
}

function ggufWeightKind(dtype: TensorDtype): Q4Weight["kind"] {
  switch (dtype) {
    case "gguf-q5_0":
      return "gguf-q5_0";
    case "gguf-q5_1":
      return "gguf-q5_1";
    case "gguf-q8_0":
      return "gguf-q8_0";
    case "gguf-q4_k":
      return "gguf-q4_k";
    case "gguf-q5_k":
      return "gguf-q5_k";
    case "gguf-q6_k":
      return "gguf-q6_k";
    default:
      throw new Error(`not a GGUF quant dtype: ${dtype}`);
  }
}

function ggufBlockSize(dtype: TensorDtype): number {
  switch (dtype) {
    case "gguf-q5_0":
    case "gguf-q5_1":
    case "gguf-q8_0":
      return 32;
    case "gguf-q4_k":
    case "gguf-q5_k":
    case "gguf-q6_k":
      return 256;
    default:
      throw new Error(`not a GGUF quant dtype: ${dtype}`);
  }
}

function ggufBlockBytes(dtype: TensorDtype): number {
  switch (dtype) {
    case "gguf-q5_0":
      return 22;
    case "gguf-q5_1":
      return 24;
    case "gguf-q8_0":
      return 34;
    case "gguf-q4_k":
      return 144;
    case "gguf-q5_k":
      return 176;
    case "gguf-q6_k":
      return 210;
    default:
      throw new Error(`not a GGUF quant dtype: ${dtype}`);
  }
}

class TensorLoader {
  constructor(
    private model: ModelDef,
    private manifest: LlamaManifest,
    private onProgress?: ProgressCallback,
  ) {}

  async bytes(name: string): Promise<Uint8Array> {
    const resolved = this.resolve(name);
    const spec = this.manifest.tensors[resolved];
    if (!spec) throw new Error(`Llama WGSL manifest missing tensor ${name}`);
    if (spec.inlineBase64) return base64Bytes(spec.inlineBase64);
    if (!spec.external) throw new Error(`Llama WGSL tensor ${name} has no payload`);

    const url = modelFileUrl(this.model, spec.external.path);
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

  has(name: string): boolean {
    return this.resolve(name) in this.manifest.tensors;
  }

  spec(name: string): TensorSpec {
    const resolved = this.resolve(name);
    const spec = this.manifest.tensors[resolved];
    if (!spec) throw new Error(`Llama WGSL manifest missing tensor ${name}`);
    return spec;
  }

  private async floatBytes(name: string): Promise<Uint8Array> {
    const spec = this.spec(name);
    const bytes = await this.bytes(name);
    if (spec.dtype === "float32") return bytes;
    if (spec.dtype !== "float16") {
      throw new Error(`Llama WGSL tensor ${name} is ${spec.dtype}, expected float32/float16`);
    }
    const src = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
    const dst = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) dst[i] = f16ToF32(src[i]);
    return new Uint8Array(dst.buffer);
  }

  private resolve(name: string): string {
    return this.manifest.aliases?.[name] ?? name;
  }

}

class LlamaWgslEngine {
  private pos = 0;
  private active: ActivePass | null = null;
  private nextBufferId = 1;
  private nextPipelineId = 1;
  private readonly bufferIds = new WeakMap<GPUBuffer, number>();
  private readonly pipelineIds = new WeakMap<GPUComputePipeline, number>();
  private readonly bindGroups = new Map<string, GPUBindGroup>();

  private readonly m: LlamaManifest["model"];
  private readonly contextLength: number;
  private readonly pipelines: Pipelines;
  private readonly ropeStorePipeline: GPUComputePipeline;
  private readonly qkNormRopeStorePipeline: GPUComputePipeline;
  private readonly qkvNormRopeStorePipeline: GPUComputePipeline;
  private readonly paramBuffers: GPUBuffer[];
  private readonly readback: GPUBuffer;

  private readonly embeddingChunks: Fp32Chunk[];
  private readonly q4EmbeddingTable: Q4Embedding | null;
  private readonly ggufEmbeddingTable: Q4Weight | null;
  private readonly embeddingProjQ4: Q4Weight | null;
  private readonly finalNorm: GPUBuffer;
  private readonly lmHeadQ4: Q4Weight | null;
  private readonly lmHeadQ4Chunks: Q4Embedding | null;
  private readonly lmHeadProjQ4: Q4Weight | null;
  private readonly layers: LayerWeights[];

  private readonly embeddingScratch: GPUBuffer;
  private readonly hiddenA: GPUBuffer;
  private readonly hiddenB: GPUBuffer;
  private readonly hiddenC: GPUBuffer;
  private readonly norm: GPUBuffer;
  private readonly q: GPUBuffer;
  private readonly qkv: GPUBuffer;
  private readonly qNormed: GPUBuffer;
  private readonly k: GPUBuffer;
  private readonly v: GPUBuffer;
  private readonly attnOut: GPUBuffer;
  private readonly gate: GPUBuffer;
  private readonly up: GPUBuffer;
  private readonly ff: GPUBuffer;
  private readonly lmHeadScratch: GPUBuffer;
  private readonly logits: GPUBuffer;
  private readonly penaltyIds: GPUBuffer;
  private readonly scores: GPUBuffer;
  private readonly argmaxScratch: GPUBuffer;
  private readonly argmaxResult: GPUBuffer;
  private readonly candidateReadback: GPUBuffer;

  static async create(
    device: GPUDevice,
    manifest: LlamaManifest,
    loader: TensorLoader,
    contextLength: number,
    onStep?: LoadStepCallback,
  ): Promise<LlamaWgslEngine> {
    const m = manifest.model;
    const embeddingDim = m.embeddingDim ?? m.hidden;
    const embeddingKind = m.embedding ?? "float";
    onStep?.({ step: "shaders", detail: "compiling WebGPU shader pipelines" });
    const pipelines = createPipelines(device, m);
    onStep?.({ step: "weights", detail: "uploading embedding weights to GPU" });
    const usageRead = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const usageState = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const usageScratch = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

    const q4 = async (stem: string, k: number, n: number): Promise<Q4Weight> => {
      if (loader.has(stem) && isGgufQuantDtype(loader.spec(stem).dtype)) {
        const spec = loader.spec(stem);
        return {
          kind: ggufWeightKind(spec.dtype),
          q: await loader.buffer(device, stem, usageRead),
          k,
          n,
          blocks: Math.ceil(k / ggufBlockSize(spec.dtype)),
          blockBytes: ggufBlockBytes(spec.dtype),
        };
      }
      throw new Error(`Llama WGSL tensor ${stem} is not a supported GGUF quantized tensor`);
    };

    let offset = 0;
    const embeddingChunks: Fp32Chunk[] = [];
    const embeddingQ4Chunks: Q4Chunk[] = [];
    for (const name of m.embeddingChunks) {
      const spec = loader.spec(name);
      const rows = spec.shape[0];
      if (embeddingKind === "gguf") {
        // GGUF embeddings are single quantized row-major tensors; rows are
        // counted here and loaded below through the generic GGUF weight path.
      } else if (embeddingKind === "q4") {
        embeddingQ4Chunks.push({
          q: await loader.buffer(device, name, usageRead),
          rows,
          offset,
        });
      } else {
        if (spec.shape[1] !== embeddingDim) {
          throw new Error(`Llama WGSL embedding ${name} width ${spec.shape[1]} != embeddingDim ${embeddingDim}`);
        }
        embeddingChunks.push({
          buffer: await loader.floatBuffer(device, name),
          rows,
          offset,
        });
      }
      offset += rows;
    }
    if (offset !== m.vocab) {
      throw new Error(`Llama WGSL embedding rows ${offset} != vocab ${m.vocab}`);
    }
    const q4EmbeddingTable = embeddingKind === "q4"
      ? {
          chunks: embeddingQ4Chunks,
          scales: await loader.floatBuffer(device, "model.embed_tokens.weight_scales"),
          zp: await loader.buffer(device, "model.embed_tokens.weight_zp", usageRead),
          blocks: Math.ceil(embeddingDim / m.blockSize),
          zpBlocks: Math.ceil(Math.ceil(embeddingDim / m.blockSize) / 2),
        }
      : null;
    const ggufEmbeddingTable = embeddingKind === "gguf"
      ? await q4(m.embeddingChunks[0], embeddingDim, m.vocab)
      : null;

    const layerHeads = validateLayerArray(m.layerHeads, m.layers, m.heads, "layerHeads");
    const layerKvHeads = validateLayerArray(m.layerKvHeads, m.layers, m.kvHeads, "layerKvHeads");
    const layerIntermediates = validateLayerArray(
      m.layerIntermediates,
      m.layers,
      m.intermediate,
      "layerIntermediates",
    );
    const layerRopeTheta = validateLayerArray(m.layerRopeTheta, m.layers, m.ropeTheta, "layerRopeTheta");
    const layerAttentionWindow = validateLayerArray(
      m.layerAttentionWindow,
      m.layers,
      0,
      "layerAttentionWindow",
    );
    const maxHeads = Math.max(...layerHeads);
    const maxKvHeads = Math.max(...layerKvHeads);
    const maxIntermediate = Math.max(...layerIntermediates);
    const maxQDim = maxHeads * m.headDim;
    const maxKvDim = maxKvHeads * m.headDim;
    const maxQkvDim = Math.max(...layerHeads.map((heads, i) => (heads + 2 * layerKvHeads[i]) * m.headDim));
    const isGemma = m.architecture === "gemma3";
    const useQkvPacked = m.attention === "qkv";
    const useGateUpPacked = m.mlp === "swiglu-packed";
    const loadLayer = async (i: number): Promise<LayerWeights> => {
      const heads = layerHeads[i];
      const kvHeads = layerKvHeads[i];
      const qDim = heads * m.headDim;
      const kvDim = kvHeads * m.headDim;
      const intermediate = layerIntermediates[i];
      const [
        inputNorm,
        postNorm,
        preFfnNorm,
        postFfnNorm,
        qNorm,
        kNorm,
        qkvProj,
        qProj,
        kProj,
        vProj,
        oProj,
        gateUp,
        gate,
        up,
        down,
      ] = await Promise.all([
        loader.floatBuffer(device, `model.layers.${i}.input_layernorm.weight`),
        loader.floatBuffer(device, `model.layers.${i}.post_attention_layernorm.weight`),
        isGemma ? loader.floatBuffer(device, `model.layers.${i}.pre_feedforward_layernorm.weight`) : Promise.resolve(undefined),
        isGemma ? loader.floatBuffer(device, `model.layers.${i}.post_feedforward_layernorm.weight`) : Promise.resolve(undefined),
        m.qkNorm ? loader.floatBuffer(device, `model.layers.${i}.attn.q_norm.layernorm.weight`) : Promise.resolve(undefined),
        m.qkNorm ? loader.floatBuffer(device, `model.layers.${i}.attn.k_norm.layernorm.weight`) : Promise.resolve(undefined),
        useQkvPacked ? q4(`model.layers.${i}.attn.qkv_proj.MatMul.weight`, m.hidden, qDim + 2 * kvDim) : Promise.resolve(undefined),
        useQkvPacked ? Promise.resolve(undefined) : q4(`model.layers.${i}.attn.q_proj.MatMul.weight`, m.hidden, qDim),
        useQkvPacked ? Promise.resolve(undefined) : q4(`model.layers.${i}.attn.k_proj.MatMul.weight`, m.hidden, kvDim),
        useQkvPacked ? Promise.resolve(undefined) : q4(`model.layers.${i}.attn.v_proj.MatMul.weight`, m.hidden, kvDim),
        q4(`model.layers.${i}.attn.o_proj.MatMul.weight`, qDim, m.hidden),
        useGateUpPacked ? q4(`model.layers.${i}.mlp.gate_up_proj.MatMul.weight`, m.hidden, intermediate * 2) : Promise.resolve(undefined),
        useGateUpPacked ? Promise.resolve(undefined) : q4(`model.layers.${i}.mlp.gate_proj.MatMul.weight`, m.hidden, intermediate),
        useGateUpPacked ? Promise.resolve(undefined) : q4(`model.layers.${i}.mlp.up_proj.MatMul.weight`, m.hidden, intermediate),
        q4(`model.layers.${i}.mlp.down_proj.MatMul.weight`, intermediate, m.hidden),
      ]);
      return {
        heads,
        kvHeads,
        kvDim,
        qDim,
        intermediate,
        ropeTheta: layerRopeTheta[i],
        attentionWindow: layerAttentionWindow[i],
        inputNorm,
        postNorm,
        preFfnNorm,
        postFfnNorm,
        qNorm,
        kNorm,
        qProj,
        kProj,
        vProj,
        qkvProj,
        oProj,
        gate,
        up,
        gateUp,
        down,
        keyCache: createEmptyBuffer(device, contextLength * kvDim * 4, usageState, `llama.layer.${i}.k_cache`),
        valueCache: createEmptyBuffer(device, contextLength * kvDim * 4, usageState, `llama.layer.${i}.v_cache`),
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

    onStep?.({ step: "weights", detail: "uploading lm head and scratch buffers" });
    const dispatchBudget = m.layers * 20 + m.embeddingChunks.length + 64;
    const paramBuffers = Array.from({ length: dispatchBudget }, (_, i) =>
      createEmptyBuffer(device, 64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, `llama.params.${i}`),
    );

    return new LlamaWgslEngine(device, manifest, {
      pipelines,
      paramBuffers,
      embeddingChunks,
      q4EmbeddingTable,
      ggufEmbeddingTable,
      embeddingProjQ4: embeddingDim === m.hidden ? null : await q4("embedding_proj.MatMul.weight", embeddingDim, m.hidden),
      finalNorm: await loader.floatBuffer(device, `model.layers.${m.layers}.final_norm_layernorm.weight`),
      lmHeadQ4: (m.lmHead ?? "embedding") === "q4"
        ? await q4("lm_head.MatMul.weight", m.hidden, m.vocab)
        : m.lmHead === "q4-projected"
          ? await q4("lm_head.MatMul.weight", embeddingDim, m.vocab)
          : null,
      lmHeadQ4Chunks: m.lmHead === "q4-chunked"
        ? await loadQ4ChunkedWeight(device, loader, m.lmHeadChunks ?? [], "lm_head.MatMul.weight", embeddingDim, usageRead)
        : null,
      lmHeadProjQ4: m.lmHead === "q4-projected"
        ? await q4("lm_head_proj.MatMul.weight", m.hidden, embeddingDim)
        : null,
      layers,
      readback: createEmptyBuffer(device, 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, "llama.argmax.readback"),
      embeddingScratch: createEmptyBuffer(device, embeddingDim * 4, usageScratch, "llama.embedding.scratch"),
      hiddenA: createEmptyBuffer(device, m.hidden * 4, usageScratch, "llama.hidden.a"),
      hiddenB: createEmptyBuffer(device, m.hidden * 4, usageScratch, "llama.hidden.b"),
      hiddenC: createEmptyBuffer(device, m.hidden * 4, usageScratch, "llama.hidden.c"),
      norm: createEmptyBuffer(device, m.hidden * 4, usageScratch, "llama.norm"),
      q: createEmptyBuffer(device, maxQDim * 4, usageScratch, "llama.q"),
      qkv: createEmptyBuffer(device, maxQkvDim * 4, usageScratch, "llama.qkv"),
      qNormed: createEmptyBuffer(device, maxQDim * 4, usageScratch, "llama.q_normed"),
      k: createEmptyBuffer(device, maxKvDim * 4, usageScratch, "llama.k"),
      v: createEmptyBuffer(device, maxKvDim * 4, usageScratch, "llama.v"),
      attnOut: createEmptyBuffer(device, maxQDim * 4, usageScratch, "llama.attn_out"),
      gate: createEmptyBuffer(device, maxIntermediate * 2 * 4, usageScratch, "llama.gate"),
      up: createEmptyBuffer(device, maxIntermediate * 4, usageScratch, "llama.up"),
      ff: createEmptyBuffer(device, maxIntermediate * 4, usageScratch, "llama.ff"),
      lmHeadScratch: createEmptyBuffer(device, embeddingDim * 4, usageScratch, "llama.lm_head.scratch"),
      logits: createEmptyBuffer(device, m.vocab * 4, usageScratch, "llama.logits"),
      penaltyIds: createEmptyBuffer(
        device,
        Math.max(4, contextLength * 4),
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        "llama.penalty_ids",
      ),
      argmaxScratch: createEmptyBuffer(
        device,
        Math.ceil(m.vocab / 256) * 8,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        "llama.argmax.scratch",
      ),
      argmaxResult: createEmptyBuffer(
        device,
        4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        "llama.argmax.result",
      ),
      candidateReadback: createEmptyBuffer(
        device,
        Math.ceil(m.vocab / 256) * 8,
        GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        "llama.candidates.readback",
      ),
    });
  }

  private constructor(
    private device: GPUDevice,
    manifest: LlamaManifest,
    buffers: {
      pipelines: Pipelines;
      paramBuffers: GPUBuffer[];
      embeddingChunks: Fp32Chunk[];
      q4EmbeddingTable: Q4Embedding | null;
      ggufEmbeddingTable: Q4Weight | null;
      embeddingProjQ4: Q4Weight | null;
      finalNorm: GPUBuffer;
      lmHeadQ4: Q4Weight | null;
      lmHeadQ4Chunks: Q4Embedding | null;
      lmHeadProjQ4: Q4Weight | null;
      layers: LayerWeights[];
      readback: GPUBuffer;
      embeddingScratch: GPUBuffer;
      hiddenA: GPUBuffer;
      hiddenB: GPUBuffer;
      hiddenC: GPUBuffer;
      norm: GPUBuffer;
      q: GPUBuffer;
      qkv: GPUBuffer;
      qNormed: GPUBuffer;
      k: GPUBuffer;
      v: GPUBuffer;
      attnOut: GPUBuffer;
      gate: GPUBuffer;
      up: GPUBuffer;
      ff: GPUBuffer;
      lmHeadScratch: GPUBuffer;
      logits: GPUBuffer;
      penaltyIds: GPUBuffer;
      argmaxScratch: GPUBuffer;
      argmaxResult: GPUBuffer;
      candidateReadback: GPUBuffer;
    },
  ) {
    this.m = manifest.model;
    this.pipelines = buffers.pipelines;
    const interleavedRope = manifest.model.ropeLayout === "interleaved";
    this.ropeStorePipeline = interleavedRope
      ? buffers.pipelines.ropeStoreInterleaved
      : buffers.pipelines.ropeStore;
    this.qkNormRopeStorePipeline = interleavedRope
      ? buffers.pipelines.qkNormRopeStoreInterleaved
      : buffers.pipelines.qkNormRopeStore;
    this.qkvNormRopeStorePipeline = interleavedRope
      ? buffers.pipelines.qkvNormRopeStoreInterleaved
      : buffers.pipelines.qkvNormRopeStore;
    this.paramBuffers = buffers.paramBuffers;
    this.embeddingChunks = buffers.embeddingChunks;
    this.q4EmbeddingTable = buffers.q4EmbeddingTable;
    this.ggufEmbeddingTable = buffers.ggufEmbeddingTable;
    this.embeddingProjQ4 = buffers.embeddingProjQ4;
    this.finalNorm = buffers.finalNorm;
    this.lmHeadQ4 = buffers.lmHeadQ4;
    this.lmHeadQ4Chunks = buffers.lmHeadQ4Chunks;
    this.lmHeadProjQ4 = buffers.lmHeadProjQ4;
    this.layers = buffers.layers;
    this.contextLength = this.layers[0].keyCache.size / (this.layers[0].kvDim * 4);
    this.readback = buffers.readback;
    this.embeddingScratch = buffers.embeddingScratch;
    this.hiddenA = buffers.hiddenA;
    this.hiddenB = buffers.hiddenB;
    this.hiddenC = buffers.hiddenC;
    this.norm = buffers.norm;
    this.q = buffers.q;
    this.qkv = buffers.qkv;
    this.qNormed = buffers.qNormed;
    this.k = buffers.k;
    this.v = buffers.v;
    this.attnOut = buffers.attnOut;
    this.gate = buffers.gate;
    this.up = buffers.up;
    this.ff = buffers.ff;
    this.lmHeadScratch = buffers.lmHeadScratch;
    this.logits = buffers.logits;
    this.penaltyIds = buffers.penaltyIds;
    this.argmaxScratch = buffers.argmaxScratch;
    this.argmaxResult = buffers.argmaxResult;
    this.candidateReadback = buffers.candidateReadback;
    this.scores = createEmptyBuffer(
      device,
      Math.max(...this.layers.map((layer) => layer.heads)) * this.contextLengthFromCache() * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      "llama.scores",
    );
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

  async runToken(
    tokenId: number,
    needLogits: boolean,
    opts: SamplingOptions & { penalizedTokenIds?: number[] } = {},
  ): Promise<number> {
    const contextLength = this.contextLengthFromCache();
    if (this.pos >= contextLength) {
      throw new Error(`Llama WGSL context exhausted at ${contextLength} tokens`);
    }

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    this.active = { encoder, pass, paramCursor: 0 };

    this.embedding(tokenId, this.hiddenA);
    let residual = this.hiddenA;
    let other = this.hiddenB;

    for (const layer of this.layers) {
      if (this.m.architecture === "gemma3") {
        this.rmsNorm(residual, layer.inputNorm, this.norm);
        if (!layer.qProj || !layer.kProj || !layer.vProj || !layer.qNorm || !layer.kNorm || !layer.preFfnNorm || !layer.postFfnNorm || !layer.gate || !layer.up) {
          throw new Error("Gemma WGSL layer is missing required weights");
        }
        this.q4Matvec(this.norm, layer.qProj, this.q);
        this.q4Matvec(this.norm, layer.kProj, this.k);
        this.q4Matvec(this.norm, layer.vProj, this.v);
        this.qkNormRopeStore(this.q, this.k, this.v, layer.qNorm, layer.kNorm, this.qNormed, layer.keyCache, layer.valueCache, layer);
        this.attention(layer.keyCache, layer.valueCache, this.qNormed, this.attnOut, layer);
        this.q4Matvec(this.attnOut, layer.oProj, this.hiddenB);
        this.rmsNorm(this.hiddenB, layer.postNorm, this.hiddenC);
        this.addClampInPlace(residual, this.hiddenC, this.m.hidden);
        residual = this.hiddenC;

        this.rmsNorm(residual, layer.preFfnNorm, this.norm);
        this.q4Matvec(this.norm, layer.gate, this.gate);
        this.q4Matvec(this.norm, layer.up, this.up);
        this.geluTanhMul(this.gate, this.up, this.ff, layer.intermediate);
        this.q4Matvec(this.ff, layer.down, this.hiddenB);
        this.rmsNorm(this.hiddenB, layer.postFfnNorm, this.hiddenA);
        this.addClampInPlace(residual, this.hiddenA, this.m.hidden);
        residual = this.hiddenA;
        other = this.hiddenB;
      } else {
        this.rmsNorm(residual, layer.inputNorm, this.norm);
        let qForAttention: GPUBuffer;
        if (layer.qkvProj && layer.qNorm && layer.kNorm) {
          this.q4Matvec(this.norm, layer.qkvProj, this.qkv);
          this.qkvNormRopeStore(this.qkv, layer.qNorm, layer.kNorm, this.qNormed, layer.keyCache, layer.valueCache, layer);
          qForAttention = this.qNormed;
        } else if (layer.qProj && layer.kProj && layer.vProj) {
          this.q4Matvec(this.norm, layer.qProj, this.q);
          this.q4Matvec(this.norm, layer.kProj, this.k);
          this.q4Matvec(this.norm, layer.vProj, this.v);
          qForAttention = layer.qNorm && layer.kNorm
            ? (this.qkNormRopeStore(this.q, this.k, this.v, layer.qNorm, layer.kNorm, this.qNormed, layer.keyCache, layer.valueCache, layer), this.qNormed)
            : (this.ropeStore(this.q, this.k, this.v, layer.keyCache, layer.valueCache, layer), this.q);
        } else {
          throw new Error("Llama WGSL layer is missing attention projection weights");
        }
        this.attention(layer.keyCache, layer.valueCache, qForAttention, this.attnOut, layer);
        this.q4Matvec(this.attnOut, layer.oProj, other);
        this.addRmsNorm(residual, other, layer.postNorm, this.norm, this.m.hidden);
        [residual, other] = [other, residual];

        if (layer.gateUp) {
          this.q4Matvec(this.norm, layer.gateUp, this.gate);
          this.siluSplitMul(this.gate, this.ff, layer.intermediate);
        } else if (layer.gate && layer.up) {
          this.q4Matvec(this.norm, layer.gate, this.gate);
          this.q4Matvec(this.norm, layer.up, this.up);
          this.siluMul(this.gate, this.up, this.ff, layer.intermediate);
        } else {
          throw new Error("Llama WGSL layer is missing MLP projection weights");
        }
        this.q4Matvec(this.ff, layer.down, other);
        this.addInPlace(residual, other, this.m.hidden);
        [residual, other] = [other, residual];
      }
    }

    const sampleFromCandidates = needLogits && needsSamplingReadback(opts);
    if (needLogits) {
      this.rmsNorm(residual, this.finalNorm, this.norm);
      if (this.lmHeadProjQ4 && this.lmHeadQ4) {
        this.q4Matvec(this.norm, this.lmHeadProjQ4, this.lmHeadScratch);
        this.q4Matvec(this.lmHeadScratch, this.lmHeadQ4, this.logits);
      } else if (this.lmHeadQ4) {
        this.q4Matvec(this.norm, this.lmHeadQ4, this.logits);
      } else if (this.lmHeadQ4Chunks) {
        for (const chunk of this.lmHeadQ4Chunks.chunks) {
          this.q4MatvecChunk(this.norm, this.lmHeadQ4Chunks, chunk, this.logits);
        }
      } else {
        for (const chunk of this.embeddingChunks) {
          this.lmHead(this.norm, chunk, this.logits);
        }
      }
      this.argmax(opts.penalizedTokenIds ?? [], opts.repetitionPenalty ?? 1, sampleFromCandidates);
    }

    pass.end();
    if (needLogits) {
      if (sampleFromCandidates) {
        encoder.copyBufferToBuffer(this.argmaxScratch, 0, this.candidateReadback, 0, Math.ceil(this.m.vocab / 256) * 8);
      } else {
        encoder.copyBufferToBuffer(this.argmaxResult, 0, this.readback, 0, 4);
      }
    }
    this.device.queue.submit([encoder.finish()]);
    this.active = null;

    const best = needLogits
      ? sampleFromCandidates
        ? await this.readCandidatesMapped(opts)
        : await this.readArgmaxMapped()
      : 0;
    if (!needLogits) {
      // Param buffers are reused from slot 0 on every token. The command buffer
      // stores references to those buffers, not snapshots of their contents, so
      // prompt-prefill steps must finish before the next token overwrites them.
      await this.device.queue.onSubmittedWorkDone();
    }
    this.pos += 1;
    return best;
  }

  dispose(): void {
    const destroyQ4 = (w: Q4Weight) => {
      w.q.destroy();
    };
    const destroyQ4Chunked = (w: Q4Embedding) => {
      for (const chunk of w.chunks) chunk.q.destroy();
      w.scales.destroy();
      w.zp.destroy();
    };
    for (const chunk of this.embeddingChunks) chunk.buffer.destroy();
    if (this.q4EmbeddingTable) destroyQ4Chunked(this.q4EmbeddingTable);
    if (this.ggufEmbeddingTable) destroyQ4(this.ggufEmbeddingTable);
    if (this.embeddingProjQ4) destroyQ4(this.embeddingProjQ4);
    this.finalNorm.destroy();
    if (this.lmHeadQ4) destroyQ4(this.lmHeadQ4);
    if (this.lmHeadQ4Chunks) destroyQ4Chunked(this.lmHeadQ4Chunks);
    if (this.lmHeadProjQ4) destroyQ4(this.lmHeadProjQ4);
    for (const layer of this.layers) {
      layer.inputNorm.destroy();
      layer.postNorm.destroy();
      layer.preFfnNorm?.destroy();
      layer.postFfnNorm?.destroy();
      layer.qNorm?.destroy();
      layer.kNorm?.destroy();
      if (layer.qProj) destroyQ4(layer.qProj);
      if (layer.kProj) destroyQ4(layer.kProj);
      if (layer.vProj) destroyQ4(layer.vProj);
      if (layer.qkvProj) destroyQ4(layer.qkvProj);
      destroyQ4(layer.oProj);
      if (layer.gate) destroyQ4(layer.gate);
      if (layer.up) destroyQ4(layer.up);
      if (layer.gateUp) destroyQ4(layer.gateUp);
      destroyQ4(layer.down);
      layer.keyCache.destroy();
      layer.valueCache.destroy();
    }
    for (const b of [
      ...this.paramBuffers,
      this.readback,
      this.embeddingScratch,
      this.hiddenA,
      this.hiddenB,
      this.hiddenC,
      this.norm,
      this.q,
      this.qkv,
      this.qNormed,
      this.k,
      this.v,
      this.attnOut,
      this.gate,
      this.up,
      this.ff,
      this.lmHeadScratch,
      this.logits,
      this.penaltyIds,
      this.scores,
      this.argmaxScratch,
      this.argmaxResult,
      this.candidateReadback,
    ]) {
      b.destroy();
    }
  }

  private contextLengthFromCache(): number {
    return this.contextLength;
  }

  private param(values: number[]): GPUBuffer {
    if (!this.active) throw new Error("Llama WGSL dispatch outside active pass");
    const buffer = this.paramBuffers[this.active.paramCursor++];
    if (!buffer) throw new Error("Llama WGSL param buffer budget exhausted");
    const data = new Uint32Array(16);
    data.set(values.slice(0, 16));
    this.device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }

  private dispatch(
    pipeline: GPUComputePipeline,
    entries: GPUBindGroupEntry[],
    x: number,
    y = 1,
    z = 1,
  ): void {
    if (!this.active) throw new Error("Llama WGSL dispatch outside active pass");
    const key = this.bindGroupKey(pipeline, entries);
    let bindGroup = this.bindGroups.get(key);
    if (!bindGroup) {
      bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries,
      });
      this.bindGroups.set(key, bindGroup);
    }
    this.active.pass.setPipeline(pipeline);
    this.active.pass.setBindGroup(0, bindGroup);
    this.active.pass.dispatchWorkgroups(x, y, z);
  }

  private bindGroupKey(pipeline: GPUComputePipeline, entries: GPUBindGroupEntry[]): string {
    const parts = [String(this.pipelineId(pipeline))];
    for (const entry of entries) {
      const resource = entry.resource as GPUBufferBinding;
      parts.push(
        `${entry.binding}:${this.bufferId(resource.buffer)}:${resource.offset ?? 0}:${resource.size ?? 0}`,
      );
    }
    return parts.join("|");
  }

  private bufferId(buffer: GPUBuffer): number {
    let id = this.bufferIds.get(buffer);
    if (!id) {
      id = this.nextBufferId++;
      this.bufferIds.set(buffer, id);
    }
    return id;
  }

  private pipelineId(pipeline: GPUComputePipeline): number {
    let id = this.pipelineIds.get(pipeline);
    if (!id) {
      id = this.nextPipelineId++;
      this.pipelineIds.set(pipeline, id);
    }
    return id;
  }

  private embedding(tokenId: number, out: GPUBuffer): void {
    if (this.ggufEmbeddingTable) {
      this.ggufEmbedding(tokenId, this.ggufEmbeddingTable, out);
      return;
    }
    if (this.q4EmbeddingTable) {
      const chunk = this.findQ4Chunk(this.q4EmbeddingTable, tokenId);
      this.dispatch(
        this.pipelines.q4Embedding,
        [
          { binding: 0, resource: { buffer: chunk.q } },
          { binding: 1, resource: { buffer: this.q4EmbeddingTable.scales } },
          { binding: 2, resource: { buffer: this.q4EmbeddingTable.zp } },
          { binding: 3, resource: { buffer: out } },
          {
            binding: 4,
            resource: {
              buffer: this.param([
                tokenId - chunk.offset,
                tokenId,
                this.q4EmbeddingTable.blocks,
                this.q4EmbeddingTable.zpBlocks,
              ]),
            },
          },
        ],
        Math.ceil((this.m.embeddingDim ?? this.m.hidden) / 128),
      );
      return;
    }
    const chunk = this.findEmbeddingChunk(tokenId);
    const embeddingOut = this.embeddingProjQ4 ? this.embeddingScratch : out;
    this.dispatch(
      this.pipelines.embedding,
      [
        { binding: 0, resource: { buffer: chunk.buffer } },
        { binding: 1, resource: { buffer: embeddingOut } },
        { binding: 2, resource: { buffer: this.param([tokenId - chunk.offset, chunk.rows]) } },
      ],
      Math.ceil((this.m.embeddingDim ?? this.m.hidden) / 128),
    );
    if (this.embeddingProjQ4) this.q4Matvec(this.embeddingScratch, this.embeddingProjQ4, out);
  }

  private ggufEmbedding(tokenId: number, weight: Q4Weight, out: GPUBuffer): void {
    const pipeline = this.ggufEmbeddingPipeline(weight);
    this.dispatch(
      pipeline,
      [
        { binding: 0, resource: { buffer: weight.q } },
        { binding: 1, resource: { buffer: out } },
        {
          binding: 2,
          resource: {
            buffer: this.param([tokenId, weight.k, weight.blockBytes]),
          },
        },
      ],
      Math.ceil(weight.k / 128),
    );
  }

  private findEmbeddingChunk(tokenId: number): Fp32Chunk {
    const chunk = this.embeddingChunks.find((c) => tokenId >= c.offset && tokenId < c.offset + c.rows);
    if (!chunk) throw new Error(`token id ${tokenId} outside embedding table`);
    return chunk;
  }

  private findQ4Chunk(table: Q4Embedding, tokenId: number): Q4Chunk {
    const chunk = table.chunks.find((c) => tokenId >= c.offset && tokenId < c.offset + c.rows);
    if (!chunk) throw new Error(`token id ${tokenId} outside q4 table`);
    return chunk;
  }

  private rmsNorm(input: GPUBuffer, weight: GPUBuffer, out: GPUBuffer): void {
    this.dispatch(
      this.pipelines.rmsNorm,
      [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: weight } },
        { binding: 2, resource: { buffer: out } },
        { binding: 3, resource: { buffer: this.param([this.m.hidden]) } },
      ],
      1,
    );
  }

  private addRmsNorm(a: GPUBuffer, io: GPUBuffer, weight: GPUBuffer, out: GPUBuffer, n: number): void {
    this.dispatch(
      this.pipelines.addRmsNorm,
      [
        { binding: 0, resource: { buffer: a } },
        { binding: 1, resource: { buffer: io } },
        { binding: 2, resource: { buffer: weight } },
        { binding: 3, resource: { buffer: out } },
        { binding: 4, resource: { buffer: this.param([n]) } },
      ],
      1,
    );
  }

  private q4Matvec(input: GPUBuffer, weight: Q4Weight, out: GPUBuffer): void {
    const xGroups = Math.min(weight.n, 32768);
    const yGroups = Math.ceil(weight.n / xGroups);
    this.dispatch(
      this.ggufMatvecPipeline(weight),
      [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: weight.q } },
        { binding: 2, resource: { buffer: out } },
        { binding: 3, resource: { buffer: this.param([weight.k, weight.n, weight.blockBytes, xGroups]) } },
      ],
      xGroups,
      yGroups,
    );
  }

  private ggufMatvecPipeline(weight: Q4Weight): GPUComputePipeline {
    switch (weight.kind) {
      case "gguf-q5_0":
        return this.pipelines.q5Matvec;
      case "gguf-q5_1":
        return this.pipelines.q5_1Matvec;
      case "gguf-q8_0":
        return this.pipelines.q8Matvec;
      case "gguf-q4_k":
        return this.pipelines.q4KMatvec;
      case "gguf-q5_k":
        return this.pipelines.q5KMatvec;
      case "gguf-q6_k":
        return this.pipelines.q6KMatvec;
      default:
        throw new Error(`not a GGUF matvec weight: ${weight.kind}`);
    }
  }

  private ggufEmbeddingPipeline(weight: Q4Weight): GPUComputePipeline {
    switch (weight.kind) {
      case "gguf-q5_0":
        return this.pipelines.q5Embedding;
      case "gguf-q5_1":
        return this.pipelines.q5_1Embedding;
      case "gguf-q8_0":
        return this.pipelines.q8Embedding;
      case "gguf-q4_k":
        return this.pipelines.q4KEmbedding;
      case "gguf-q5_k":
        return this.pipelines.q5KEmbedding;
      case "gguf-q6_k":
        return this.pipelines.q6KEmbedding;
      default:
        throw new Error(`not a GGUF embedding weight: ${weight.kind}`);
    }
  }

  private q4MatvecChunk(input: GPUBuffer, table: Q4Embedding, chunk: Q4Chunk, out: GPUBuffer): void {
    const xGroups = Math.min(chunk.rows, 32768);
    const yGroups = Math.ceil(chunk.rows / xGroups);
    this.dispatch(
      this.pipelines.q4MatvecZpOffset,
      [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: chunk.q } },
        { binding: 2, resource: { buffer: table.scales } },
        { binding: 3, resource: { buffer: table.zp } },
        { binding: 4, resource: { buffer: out } },
        {
          binding: 5,
          resource: {
            buffer: this.param([
              this.m.hidden,
              chunk.rows,
              table.blocks,
              xGroups,
              chunk.offset,
              table.zpBlocks,
            ]),
          },
        },
      ],
      xGroups,
      yGroups,
    );
  }

  private ropeStore(q: GPUBuffer, k: GPUBuffer, v: GPUBuffer, keyCache: GPUBuffer, valueCache: GPUBuffer, layer: LayerWeights): void {
    const n = Math.max((layer.heads * this.m.headDim) / 2, layer.kvDim);
    this.dispatch(
      this.ropeStorePipeline,
      [
        { binding: 0, resource: { buffer: q } },
        { binding: 1, resource: { buffer: k } },
        { binding: 2, resource: { buffer: v } },
        { binding: 3, resource: { buffer: keyCache } },
        { binding: 4, resource: { buffer: valueCache } },
        { binding: 5, resource: { buffer: this.param([this.pos, this.contextLengthFromCache(), layer.heads, layer.kvHeads, Math.round(layer.ropeTheta)]) } },
      ],
      Math.ceil(n / 128),
    );
  }

  private qkNormRopeStore(
    q: GPUBuffer,
    k: GPUBuffer,
    v: GPUBuffer,
    qNorm: GPUBuffer,
    kNorm: GPUBuffer,
    qOut: GPUBuffer,
    keyCache: GPUBuffer,
    valueCache: GPUBuffer,
    layer: LayerWeights,
  ): void {
    this.dispatch(
      this.qkNormRopeStorePipeline,
      [
        { binding: 0, resource: { buffer: q } },
        { binding: 1, resource: { buffer: k } },
        { binding: 2, resource: { buffer: v } },
        { binding: 3, resource: { buffer: qNorm } },
        { binding: 4, resource: { buffer: kNorm } },
        { binding: 5, resource: { buffer: qOut } },
        { binding: 6, resource: { buffer: keyCache } },
        { binding: 7, resource: { buffer: valueCache } },
        { binding: 8, resource: { buffer: this.param([this.pos, this.contextLengthFromCache(), layer.heads, layer.kvHeads, Math.round(layer.ropeTheta)]) } },
      ],
      layer.heads,
    );
  }

  private qkvNormRopeStore(
    qkv: GPUBuffer,
    qNorm: GPUBuffer,
    kNorm: GPUBuffer,
    qOut: GPUBuffer,
    keyCache: GPUBuffer,
    valueCache: GPUBuffer,
    layer: LayerWeights,
  ): void {
    this.dispatch(
      this.qkvNormRopeStorePipeline,
      [
        { binding: 0, resource: { buffer: qkv } },
        { binding: 1, resource: { buffer: qNorm } },
        { binding: 2, resource: { buffer: kNorm } },
        { binding: 3, resource: { buffer: qOut } },
        { binding: 4, resource: { buffer: keyCache } },
        { binding: 5, resource: { buffer: valueCache } },
        {
          binding: 6,
          resource: {
            buffer: this.param([
              this.pos,
              this.contextLengthFromCache(),
              layer.heads,
              layer.kvHeads,
              layer.qDim,
              layer.kvDim,
              Math.round(layer.ropeTheta),
            ]),
          },
        },
      ],
      layer.heads,
    );
  }

  private attention(keyCache: GPUBuffer, valueCache: GPUBuffer, q: GPUBuffer, out: GPUBuffer, layer: LayerWeights): void {
    this.dispatch(
      this.pipelines.attentionScore,
      [
        { binding: 0, resource: { buffer: q } },
        { binding: 1, resource: { buffer: keyCache } },
        { binding: 2, resource: { buffer: this.scores } },
        { binding: 3, resource: { buffer: this.param([this.pos, this.contextLengthFromCache(), layer.heads, layer.kvHeads, layer.attentionWindow]) } },
      ],
      layer.heads,
      this.pos + 1,
    );
    this.dispatch(
      this.pipelines.attentionValue,
      [
        { binding: 0, resource: { buffer: this.scores } },
        { binding: 1, resource: { buffer: valueCache } },
        { binding: 2, resource: { buffer: out } },
        { binding: 3, resource: { buffer: this.param([this.pos, this.contextLengthFromCache(), layer.heads, layer.kvHeads, layer.attentionWindow]) } },
      ],
      layer.heads,
    );
  }

  private addInPlace(a: GPUBuffer, io: GPUBuffer, n: number): void {
    this.dispatch(
      this.pipelines.add,
      [
        { binding: 0, resource: { buffer: a } },
        { binding: 1, resource: { buffer: io } },
        { binding: 2, resource: { buffer: this.param([n]) } },
      ],
      Math.ceil(n / 128),
    );
  }

  private addClampInPlace(a: GPUBuffer, io: GPUBuffer, n: number): void {
    this.dispatch(
      this.pipelines.addClamp,
      [
        { binding: 0, resource: { buffer: a } },
        { binding: 1, resource: { buffer: io } },
        { binding: 2, resource: { buffer: this.param([n]) } },
      ],
      Math.ceil(n / 128),
    );
  }

  private siluMul(gate: GPUBuffer, up: GPUBuffer, out: GPUBuffer, n: number): void {
    this.dispatch(
      this.pipelines.siluMul,
      [
        { binding: 0, resource: { buffer: gate } },
        { binding: 1, resource: { buffer: up } },
        { binding: 2, resource: { buffer: out } },
        { binding: 3, resource: { buffer: this.param([n]) } },
      ],
      Math.ceil(n / 128),
    );
  }

  private siluSplitMul(gateUp: GPUBuffer, out: GPUBuffer, n: number): void {
    this.dispatch(
      this.pipelines.siluSplitMul,
      [
        { binding: 0, resource: { buffer: gateUp } },
        { binding: 1, resource: { buffer: out } },
        { binding: 2, resource: { buffer: this.param([n]) } },
      ],
      Math.ceil(n / 128),
    );
  }

  private geluTanhMul(gate: GPUBuffer, up: GPUBuffer, out: GPUBuffer, n: number): void {
    this.dispatch(
      this.pipelines.geluTanhMul,
      [
        { binding: 0, resource: { buffer: gate } },
        { binding: 1, resource: { buffer: up } },
        { binding: 2, resource: { buffer: out } },
        { binding: 3, resource: { buffer: this.param([n]) } },
      ],
      Math.ceil(n / 128),
    );
  }

  private lmHead(hidden: GPUBuffer, chunk: Fp32Chunk, logits: GPUBuffer): void {
    this.dispatch(
      this.pipelines.lmHead,
      [
        { binding: 0, resource: { buffer: hidden } },
        { binding: 1, resource: { buffer: chunk.buffer } },
        { binding: 2, resource: { buffer: logits } },
        { binding: 3, resource: { buffer: this.param([chunk.rows, chunk.offset]) } },
      ],
      chunk.rows,
    );
  }

  private argmax(penalizedTokenIds: number[] = [], repetitionPenalty = 1, candidatesOnly = false): void {
    const groups = Math.ceil(this.m.vocab / 256);
    const applyPenalty = repetitionPenalty > 1 && penalizedTokenIds.length > 0;
    const ids = applyPenalty
      ? penalizedTokenIds.slice(-this.contextLengthFromCache())
      : [];
    if (ids.length) {
      this.device.queue.writeBuffer(this.penaltyIds, 0, new Uint32Array(ids));
    }
    this.dispatch(
      this.pipelines.argmaxStage1,
      [
        { binding: 0, resource: { buffer: this.logits } },
        { binding: 1, resource: { buffer: this.argmaxScratch } },
        { binding: 2, resource: { buffer: this.param([this.m.vocab, ids.length, Math.round(repetitionPenalty * 1000)]) } },
        { binding: 3, resource: { buffer: this.penaltyIds } },
      ],
      groups,
    );
    if (!candidatesOnly) {
      this.dispatch(
        this.pipelines.argmaxStage2,
        [
          { binding: 0, resource: { buffer: this.argmaxScratch } },
          { binding: 1, resource: { buffer: this.argmaxResult } },
          { binding: 2, resource: { buffer: this.param([groups]) } },
        ],
        1,
      );
    }
  }

  private async readArgmaxMapped(): Promise<number> {
    await this.readback.mapAsync(GPUMapMode.READ);
    const best = new Uint32Array(this.readback.getMappedRange())[0];
    this.readback.unmap();
    return best;
  }

  private async readCandidatesMapped(sampling: SamplingOptions): Promise<number> {
    const groups = Math.ceil(this.m.vocab / 256);
    await this.candidateReadback.mapAsync(GPUMapMode.READ);
    const best = sampleFromCandidateBuffer(this.candidateReadback.getMappedRange(), sampling, groups);
    this.candidateReadback.unmap();
    return best;
  }
}

function createPipelines(device: GPUDevice, m: LlamaManifest["model"]): Pipelines {
  const constants = `
const HIDDEN: u32 = ${m.hidden}u;
const EMBEDDING_DIM: u32 = ${m.embeddingDim ?? m.hidden}u;
const INTERMEDIATE: u32 = ${m.intermediate}u;
const HEADS: u32 = ${m.heads}u;
const KV_HEADS: u32 = ${m.kvHeads}u;
const HEAD_DIM: u32 = ${m.headDim}u;
const ROTARY_HALF: u32 = ${Math.floor(m.headDim / 2)}u;
const KV_GROUP: u32 = ${Math.floor(m.heads / m.kvHeads)}u;
const KV_DIM: u32 = ${m.kvHeads * m.headDim}u;
const ROPE_THETA: f32 = ${m.ropeTheta.toFixed(1)};
const NORM_EPS: f32 = ${m.normEps};
const QK_NORM_EPS: f32 = ${m.qkNormEps ?? m.normEps};
const ATTN_SCALE: f32 = ${1 / Math.sqrt(m.headDim)};
const EMBEDDING_SCALE: f32 = ${m.embeddingScale ?? 1};
`;
  const module = (label: string, code: string) => device.createShaderModule({ label, code: constants + code });
  const pipeline = (label: string, code: string) =>
    device.createComputePipeline({
      label,
      layout: "auto",
      compute: { module: module(label, code), entryPoint: "main" },
    });

  return {
    embedding: pipeline("llama.embedding", EMBEDDING_WGSL),
    rmsNorm: pipeline("llama.rms_norm", RMS_NORM_WGSL),
    addRmsNorm: pipeline("llama.add_rms_norm", ADD_RMS_NORM_WGSL),
    q4Matvec: pipeline("llama.q4_matvec", Q4_MATVEC_WGSL),
    q4MatvecZp: pipeline("llama.q4_matvec_zp", Q4_MATVEC_ZP_WGSL),
    q5Matvec: pipeline("llama.q5_matvec", GGUF_Q5_0_MATVEC_WGSL),
    q5_1Matvec: pipeline("llama.q5_1_matvec", GGUF_Q5_1_MATVEC_WGSL),
    q8Matvec: pipeline("llama.q8_matvec", GGUF_Q8_0_MATVEC_WGSL),
    q4KMatvec: pipeline("llama.q4_k_matvec", GGUF_Q4_K_MATVEC_WGSL),
    q5KMatvec: pipeline("llama.q5_k_matvec", GGUF_Q5_K_MATVEC_WGSL),
    q6KMatvec: pipeline("llama.q6_k_matvec", GGUF_Q6_K_MATVEC_WGSL),
    ropeStore: pipeline("llama.rope_store", ROPE_STORE_WGSL),
    qkNormRopeStore: pipeline("llama.qk_norm_rope_store", QK_NORM_ROPE_STORE_WGSL),
    qkvNormRopeStore: pipeline("llama.qkv_norm_rope_store", QKV_NORM_ROPE_STORE_WGSL),
    ropeStoreInterleaved: pipeline("llama.rope_store_interleaved", ROPE_STORE_INTERLEAVED_WGSL),
    qkNormRopeStoreInterleaved: pipeline("llama.qk_norm_rope_store_interleaved", QK_NORM_ROPE_STORE_INTERLEAVED_WGSL),
    qkvNormRopeStoreInterleaved: pipeline("llama.qkv_norm_rope_store_interleaved", QKV_NORM_ROPE_STORE_INTERLEAVED_WGSL),
    attentionScore: pipeline("llama.attention_score", ATTENTION_SCORE_WGSL),
    attentionValue: pipeline("llama.attention_value", ATTENTION_VALUE_WGSL),
    add: pipeline("llama.add", ADD_WGSL),
    addClamp: pipeline("llama.add_clamp", ADD_CLAMP_WGSL),
    siluMul: pipeline("llama.silu_mul", SILU_MUL_WGSL),
    siluSplitMul: pipeline("llama.silu_split_mul", SILU_SPLIT_MUL_WGSL),
    geluTanhMul: pipeline("llama.gelu_tanh_mul", GELU_TANH_MUL_WGSL),
    q4Embedding: pipeline("llama.q4_embedding", Q4_EMBEDDING_WGSL),
    q5Embedding: pipeline("llama.q5_embedding", GGUF_Q5_0_EMBEDDING_WGSL),
    q5_1Embedding: pipeline("llama.q5_1_embedding", GGUF_Q5_1_EMBEDDING_WGSL),
    q8Embedding: pipeline("llama.q8_embedding", GGUF_Q8_0_EMBEDDING_WGSL),
    q4KEmbedding: pipeline("llama.q4_k_embedding", GGUF_Q4_K_EMBEDDING_WGSL),
    q5KEmbedding: pipeline("llama.q5_k_embedding", GGUF_Q5_K_EMBEDDING_WGSL),
    q6KEmbedding: pipeline("llama.q6_k_embedding", GGUF_Q6_K_EMBEDDING_WGSL),
    q4MatvecZpOffset: pipeline("llama.q4_matvec_zp_offset", Q4_MATVEC_ZP_OFFSET_WGSL),
    lmHead: pipeline("llama.lm_head", LM_HEAD_WGSL),
    argmaxStage1: pipeline("llama.argmax_stage1", ARGMAX_STAGE1_WGSL),
    argmaxStage2: pipeline("llama.argmax_stage2", ARGMAX_STAGE2_WGSL),
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

function align4(n: number): number {
  return (n + 3) & ~3;
}

async function loadQ4ChunkedWeight(
  device: GPUDevice,
  loader: TensorLoader,
  chunks: string[],
  stem: string,
  k: number,
  usage: GPUBufferUsageFlags,
): Promise<Q4Embedding> {
  if (!chunks.length) throw new Error(`Llama WGSL ${stem} has no q4 chunks`);
  const q4Chunks: Q4Chunk[] = [];
  let offset = 0;
  for (const name of chunks) {
    const spec = loader.spec(name);
    q4Chunks.push({
      q: await loader.buffer(device, name, usage),
      rows: spec.shape[0],
      offset,
    });
    offset += spec.shape[0];
  }
  const blocks = Math.ceil(k / 32);
  return {
    chunks: q4Chunks,
    scales: await loader.floatBuffer(device, `${stem}_scales`),
    zp: await loader.buffer(device, `${stem}_zp`, usage),
    blocks,
    zpBlocks: Math.ceil(blocks / 2),
  };
}

function validateLayerArray(values: number[] | undefined, layers: number, fallback: number, name: string): number[] {
  if (!values) return Array.from({ length: layers }, () => fallback);
  if (values.length !== layers) {
    throw new Error(`Llama WGSL manifest ${name} length ${values.length} != layers ${layers}`);
  }
  return values;
}

function commonPrefixLength(a: readonly number[], b: readonly number[]): number {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a[i] === b[i]) i++;
  return i;
}

function base64Bytes(s: string): Uint8Array {
  const binary = atob(s);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}

function f16ToF32(h: number): number {
  const sign = (h & 0x8000) ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x03ff;
  if (exp === 0) return sign * Math.pow(2, -14) * (frac / 1024);
  if (exp === 0x1f) return frac ? NaN : sign * Infinity;
  return sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

const EMBEDDING_WGSL = /* wgsl */ `
struct Params {
  token: u32,
  rows: u32,
};

@group(0) @binding(0) var<storage, read> embedding: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h = gid.x;
  if (h < EMBEDDING_DIM && params.token < params.rows) {
    output[h] = embedding[params.token * EMBEDDING_DIM + h];
  }
}
`;

const Q4_EMBEDDING_WGSL = /* wgsl */ `
struct Params {
  local_token: u32,
  global_token: u32,
  blocks: u32,
  zp_blocks: u32,
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
  if (k >= EMBEDDING_DIM) {
    return;
  }
  let block = k >> 5u;
  let within = k & 31u;
  let q_index = params.local_token * params.blocks * 16u + block * 16u + (within >> 1u);
  let q_byte = q_byte_at(q_index);
  let qv = select(q_byte >> 4u, q_byte & 15u, (within & 1u) == 0u);
  let zp_index = params.global_token * params.zp_blocks + (block >> 1u);
  let zp_byte = zp_byte_at(zp_index);
  let zv = select(zp_byte >> 4u, zp_byte & 15u, (block & 1u) == 0u);
  let scale = scales[params.global_token * params.blocks + block];
  output[k] = f32(i32(qv) - i32(zv)) * scale * EMBEDDING_SCALE;
}
`;

const RMS_NORM_WGSL = /* wgsl */ `
struct Params {
  n: u32,
};

@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  var sum = 0.0;
  for (var i = lid.x; i < params.n; i = i + 256u) {
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

  let inv = inverseSqrt(partial[0] / f32(params.n) + NORM_EPS);
  for (var i = lid.x; i < params.n; i = i + 256u) {
    output[i] = input[i] * inv * weight[i];
  }
}
`;

const ADD_RMS_NORM_WGSL = /* wgsl */ `
struct Params {
  n: u32,
};

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> io: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  var sum = 0.0;
  for (var i = lid.x; i < params.n; i = i + 256u) {
    let v = a[i] + io[i];
    io[i] = v;
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

  let inv = inverseSqrt(partial[0] / f32(params.n) + NORM_EPS);
  for (var i = lid.x; i < params.n; i = i + 256u) {
    output[i] = io[i] * inv * weight[i];
  }
}
`;

const Q4_MATVEC_WGSL = /* wgsl */ `
struct Params {
  k: u32,
  n: u32,
  blocks: u32,
  x_groups: u32,
};

@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read> q: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> partial: array<f32, 128>;

fn q_byte_at(index: u32) -> u32 {
  let word = q[index >> 2u];
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
    let w = f32(i32(qv) - 8) * scales[n * params.blocks + block];
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

const Q4_MATVEC_ZP_WGSL = /* wgsl */ `
struct Params {
  k: u32,
  n: u32,
  blocks: u32,
  x_groups: u32,
};

@group(0) @binding(0) var<storage, read_write> input: array<f32>;
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

    let zp_index = n * ((params.blocks + 1u) >> 1u) + (block >> 1u);
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

const Q4_MATVEC_ZP_OFFSET_WGSL = /* wgsl */ `
struct Params {
  k: u32,
  n: u32,
  blocks: u32,
  x_groups: u32,
  row_offset: u32,
  zp_blocks: u32,
};

@group(0) @binding(0) var<storage, read_write> input: array<f32>;
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
  let local_n = wg.x + wg.y * params.x_groups;
  if (local_n >= params.n) {
    return;
  }
  let global_n = params.row_offset + local_n;

  var sum = 0.0;
  for (var k = lid.x; k < params.k; k = k + 128u) {
    let block = k >> 5u;
    let within = k & 31u;
    let q_index = local_n * params.blocks * 16u + block * 16u + (within >> 1u);
    let q_byte = q_byte_at(q_index);
    let qv = select(q_byte >> 4u, q_byte & 15u, (within & 1u) == 0u);

    let zp_index = global_n * params.zp_blocks + (block >> 1u);
    let zp_byte = zp_byte_at(zp_index);
    let zv = select(zp_byte >> 4u, zp_byte & 15u, (block & 1u) == 0u);
    let w = f32(i32(qv) - i32(zv)) * scales[global_n * params.blocks + block];
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
    output[global_n] = partial[0];
  }
}
`;

const GGUF_Q5_0_MATVEC_WGSL = /* wgsl */ `
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

fn u32_at(index: u32) -> u32 {
  return byte_at(index) |
    (byte_at(index + 1u) << 8u) |
    (byte_at(index + 2u) << 16u) |
    (byte_at(index + 3u) << 24u);
}

fn q5_0_weight(row: u32, col: u32) -> f32 {
  let blocks = params.k / 32u;
  let block = col >> 5u;
  let within = col & 31u;
  let base = row * blocks * params.block_bytes + block * params.block_bytes;
  let qh = u32_at(base + 2u);
  let q_byte = byte_at(base + 6u + (within & 15u));
  let low4 = select(q_byte & 15u, q_byte >> 4u, within >= 16u);
  let high = (qh >> within) & 1u;
  let qv = low4 | (high << 4u);
  return f16_at(base) * f32(i32(qv) - 16);
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = wg.x + wg.y * params.x_groups;
  if (n >= params.n) {
    return;
  }

  var sum = 0.0;
  for (var k = lid.x; k < params.k; k = k + 128u) {
    sum = sum + input[k] * q5_0_weight(n, k);
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

const GGUF_Q5_1_MATVEC_WGSL = GGUF_Q5_0_MATVEC_WGSL
  .replace(/q5_0_weight/g, "q5_1_weight")
  .replace(
    "let qh = u32_at(base + 2u);\n  let q_byte = byte_at(base + 6u + (within & 15u));",
    "let qh = u32_at(base + 4u);\n  let q_byte = byte_at(base + 8u + (within & 15u));",
  )
  .replace(
    "return f16_at(base) * f32(i32(qv) - 16);",
    "return f16_at(base) * f32(qv) + f16_at(base + 2u);",
  );

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

const GGUF_Q5_0_EMBEDDING_WGSL = /* wgsl */ `
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

fn u32_at(index: u32) -> u32 {
  return byte_at(index) |
    (byte_at(index + 1u) << 8u) |
    (byte_at(index + 2u) << 16u) |
    (byte_at(index + 3u) << 24u);
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
  let qh = u32_at(base + 2u);
  let q_byte = byte_at(base + 6u + (within & 15u));
  let low4 = select(q_byte & 15u, q_byte >> 4u, within >= 16u);
  let high = (qh >> within) & 1u;
  let qv = low4 | (high << 4u);
  output[k] = f16_at(base) * f32(i32(qv) - 16) * EMBEDDING_SCALE;
}
`;

const GGUF_Q5_1_EMBEDDING_WGSL = GGUF_Q5_0_EMBEDDING_WGSL
  .replace(
    "let qh = u32_at(base + 2u);\n  let q_byte = byte_at(base + 6u + (within & 15u));",
    "let qh = u32_at(base + 4u);\n  let q_byte = byte_at(base + 8u + (within & 15u));",
  )
  .replace(
    "output[k] = f16_at(base) * f32(i32(qv) - 16) * EMBEDDING_SCALE;",
    "output[k] = (f16_at(base) * f32(qv) + f16_at(base + 2u)) * EMBEDDING_SCALE;",
  );

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
  output[k] = f32(i8_at(base + 2u + within)) * f16_at(base) * EMBEDDING_SCALE;
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
  output[k] = (f16_at(base) * f32(sm.x) * f32(qv) - f16_at(base + 2u) * f32(sm.y)) * EMBEDDING_SCALE;
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
  output[k] = (f16_at(base) * f32(sm.x) * f32(low4 + high_bit) - f16_at(base + 2u) * f32(sm.y)) * EMBEDDING_SCALE;
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
  output[k] = f16_at(base + 208u) * f32(scale) * f32(q) * EMBEDDING_SCALE;
}
`;

const ROPE_STORE_WGSL = /* wgsl */ `
struct Params {
  pos: u32,
  max_context: u32,
  heads: u32,
  kv_heads: u32,
  rope_theta: u32,
};

@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;
@group(0) @binding(2) var<storage, read_write> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> key_cache: array<f32>;
@group(0) @binding(4) var<storage, read_write> value_cache: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

fn rope_angle(d: u32) -> f32 {
  let inv = pow(f32(params.rope_theta), -2.0 * f32(d) / f32(HEAD_DIM));
  return f32(params.pos) * inv;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;

  if (idx < params.heads * ROTARY_HALF) {
    let head = idx / ROTARY_HALF;
    let d = idx - head * ROTARY_HALF;
    let base = head * HEAD_DIM;
    let a = q[base + d];
    let b = q[base + ROTARY_HALF + d];
    let angle = rope_angle(d);
    let co = cos(angle);
    let si = sin(angle);
    q[base + d] = a * co - b * si;
    q[base + ROTARY_HALF + d] = a * si + b * co;
  }

  if (idx < params.kv_heads * ROTARY_HALF) {
    let head = idx / ROTARY_HALF;
    let d = idx - head * ROTARY_HALF;
    let base = head * HEAD_DIM;
    let a = k[base + d];
    let b = k[base + ROTARY_HALF + d];
    let angle = rope_angle(d);
    let co = cos(angle);
    let si = sin(angle);
    let cache_base = (head * params.max_context + params.pos) * HEAD_DIM;
    key_cache[cache_base + d] = a * co - b * si;
    key_cache[cache_base + ROTARY_HALF + d] = a * si + b * co;
  }

  if (idx < params.kv_heads * HEAD_DIM) {
    let head = idx / HEAD_DIM;
    let d = idx - head * HEAD_DIM;
    let cache_base = (head * params.max_context + params.pos) * HEAD_DIM;
    value_cache[cache_base + d] = v[idx];
  }
}
`;

const QK_NORM_ROPE_STORE_WGSL = /* wgsl */ `
struct Params {
  pos: u32,
  max_context: u32,
  heads: u32,
  kv_heads: u32,
  rope_theta: u32,
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

var<workgroup> q_partial: array<f32, 256>;
var<workgroup> k_partial: array<f32, 256>;

fn rope_angle(d: u32) -> f32 {
  let inv = pow(f32(params.rope_theta), -2.0 * f32(d) / f32(HEAD_DIM));
  return f32(params.pos) * inv;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let q_head = wg.x;
  let kv_group = params.heads / params.kv_heads;
  let kv_head = q_head / kv_group;
  let d = lid.x;
  let q_base = q_head * HEAD_DIM;
  let k_base = kv_head * HEAD_DIM;
  let owns_kv = (q_head % kv_group) == 0u;

  if (d < HEAD_DIM) {
    let qv = q[q_base + d];
    q_partial[d] = qv * qv;
    if (owns_kv) {
      let kv = k[k_base + d];
      k_partial[d] = kv * kv;
    } else {
      k_partial[d] = 0.0;
    }
  } else {
    q_partial[d] = 0.0;
    k_partial[d] = 0.0;
  }
  workgroupBarrier();

  var stride = 128u;
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

  let q_inv = inverseSqrt(q_partial[0] / f32(HEAD_DIM) + QK_NORM_EPS);
  if (d < ROTARY_HALF) {
    let a = q[q_base + d] * q_inv * q_weight[d];
    let b = q[q_base + ROTARY_HALF + d] * q_inv * q_weight[ROTARY_HALF + d];
    let angle = rope_angle(d);
    let co = cos(angle);
    let si = sin(angle);
    q_out[q_base + d] = a * co - b * si;
    q_out[q_base + ROTARY_HALF + d] = a * si + b * co;
  }

  if (owns_kv) {
    let k_inv = inverseSqrt(k_partial[0] / f32(HEAD_DIM) + QK_NORM_EPS);
    if (d < ROTARY_HALF) {
      let a = k[k_base + d] * k_inv * k_weight[d];
      let b = k[k_base + ROTARY_HALF + d] * k_inv * k_weight[ROTARY_HALF + d];
      let angle = rope_angle(d);
      let co = cos(angle);
      let si = sin(angle);
      let cache_base = (kv_head * params.max_context + params.pos) * HEAD_DIM;
      key_cache[cache_base + d] = a * co - b * si;
      key_cache[cache_base + ROTARY_HALF + d] = a * si + b * co;
    }
    if (d < HEAD_DIM) {
      let cache_base = (kv_head * params.max_context + params.pos) * HEAD_DIM;
      value_cache[cache_base + d] = v[k_base + d];
    }
  }
}
`;

const QKV_NORM_ROPE_STORE_WGSL = /* wgsl */ `
struct Params {
  pos: u32,
  max_context: u32,
  heads: u32,
  kv_heads: u32,
  q_dim: u32,
  kv_dim: u32,
  rope_theta: u32,
};

@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> q_weight: array<f32>;
@group(0) @binding(2) var<storage, read> k_weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> q_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> key_cache: array<f32>;
@group(0) @binding(5) var<storage, read_write> value_cache: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

var<workgroup> q_partial: array<f32, 256>;
var<workgroup> k_partial: array<f32, 256>;

fn rope_angle(d: u32) -> f32 {
  let inv = pow(f32(params.rope_theta), -2.0 * f32(d) / f32(HEAD_DIM));
  return f32(params.pos) * inv;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let q_head = wg.x;
  let kv_group = params.heads / params.kv_heads;
  let kv_head = q_head / kv_group;
  let d = lid.x;
  let q_base = q_head * HEAD_DIM;
  let k_base = params.q_dim + kv_head * HEAD_DIM;
  let v_base = params.q_dim + params.kv_dim + kv_head * HEAD_DIM;
  let owns_kv = (q_head % kv_group) == 0u;

  if (d < HEAD_DIM) {
    let qv = qkv[q_base + d];
    q_partial[d] = qv * qv;
    if (owns_kv) {
      let kv = qkv[k_base + d];
      k_partial[d] = kv * kv;
    } else {
      k_partial[d] = 0.0;
    }
  } else {
    q_partial[d] = 0.0;
    k_partial[d] = 0.0;
  }
  workgroupBarrier();

  var stride = 128u;
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

  let q_inv = inverseSqrt(q_partial[0] / f32(HEAD_DIM) + QK_NORM_EPS);
  if (d < ROTARY_HALF) {
    let a = qkv[q_base + d] * q_inv * q_weight[d];
    let b = qkv[q_base + ROTARY_HALF + d] * q_inv * q_weight[ROTARY_HALF + d];
    let angle = rope_angle(d);
    let co = cos(angle);
    let si = sin(angle);
    q_out[q_base + d] = a * co - b * si;
    q_out[q_base + ROTARY_HALF + d] = a * si + b * co;
  }

  if (owns_kv) {
    let k_inv = inverseSqrt(k_partial[0] / f32(HEAD_DIM) + QK_NORM_EPS);
    if (d < ROTARY_HALF) {
      let a = qkv[k_base + d] * k_inv * k_weight[d];
      let b = qkv[k_base + ROTARY_HALF + d] * k_inv * k_weight[ROTARY_HALF + d];
      let angle = rope_angle(d);
      let co = cos(angle);
      let si = sin(angle);
      let cache_base = (kv_head * params.max_context + params.pos) * HEAD_DIM;
      key_cache[cache_base + d] = a * co - b * si;
      key_cache[cache_base + ROTARY_HALF + d] = a * si + b * co;
    }
    if (d < HEAD_DIM) {
      let cache_base = (kv_head * params.max_context + params.pos) * HEAD_DIM;
      value_cache[cache_base + d] = qkv[v_base + d];
    }
  }
}
`;

const ROPE_STORE_INTERLEAVED_WGSL = /* wgsl */ `
struct Params {
  pos: u32,
  max_context: u32,
  heads: u32,
  kv_heads: u32,
  rope_theta: u32,
};

@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;
@group(0) @binding(2) var<storage, read_write> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> key_cache: array<f32>;
@group(0) @binding(4) var<storage, read_write> value_cache: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

fn rope_angle(d: u32) -> f32 {
  let inv = pow(f32(params.rope_theta), -2.0 * f32(d) / f32(HEAD_DIM));
  return f32(params.pos) * inv;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;

  if (idx < params.heads * ROTARY_HALF) {
    let head = idx / ROTARY_HALF;
    let d = idx - head * ROTARY_HALF;
    let base = head * HEAD_DIM;
    let pair_base = base + 2u * d;
    let a = q[pair_base];
    let b = q[pair_base + 1u];
    let angle = rope_angle(d);
    let co = cos(angle);
    let si = sin(angle);
    q[pair_base]      = a * co - b * si;
    q[pair_base + 1u] = a * si + b * co;
  }

  if (idx < params.kv_heads * ROTARY_HALF) {
    let head = idx / ROTARY_HALF;
    let d = idx - head * ROTARY_HALF;
    let base = head * HEAD_DIM;
    let pair_base = base + 2u * d;
    let a = k[pair_base];
    let b = k[pair_base + 1u];
    let angle = rope_angle(d);
    let co = cos(angle);
    let si = sin(angle);
    let cache_base = (head * params.max_context + params.pos) * HEAD_DIM;
    let cache_pair = cache_base + 2u * d;
    key_cache[cache_pair]      = a * co - b * si;
    key_cache[cache_pair + 1u] = a * si + b * co;
  }

  if (idx < params.kv_heads * HEAD_DIM) {
    let head = idx / HEAD_DIM;
    let d = idx - head * HEAD_DIM;
    let cache_base = (head * params.max_context + params.pos) * HEAD_DIM;
    value_cache[cache_base + d] = v[idx];
  }
}
`;

const QK_NORM_ROPE_STORE_INTERLEAVED_WGSL = /* wgsl */ `
struct Params {
  pos: u32,
  max_context: u32,
  heads: u32,
  kv_heads: u32,
  rope_theta: u32,
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

var<workgroup> q_partial: array<f32, 256>;
var<workgroup> k_partial: array<f32, 256>;

fn rope_angle(d: u32) -> f32 {
  let inv = pow(f32(params.rope_theta), -2.0 * f32(d) / f32(HEAD_DIM));
  return f32(params.pos) * inv;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let q_head = wg.x;
  let kv_group = params.heads / params.kv_heads;
  let kv_head = q_head / kv_group;
  let d = lid.x;
  let q_base = q_head * HEAD_DIM;
  let k_base = kv_head * HEAD_DIM;
  let owns_kv = (q_head % kv_group) == 0u;

  if (d < HEAD_DIM) {
    let qv = q[q_base + d];
    q_partial[d] = qv * qv;
    if (owns_kv) {
      let kv = k[k_base + d];
      k_partial[d] = kv * kv;
    } else {
      k_partial[d] = 0.0;
    }
  } else {
    q_partial[d] = 0.0;
    k_partial[d] = 0.0;
  }
  workgroupBarrier();

  var stride = 128u;
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

  let q_inv = inverseSqrt(q_partial[0] / f32(HEAD_DIM) + QK_NORM_EPS);
  if (d < ROTARY_HALF) {
    let pair_base = q_base + 2u * d;
    let a = q[pair_base]      * q_inv * q_weight[2u * d];
    let b = q[pair_base + 1u] * q_inv * q_weight[2u * d + 1u];
    let angle = rope_angle(d);
    let co = cos(angle);
    let si = sin(angle);
    q_out[pair_base]      = a * co - b * si;
    q_out[pair_base + 1u] = a * si + b * co;
  }

  if (owns_kv) {
    let k_inv = inverseSqrt(k_partial[0] / f32(HEAD_DIM) + QK_NORM_EPS);
    if (d < ROTARY_HALF) {
      let pair_base = k_base + 2u * d;
      let a = k[pair_base]      * k_inv * k_weight[2u * d];
      let b = k[pair_base + 1u] * k_inv * k_weight[2u * d + 1u];
      let angle = rope_angle(d);
      let co = cos(angle);
      let si = sin(angle);
      let cache_base = (kv_head * params.max_context + params.pos) * HEAD_DIM;
      let cache_pair = cache_base + 2u * d;
      key_cache[cache_pair]      = a * co - b * si;
      key_cache[cache_pair + 1u] = a * si + b * co;
    }
    if (d < HEAD_DIM) {
      let cache_base = (kv_head * params.max_context + params.pos) * HEAD_DIM;
      value_cache[cache_base + d] = v[k_base + d];
    }
  }
}
`;

const QKV_NORM_ROPE_STORE_INTERLEAVED_WGSL = /* wgsl */ `
struct Params {
  pos: u32,
  max_context: u32,
  heads: u32,
  kv_heads: u32,
  q_dim: u32,
  kv_dim: u32,
  rope_theta: u32,
};

@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> q_weight: array<f32>;
@group(0) @binding(2) var<storage, read> k_weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> q_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> key_cache: array<f32>;
@group(0) @binding(5) var<storage, read_write> value_cache: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

var<workgroup> q_partial: array<f32, 256>;
var<workgroup> k_partial: array<f32, 256>;

fn rope_angle(d: u32) -> f32 {
  let inv = pow(f32(params.rope_theta), -2.0 * f32(d) / f32(HEAD_DIM));
  return f32(params.pos) * inv;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let q_head = wg.x;
  let kv_group = params.heads / params.kv_heads;
  let kv_head = q_head / kv_group;
  let d = lid.x;
  let q_base = q_head * HEAD_DIM;
  let k_base = params.q_dim + kv_head * HEAD_DIM;
  let v_base = params.q_dim + params.kv_dim + kv_head * HEAD_DIM;
  let owns_kv = (q_head % kv_group) == 0u;

  if (d < HEAD_DIM) {
    let qv = qkv[q_base + d];
    q_partial[d] = qv * qv;
    if (owns_kv) {
      let kv = qkv[k_base + d];
      k_partial[d] = kv * kv;
    } else {
      k_partial[d] = 0.0;
    }
  } else {
    q_partial[d] = 0.0;
    k_partial[d] = 0.0;
  }
  workgroupBarrier();

  var stride = 128u;
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

  let q_inv = inverseSqrt(q_partial[0] / f32(HEAD_DIM) + QK_NORM_EPS);
  if (d < ROTARY_HALF) {
    let pair_base = q_base + 2u * d;
    let a = qkv[pair_base]      * q_inv * q_weight[2u * d];
    let b = qkv[pair_base + 1u] * q_inv * q_weight[2u * d + 1u];
    let angle = rope_angle(d);
    let co = cos(angle);
    let si = sin(angle);
    q_out[pair_base]      = a * co - b * si;
    q_out[pair_base + 1u] = a * si + b * co;
  }

  if (owns_kv) {
    let k_inv = inverseSqrt(k_partial[0] / f32(HEAD_DIM) + QK_NORM_EPS);
    if (d < ROTARY_HALF) {
      let pair_base = k_base + 2u * d;
      let a = qkv[pair_base]      * k_inv * k_weight[2u * d];
      let b = qkv[pair_base + 1u] * k_inv * k_weight[2u * d + 1u];
      let angle = rope_angle(d);
      let co = cos(angle);
      let si = sin(angle);
      let cache_base = (kv_head * params.max_context + params.pos) * HEAD_DIM;
      let cache_pair = cache_base + 2u * d;
      key_cache[cache_pair]      = a * co - b * si;
      key_cache[cache_pair + 1u] = a * si + b * co;
    }
    if (d < HEAD_DIM) {
      let cache_base = (kv_head * params.max_context + params.pos) * HEAD_DIM;
      value_cache[cache_base + d] = qkv[v_base + d];
    }
  }
}
`;

const ATTENTION_SCORE_WGSL = /* wgsl */ `
struct Params {
  pos: u32,
  max_context: u32,
  heads: u32,
  kv_heads: u32,
  window: u32,
};

@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> key_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let head = wg.x;
  let t = wg.y;
  let start = select(0u, params.pos + 1u - params.window, params.window != 0u && params.pos + 1u > params.window);
  if (t < start) {
    return;
  }
  let kv_group = params.heads / params.kv_heads;
  let kv_head = head / kv_group;
  let q_base = head * HEAD_DIM;
  let k_base = (kv_head * params.max_context + t) * HEAD_DIM;
  let d = lid.x;
  if (d < HEAD_DIM) {
    partial[d] = q[q_base + d] * key_cache[k_base + d];
  } else {
    partial[d] = 0.0;
  }
  workgroupBarrier();

  var stride = 128u;
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
    scores[head * params.max_context + t] = partial[0] * ATTN_SCALE;
  }
}
`;

const ATTENTION_VALUE_WGSL = /* wgsl */ `
struct Params {
  pos: u32,
  max_context: u32,
  heads: u32,
  kv_heads: u32,
  window: u32,
};

@group(0) @binding(0) var<storage, read_write> scores: array<f32>;
@group(0) @binding(1) var<storage, read_write> value_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> max_score: f32;
var<workgroup> denom: f32;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let head = wg.x;
  let d = lid.x;
  let kv_group = params.heads / params.kv_heads;
  let kv_head = head / kv_group;

  if (d == 0u) {
    let start = select(0u, params.pos + 1u - params.window, params.window != 0u && params.pos + 1u > params.window);
    var m = -3.402823e38;
    for (var t = start; t <= params.pos; t = t + 1u) {
      m = max(m, scores[head * params.max_context + t]);
    }
    max_score = m;

    var s = 0.0;
    for (var t = start; t <= params.pos; t = t + 1u) {
      s = s + exp(scores[head * params.max_context + t] - m);
    }
    denom = s;
  }
  workgroupBarrier();

  var acc = 0.0;
  if (d < HEAD_DIM) {
    let start = select(0u, params.pos + 1u - params.window, params.window != 0u && params.pos + 1u > params.window);
    for (var t = start; t <= params.pos; t = t + 1u) {
      let p = exp(scores[head * params.max_context + t] - max_score) / denom;
      let v_base = (kv_head * params.max_context + t) * HEAD_DIM;
      acc = acc + p * value_cache[v_base + d];
    }
    output[head * HEAD_DIM + d] = acc;
  }
}
`;

const ADD_WGSL = /* wgsl */ `
struct Params {
  n: u32,
};

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> io: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < params.n) {
    io[i] = a[i] + io[i];
  }
}
`;

const ADD_CLAMP_WGSL = /* wgsl */ `
struct Params {
  n: u32,
};

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> io: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < params.n) {
    let v = a[i] + io[i];
    io[i] = min(max(v, -65000.0), 65000.0);
  }
}
`;

const SILU_MUL_WGSL = /* wgsl */ `
struct Params {
  n: u32,
};

@group(0) @binding(0) var<storage, read_write> gate: array<f32>;
@group(0) @binding(1) var<storage, read_write> up: array<f32>;
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

const SILU_SPLIT_MUL_WGSL = /* wgsl */ `
struct Params {
  n: u32,
};

@group(0) @binding(0) var<storage, read_write> gate_up: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < params.n) {
    let g = gate_up[i];
    output[i] = (g / (1.0 + exp(-g))) * gate_up[params.n + i];
  }
}
`;

const GELU_TANH_MUL_WGSL = /* wgsl */ `
struct Params {
  n: u32,
};

@group(0) @binding(0) var<storage, read_write> gate: array<f32>;
@group(0) @binding(1) var<storage, read_write> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn gelu_tanh(x: f32) -> f32 {
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
    output[i] = gelu_tanh(gate[i]) * up[i];
  }
}
`;

const LM_HEAD_WGSL = /* wgsl */ `
struct Params {
  rows: u32,
  vocab_offset: u32,
};

@group(0) @binding(0) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> embedding: array<f32>;
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partial: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.x;
  if (row >= params.rows) {
    return;
  }
  var sum = 0.0;
  for (var h = lid.x; h < HIDDEN; h = h + 128u) {
    sum = sum + hidden[h] * embedding[row * HIDDEN + h];
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
    logits[params.vocab_offset + row] = partial[0];
  }
}
`;

const ARGMAX_STAGE1_WGSL = /* wgsl */ `
struct Params {
  vocab: u32,
  penalty_count: u32,
  repetition_penalty_milli: u32,
};

struct Candidate {
  value: f32,
  index: u32,
};

@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<Candidate>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> penalty_ids: array<u32>;

var<workgroup> values: array<f32, 256>;
var<workgroup> indices: array<u32, 256>;

fn better(v: f32, i: u32, best_v: f32, best_i: u32) -> bool {
  if (v != v) {
    return false;
  }
  if (best_v != best_v) {
    return true;
  }
  return (v > best_v) || ((v == best_v) && (i < best_i));
}

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
  if (i < params.vocab) {
    values[lid.x] = repetition_penalized(i, logits[i]);
    indices[lid.x] = i;
  } else {
    values[lid.x] = -3.402823e38;
    indices[lid.x] = 4294967295u;
  }
  workgroupBarrier();

  var stride = 128u;
  loop {
    if (lid.x < stride) {
      let ov = values[lid.x + stride];
      let oi = indices[lid.x + stride];
      if (better(ov, oi, values[lid.x], indices[lid.x])) {
        values[lid.x] = ov;
        indices[lid.x] = oi;
      }
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride = stride >> 1u;
  }

  if (lid.x == 0u) {
    partials[wg.x].value = values[0];
    partials[wg.x].index = indices[0];
  }
}
`;

const ARGMAX_STAGE2_WGSL = /* wgsl */ `
struct Params {
  groups: u32,
};

struct Candidate {
  value: f32,
  index: u32,
};

@group(0) @binding(0) var<storage, read_write> partials: array<Candidate>;
@group(0) @binding(1) var<storage, read_write> result: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> values: array<f32, 256>;
var<workgroup> indices: array<u32, 256>;

fn better(v: f32, i: u32, best_v: f32, best_i: u32) -> bool {
  if (v != v) {
    return false;
  }
  if (best_v != best_v) {
    return true;
  }
  return (v > best_v) || ((v == best_v) && (i < best_i));
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  var best_v = -3.402823e38;
  var best_i = 4294967295u;
  for (var i = lid.x; i < params.groups; i = i + 256u) {
    let candidate = partials[i];
    if (better(candidate.value, candidate.index, best_v, best_i)) {
      best_v = candidate.value;
      best_i = candidate.index;
    }
  }
  values[lid.x] = best_v;
  indices[lid.x] = best_i;
  workgroupBarrier();

  var stride = 128u;
  loop {
    if (lid.x < stride) {
      let ov = values[lid.x + stride];
      let oi = indices[lid.x + stride];
      if (better(ov, oi, values[lid.x], indices[lid.x])) {
        values[lid.x] = ov;
        indices[lid.x] = oi;
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
