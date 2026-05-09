// Runtime registry. The site now ships GGUF artifacts only; every entry here
// points at a single GGUF blob plus tokenizer metadata when the runtime needs
// an external tokenizer.

export type ModelKind = "llama" | "lfm2" | "granite" | "gpt2";

export type ModelRuntime =
  | "gemma-webgpu"
  | "lfm2-webgpu"
  | "llama-webgpu"
  | "granite-webgpu"
  | "gpt-webgpu";

export interface ModelDef {
  id: string;
  label: string;
  kind: ModelKind;
  runtime: ModelRuntime;
  pitch: string;
  chat: boolean;
  eosIds: number[];
  vocab: number;
  layers: number;
  kvHeads: number;
  headDim: number;
  kvDtype: "float32";
  maxContext: number;
  layerKvHeads?: number[];
  layerTypes?: ("conv" | "attention" | "mamba")[];
  convDim?: number;
  convL?: number;
  ssmShape?: number[];
  noPositionIds?: boolean;
  thinking?: boolean;
  preferredEp?: "webgpu";
  base: string;
  gguf: string;
  tokenizer?: string;
  tokenizerConfig?: string;
  rawAddSpecialTokens?: boolean;
  devOnly?: boolean;
  defaultRepetitionPenalty?: number;
  defaultTemperature?: number;
  defaultTopP?: number;
  defaultTopK?: number;
}

const graniteLayerTypes: ("mamba" | "attention")[] = [
  "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba",
  "mamba", "mamba", "attention", "mamba", "mamba", "attention", "mamba", "mamba",
  "mamba", "attention", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba",
  "mamba", "mamba", "mamba", "attention", "mamba", "mamba", "mamba", "mamba",
];

const openelm270mLayerKvHeads = [
  3, 3, 3, 3, 3,
  4, 4, 4, 4, 4, 4, 4,
  5, 5, 5, 5,
];

const lfm2_5_350mLayerTypes: ("conv" | "attention")[] = [
  "conv", "conv", "attention", "conv", "conv", "attention",
  "conv", "conv", "attention", "conv", "attention", "conv",
  "attention", "conv", "attention", "conv",
];

export const MODELS: ModelDef[] = [
  {
    id: "lfm2_5-350m-q4-k-m-gguf-wgsl",
    label: "LFM2.5-350M (Q4_K_M GGUF)",
    kind: "lfm2",
    runtime: "lfm2-webgpu",
    pitch: "219 MB · chat · structured extraction · Q4_K_M GGUF",
    chat: true,
    eosIds: [7, 2],
    vocab: 65536,
    layers: 16,
    kvHeads: 8,
    headDim: 64,
    kvDtype: "float32",
    maxContext: 4096,
    noPositionIds: true,
    layerTypes: lfm2_5_350mLayerTypes,
    convDim: 1024,
    convL: 3,
    preferredEp: "webgpu",
    base: "/models/lfm2_5-350m",
    gguf: "gguf/q4_k_m",
    tokenizer: "raw/tokenizer.json",
    tokenizerConfig: "raw/tokenizer_config.json",
  },
  {
    id: "lfm2_5-350m-q8-gguf-wgsl",
    label: "LFM2.5-350M (Q8_0 GGUF)",
    kind: "lfm2",
    runtime: "lfm2-webgpu",
    pitch: "362 MB · chat · structured extraction · Q8_0 GGUF",
    chat: true,
    eosIds: [7, 2],
    vocab: 65536,
    layers: 16,
    kvHeads: 8,
    headDim: 64,
    kvDtype: "float32",
    maxContext: 4096,
    noPositionIds: true,
    layerTypes: lfm2_5_350mLayerTypes,
    convDim: 1024,
    convL: 3,
    preferredEp: "webgpu",
    base: "/models/lfm2_5-350m",
    gguf: "gguf/q8_0",
    tokenizer: "raw/tokenizer.json",
    tokenizerConfig: "raw/tokenizer_config.json",
  },
  {
    id: "gemma-3-270m-it-q4-k-m-gguf-wgsl",
    label: "Gemma 3 270M-it (Q4_K_M GGUF)",
    kind: "llama",
    runtime: "llama-webgpu",
    pitch: "241 MB · chat · Gemma 3 · Q4_K_M GGUF",
    chat: true,
    eosIds: [1, 106],
    vocab: 262144,
    layers: 18,
    kvHeads: 1,
    headDim: 256,
    kvDtype: "float32",
    maxContext: 512,
    noPositionIds: true,
    preferredEp: "webgpu",
    base: "/models/gemma-3-270m-it",
    gguf: "gguf/q4_k_m",
    tokenizer: "raw/tokenizer.json",
    tokenizerConfig: "raw/tokenizer_config.json",
  },
  {
    id: "gemma-3-270m-it-q8-gguf-wgsl",
    label: "Gemma 3 270M-it (Q8_0 GGUF)",
    kind: "llama",
    runtime: "llama-webgpu",
    pitch: "278 MB · chat · Gemma 3 · Q8_0 GGUF",
    chat: true,
    eosIds: [1, 106],
    vocab: 262144,
    layers: 18,
    kvHeads: 1,
    headDim: 256,
    kvDtype: "float32",
    maxContext: 512,
    noPositionIds: true,
    preferredEp: "webgpu",
    base: "/models/gemma-3-270m-it",
    gguf: "gguf/q8_0",
    tokenizer: "raw/tokenizer.json",
    tokenizerConfig: "raw/tokenizer_config.json",
    devOnly: true,
  },
  {
    id: "gemma-3-270m-it-q8-imported",
    label: "Gemma 3 270M-it (Q8_0 imported)",
    kind: "llama",
    runtime: "gemma-webgpu",
    pitch: "278 MB · chat · Gemma 3 · imported GGUF runtime",
    chat: true,
    eosIds: [1, 106],
    vocab: 262144,
    layers: 18,
    kvHeads: 1,
    headDim: 256,
    kvDtype: "float32",
    maxContext: 512,
    preferredEp: "webgpu",
    base: "/models/gemma-3-270m-it",
    gguf: "gguf/q8_0",
    devOnly: true,
  },
  {
    id: "qwen3-0.6b-q4-k-m-gguf-wgsl",
    label: "Qwen3-0.6B (Q4_K_M GGUF)",
    kind: "llama",
    runtime: "llama-webgpu",
    pitch: "462 MB · chat with thinking · Q4_K_M GGUF",
    chat: true,
    eosIds: [151645, 151643],
    vocab: 151936,
    layers: 28,
    kvHeads: 8,
    headDim: 128,
    kvDtype: "float32",
    maxContext: 2048,
    thinking: true,
    preferredEp: "webgpu",
    base: "/models/qwen3-0.6b",
    gguf: "gguf/q4_k_m",
    tokenizer: "raw/tokenizer.json",
    tokenizerConfig: "raw/tokenizer_config.json",
    // Greedy collapses to <|im_end|> without closing </think>; upstream-recommended values.
    defaultTemperature: 0.6,
    defaultTopP: 0.95,
    defaultTopK: 20,
  },
  {
    id: "qwen3-0.6b-q8-gguf-wgsl",
    label: "Qwen3-0.6B (Q8_0 GGUF)",
    kind: "llama",
    runtime: "llama-webgpu",
    pitch: "767 MB · chat with thinking · Q8_0 GGUF",
    chat: true,
    eosIds: [151645, 151643],
    vocab: 151936,
    layers: 28,
    kvHeads: 8,
    headDim: 128,
    kvDtype: "float32",
    maxContext: 2048,
    thinking: true,
    preferredEp: "webgpu",
    base: "/models/qwen3-0.6b",
    gguf: "gguf/q8_0",
    tokenizer: "raw/tokenizer.json",
    tokenizerConfig: "raw/tokenizer_config.json",
    devOnly: true,
    defaultTemperature: 0.6,
    defaultTopP: 0.95,
    defaultTopK: 20,
  },
  {
    id: "smollm2-360m-instruct-q4-k-m-gguf-wgsl",
    label: "SmolLM2-360M-Instruct (Q4_K_M GGUF)",
    kind: "llama",
    runtime: "llama-webgpu",
    pitch: "258 MB · chat · Q4_K_M GGUF",
    chat: true,
    eosIds: [2],
    defaultRepetitionPenalty: 1.15,
    vocab: 49152,
    layers: 32,
    kvHeads: 5,
    headDim: 64,
    kvDtype: "float32",
    maxContext: 512,
    preferredEp: "webgpu",
    base: "/models/smollm2-360m-instruct",
    gguf: "gguf/q4_k_m",
    tokenizer: "raw/tokenizer.json",
    tokenizerConfig: "raw/tokenizer_config.json",
  },
  {
    id: "smollm2-360m-instruct-q8-gguf-wgsl",
    label: "SmolLM2-360M-Instruct (Q8_0 GGUF)",
    kind: "llama",
    runtime: "llama-webgpu",
    pitch: "369 MB · chat · Q8_0 GGUF",
    chat: true,
    eosIds: [2],
    defaultRepetitionPenalty: 1.15,
    vocab: 49152,
    layers: 32,
    kvHeads: 5,
    headDim: 64,
    kvDtype: "float32",
    maxContext: 512,
    preferredEp: "webgpu",
    base: "/models/smollm2-360m-instruct",
    gguf: "gguf/q8_0",
    tokenizer: "raw/tokenizer.json",
    tokenizerConfig: "raw/tokenizer_config.json",
  },
  {
    id: "granite-4.0-h-350m-q4-k-m-gguf-wgsl",
    label: "Granite 4.0 H 350M (Q4_K_M GGUF)",
    kind: "granite",
    runtime: "granite-webgpu",
    pitch: "213 MB · chat · IBM hybrid Mamba/attention · Q4_K_M GGUF",
    chat: true,
    eosIds: [100257, 100265],
    vocab: 100352,
    layers: 32,
    kvHeads: 6,
    headDim: 64,
    kvDtype: "float32",
    maxContext: 4096,
    noPositionIds: true,
    layerTypes: graniteLayerTypes,
    convDim: 3072,
    convL: 4,
    ssmShape: [48, 32, 128],
    preferredEp: "webgpu",
    base: "/models/granite-4.0-h-350m",
    gguf: "gguf/q4_k_m",
    tokenizer: "raw/tokenizer.json",
    tokenizerConfig: "raw/tokenizer_config.json",
  },
  {
    id: "granite-4.0-h-350m-q8-gguf-wgsl",
    label: "Granite 4.0 H 350M (Q8_0 GGUF)",
    kind: "granite",
    runtime: "granite-webgpu",
    pitch: "349 MB · chat · IBM hybrid Mamba/attention · Q8_0 GGUF",
    chat: true,
    eosIds: [100257, 100265],
    vocab: 100352,
    layers: 32,
    kvHeads: 6,
    headDim: 64,
    kvDtype: "float32",
    maxContext: 4096,
    noPositionIds: true,
    layerTypes: graniteLayerTypes,
    convDim: 3072,
    convL: 4,
    ssmShape: [48, 32, 128],
    preferredEp: "webgpu",
    base: "/models/granite-4.0-h-350m",
    gguf: "gguf/q8_0",
    tokenizer: "raw/tokenizer.json",
    tokenizerConfig: "raw/tokenizer_config.json",
  },
  {
    id: "distilgpt2-q4-k-m-gguf-wgsl",
    label: "distilgpt2 (Q4_K_M GGUF)",
    kind: "gpt2",
    runtime: "gpt-webgpu",
    pitch: "81 MB · completion-only historical baseline · Q4_K_M GGUF",
    chat: false,
    eosIds: [50256],
    vocab: 50257,
    layers: 6,
    kvHeads: 12,
    headDim: 64,
    kvDtype: "float32",
    maxContext: 1024,
    preferredEp: "webgpu",
    base: "/models/distilgpt2",
    gguf: "gguf/q4_k_m",
    tokenizer: "raw/tokenizer.json",
    tokenizerConfig: "raw/tokenizer_config.json",
  },
  {
    id: "distilgpt2-q8-gguf-wgsl",
    label: "distilgpt2 (Q8_0 GGUF)",
    kind: "gpt2",
    runtime: "gpt-webgpu",
    pitch: "126 MB · completion-only historical baseline · Q8_0 GGUF",
    chat: false,
    eosIds: [50256],
    vocab: 50257,
    layers: 6,
    kvHeads: 12,
    headDim: 64,
    kvDtype: "float32",
    maxContext: 1024,
    preferredEp: "webgpu",
    base: "/models/distilgpt2",
    gguf: "gguf/q8_0",
    tokenizer: "raw/tokenizer.json",
    tokenizerConfig: "raw/tokenizer_config.json",
    devOnly: true,
  },
  {
    id: "gpt2-q4-k-m-gguf-wgsl",
    label: "gpt2 (Q4_K_M GGUF)",
    kind: "gpt2",
    runtime: "gpt-webgpu",
    pitch: "108 MB · completion-only historical baseline · Q4_K_M GGUF",
    chat: false,
    eosIds: [50256],
    vocab: 50257,
    layers: 12,
    kvHeads: 12,
    headDim: 64,
    kvDtype: "float32",
    maxContext: 1024,
    preferredEp: "webgpu",
    base: "/models/gpt2",
    gguf: "gguf/q4_k_m",
    tokenizer: "raw/tokenizer.json",
    tokenizerConfig: "raw/tokenizer_config.json",
  },
  {
    id: "gpt2-q8-gguf-wgsl",
    label: "gpt2 (Q8_0 GGUF)",
    kind: "gpt2",
    runtime: "gpt-webgpu",
    pitch: "168 MB · completion-only historical baseline · Q8_0 GGUF",
    chat: false,
    eosIds: [50256],
    vocab: 50257,
    layers: 12,
    kvHeads: 12,
    headDim: 64,
    kvDtype: "float32",
    maxContext: 1024,
    preferredEp: "webgpu",
    base: "/models/gpt2",
    gguf: "gguf/q8_0",
    tokenizer: "raw/tokenizer.json",
    tokenizerConfig: "raw/tokenizer_config.json",
    devOnly: true,
  },
  {
    id: "gpt2-medium-q4-k-m-gguf-wgsl",
    label: "gpt2-medium (Q4_K_M GGUF)",
    kind: "gpt2",
    runtime: "gpt-webgpu",
    pitch: "258 MB · 355M completion baseline · Q4_K_M GGUF",
    chat: false,
    eosIds: [50256],
    vocab: 50257,
    layers: 24,
    kvHeads: 16,
    headDim: 64,
    kvDtype: "float32",
    maxContext: 1024,
    preferredEp: "webgpu",
    base: "/models/gpt2-medium",
    gguf: "gguf/q4_k_m",
    tokenizer: "raw/tokenizer.json",
    tokenizerConfig: "raw/tokenizer_config.json",
    devOnly: true,
  },
  {
    id: "gpt2-medium-q8-gguf-wgsl",
    label: "gpt2-medium (Q8_0 GGUF)",
    kind: "gpt2",
    runtime: "gpt-webgpu",
    pitch: "417 MB · 355M completion baseline · Q8_0 GGUF",
    chat: false,
    eosIds: [50256],
    vocab: 50257,
    layers: 24,
    kvHeads: 16,
    headDim: 64,
    kvDtype: "float32",
    maxContext: 1024,
    preferredEp: "webgpu",
    base: "/models/gpt2-medium",
    gguf: "gguf/q8_0",
    tokenizer: "raw/tokenizer.json",
    tokenizerConfig: "raw/tokenizer_config.json",
    devOnly: true,
  },
  {
    id: "smollm2-135m-instruct-q4-k-m-gguf-wgsl",
    label: "SmolLM2-135M-Instruct (Q4_K_M GGUF)",
    kind: "llama",
    runtime: "llama-webgpu",
    pitch: "101 MB · chat · Q4_K_M GGUF",
    chat: true,
    eosIds: [2],
    vocab: 49152,
    layers: 30,
    kvHeads: 3,
    headDim: 64,
    kvDtype: "float32",
    maxContext: 512,
    preferredEp: "webgpu",
    base: "/models/smollm2-135m-instruct",
    gguf: "gguf/q4_k_m",
    tokenizer: "raw/tokenizer.json",
    tokenizerConfig: "raw/tokenizer_config.json",
  },
  {
    id: "smollm2-135m-instruct-q8-gguf-wgsl",
    label: "SmolLM2-135M-Instruct (Q8_0 GGUF)",
    kind: "llama",
    runtime: "llama-webgpu",
    pitch: "138 MB · chat · Q8_0 GGUF",
    chat: true,
    eosIds: [2],
    vocab: 49152,
    layers: 30,
    kvHeads: 3,
    headDim: 64,
    kvDtype: "float32",
    maxContext: 512,
    preferredEp: "webgpu",
    base: "/models/smollm2-135m-instruct",
    gguf: "gguf/q8_0",
    tokenizer: "raw/tokenizer.json",
    tokenizerConfig: "raw/tokenizer_config.json",
    devOnly: true,
  },
  {
    id: "monad-q4-k-m-gguf-wgsl",
    label: "Monad (Q4_K_M GGUF)",
    kind: "llama",
    runtime: "llama-webgpu",
    pitch: "33 MB · single-turn chat with thinking · Q4_K_M GGUF",
    chat: true,
    eosIds: [2],
    vocab: 8192,
    layers: 64,
    kvHeads: 2,
    headDim: 64,
    kvDtype: "float32",
    maxContext: 512,
    noPositionIds: true,
    thinking: true,
    preferredEp: "webgpu",
    base: "/models/monad",
    gguf: "gguf/q4_k_m",
    tokenizer: "raw/tokenizer.json",
    tokenizerConfig: "raw/tokenizer_config.json",
    // Greedy truncates inside <think> like Qwen3; PleIAs publishes no preset.
    defaultTemperature: 0.7,
    defaultTopP: 0.9,
    defaultTopK: 40,
  },
  {
    id: "monad-q8-gguf-wgsl",
    label: "Monad (Q8_0 GGUF)",
    kind: "llama",
    runtime: "llama-webgpu",
    pitch: "58 MB · single-turn chat with thinking · Q8_0 GGUF",
    chat: true,
    eosIds: [2],
    vocab: 8192,
    layers: 64,
    kvHeads: 2,
    headDim: 64,
    kvDtype: "float32",
    maxContext: 512,
    noPositionIds: true,
    thinking: true,
    preferredEp: "webgpu",
    base: "/models/monad",
    gguf: "gguf/q8_0",
    tokenizer: "raw/tokenizer.json",
    tokenizerConfig: "raw/tokenizer_config.json",
    devOnly: true,
    defaultTemperature: 0.7,
    defaultTopP: 0.9,
    defaultTopK: 40,
  },
  {
    id: "baguettotron-q4-k-m-gguf-wgsl",
    label: "Baguettotron (Q4_K_M GGUF)",
    kind: "llama",
    runtime: "llama-webgpu",
    pitch: "229 MB · French chat with thinking · Q4_K_M GGUF",
    chat: true,
    eosIds: [2],
    vocab: 65536,
    layers: 80,
    kvHeads: 4,
    headDim: 64,
    kvDtype: "float32",
    maxContext: 512,
    noPositionIds: true,
    thinking: true,
    preferredEp: "webgpu",
    base: "/models/baguettotron",
    gguf: "gguf/q4_k_m",
    tokenizer: "raw/tokenizer.json",
    tokenizerConfig: "raw/tokenizer_config.json",
    defaultTemperature: 0.7,
    defaultTopP: 0.9,
    defaultTopK: 40,
  },
  {
    id: "baguettotron-q8-gguf-wgsl",
    label: "Baguettotron (Q8_0 GGUF)",
    kind: "llama",
    runtime: "llama-webgpu",
    pitch: "328 MB · French chat with thinking · Q8_0 GGUF",
    chat: true,
    eosIds: [2],
    vocab: 65536,
    layers: 80,
    kvHeads: 4,
    headDim: 64,
    kvDtype: "float32",
    maxContext: 512,
    noPositionIds: true,
    thinking: true,
    preferredEp: "webgpu",
    base: "/models/baguettotron",
    gguf: "gguf/q8_0",
    tokenizer: "raw/tokenizer.json",
    tokenizerConfig: "raw/tokenizer_config.json",
    devOnly: true,
    defaultTemperature: 0.7,
    defaultTopP: 0.9,
    defaultTopK: 40,
  },
  {
    id: "openelm-270m-instruct-q8-gguf-wgsl",
    label: "OpenELM-270M-Instruct (Q8_0 GGUF)",
    kind: "llama",
    runtime: "llama-webgpu",
    pitch: "276 MB · raw instruct prompt · Q8_0 GGUF · variable GQA",
    chat: false,
    eosIds: [2],
    vocab: 32000,
    layers: 16,
    kvHeads: 5,
    headDim: 64,
    kvDtype: "float32",
    maxContext: 512,
    noPositionIds: true,
    layerKvHeads: openelm270mLayerKvHeads,
    preferredEp: "webgpu",
    base: "/models/openelm-270m-instruct",
    gguf: "gguf/q8_0",
    tokenizer: "raw/tokenizer.json",
    tokenizerConfig: "raw/tokenizer_config.json",
    rawAddSpecialTokens: true,
    defaultRepetitionPenalty: 1.2,
  },
  {
    id: "openelm-270m-instruct-q4-k-m-gguf-wgsl",
    label: "OpenELM-270M-Instruct (Q4_K_M GGUF)",
    kind: "llama",
    runtime: "llama-webgpu",
    pitch: "167 MB · raw instruct prompt · Q4_K_M GGUF · variable GQA",
    chat: false,
    eosIds: [2],
    vocab: 32000,
    layers: 16,
    kvHeads: 5,
    headDim: 64,
    kvDtype: "float32",
    maxContext: 512,
    noPositionIds: true,
    layerKvHeads: openelm270mLayerKvHeads,
    preferredEp: "webgpu",
    base: "/models/openelm-270m-instruct",
    gguf: "gguf/q4_k_m",
    tokenizer: "raw/tokenizer.json",
    tokenizerConfig: "raw/tokenizer_config.json",
    rawAddSpecialTokens: true,
    defaultRepetitionPenalty: 1.2,
  },
];

export const DEFAULT_MODEL_ID = "lfm2_5-350m-q4-k-m-gguf-wgsl";

export function getModel(id: string): ModelDef {
  const m = MODELS.find((model) => model.id === id);
  if (!m) throw new Error(`unknown model id: ${id}`);
  return m;
}

export function modelCacheUrls(model: ModelDef): string[] {
  const urls: string[] = [];
  const add = (file?: string) => {
    if (!file) return;
    urls.push(file.startsWith("http") ? file : `${model.base}/${file}`);
  };
  add(model.tokenizer);
  add(model.tokenizerConfig);
  add(model.gguf);
  return urls;
}

export function totalDownloadBytes(model: ModelDef, knownSizes: Record<string, number>): number {
  let total = 0;
  for (const f of [model.tokenizer, model.tokenizerConfig, model.gguf]) {
    if (f) total += knownSizes[f] ?? knownSizes[f.startsWith("http") ? f : `${model.base}/${f}`] ?? 0;
  }
  return total;
}

export function isQ8(model: ModelDef): boolean {
  return /q8/i.test(model.gguf);
}

export function modelQuant(model: ModelDef): string {
  const file = model.gguf.split("/").pop() ?? "";
  if (/q4_k_m/i.test(file)) return "Q4_K_M";
  if (/q8_0/i.test(file)) return "Q8_0";
  return file.replace(/\.gguf$/i, "");
}

export function modelDisplayLabel(model: ModelDef): string {
  return model.label.replace(/\s*\([^)]*\)\s*$/, "");
}

export function modelDisplayPitch(model: ModelDef): string {
  return model.pitch
    .replace(/\s*·\s*Q\d+(?:_K_M|_0)?\s*GGUF/gi, "")
    .replace(/\s*·\s*imported\s*GGUF\s*runtime/gi, "")
    .replace(/^\s*·\s*/, "")
    .replace(/\s*·\s*$/, "");
}

export interface ModelLogo {
  src: string;
  alt: string;
}

export function modelLogo(model: ModelDef): ModelLogo | null {
  const id = model.id.toLowerCase();
  if (id.startsWith("gemma-")) return { src: "/logos/google.svg", alt: "Google" };
  if (id.startsWith("granite-")) return { src: "/logos/ibm.svg", alt: "IBM" };
  if (id.startsWith("lfm")) return { src: "/logos/liquid.webp", alt: "Liquid AI" };
  if (id.startsWith("smollm")) return { src: "/logos/huggingface.svg", alt: "Hugging Face" };
  if (id.startsWith("openelm")) return { src: "/logos/apple.svg", alt: "Apple" };
  if (id.startsWith("qwen")) return { src: "/logos/qwen.svg", alt: "Qwen" };
  if (id.startsWith("monad-") || id.startsWith("baguettotron-")) return { src: "/logos/pleias.webp", alt: "PleIAs" };
  if (id.startsWith("gpt2-") || id.startsWith("distilgpt2-")) return { src: "/logos/openai.svg", alt: "OpenAI" };
  return null;
}
