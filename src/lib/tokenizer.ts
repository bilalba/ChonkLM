import { Tokenizer as HfTokenizer } from "@huggingface/tokenizers";
import { fetchCached } from "./cache";
import type { ModelDef } from "./registry";

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

type TokenizerConfig = Record<string, unknown>;

export class Tokenizer {
  private inner: HfTokenizer;
  private config: TokenizerConfig;
  private model: ModelDef;

  constructor(inner: HfTokenizer, config: TokenizerConfig, model: ModelDef) {
    this.inner = inner;
    this.config = config;
    this.model = model;
  }

  static async load(model: ModelDef): Promise<Tokenizer> {
    if (!model.tokenizer || !model.tokenizerConfig) {
      throw new Error(`Tokenizer.load: ${model.id} missing tokenizer metadata`);
    }
    const [tokBuf, cfgBuf] = await Promise.all([
      fetchCached(`${model.base}/${model.tokenizer}`),
      fetchCached(`${model.base}/${model.tokenizerConfig}`),
    ]);
    const tokJson = JSON.parse(new TextDecoder().decode(tokBuf));
    const cfgJson = JSON.parse(new TextDecoder().decode(cfgBuf)) as TokenizerConfig;
    return new Tokenizer(new HfTokenizer(tokJson, cfgJson), cfgJson, model);
  }

  encode(text: string, opts: { addSpecialTokens?: boolean } = {}): number[] {
    return this.inner.encode(text, { add_special_tokens: opts.addSpecialTokens ?? false }).ids;
  }

  encodeChat(messages: ChatMessage[]): number[] {
    return this.encode(renderChat(this.model, this.config, messages), { addSpecialTokens: false });
  }

  decode(ids: number[], skipSpecial = true): string {
    return this.inner.decode(ids, { skip_special_tokens: skipSpecial });
  }

  decodeOne(id: number, skipSpecial = true): string {
    return this.decode([id], skipSpecial);
  }
}

function renderChat(model: ModelDef, config: TokenizerConfig, messages: ChatMessage[]): string {
  const template = String(config.chat_template ?? "");
  if (model.id.startsWith("gemma-3-270m-it")) return renderGemma(messages);
  if (model.id.startsWith("granite-4.0-h-350m")) return renderGranite(messages);
  if (model.id.startsWith("lfm2_5-350m")) return renderChatml(messages, { bos: "<|startoftext|>" });
  if (model.id.startsWith("smollm2-")) return renderSmolLm(messages);
  if (model.id.startsWith("monad") || model.id.startsWith("baguettotron")) {
    return renderChatml(messages, { assistantPrefix: "<think>\n" });
  }
  if (model.id.startsWith("qwen3-")) return renderChatml(messages);

  if (template.includes("<start_of_turn>")) return renderGemma(messages);
  if (template.includes("<|start_of_role|>")) return renderGranite(messages);
  if (template.includes("<|im_start|>")) return renderChatml(messages);

  throw new Error(`No local chat renderer for ${model.id}`);
}

function renderChatml(
  messages: ChatMessage[],
  opts: { bos?: string; assistantPrefix?: string } = {},
): string {
  let out = opts.bos ?? "";
  for (const message of messages) {
    out += `<|im_start|>${message.role}\n${message.content}<|im_end|>\n`;
  }
  out += `<|im_start|>assistant\n${opts.assistantPrefix ?? ""}`;
  return out;
}

function renderSmolLm(messages: ChatMessage[]): string {
  const hasSystem = messages[0]?.role === "system";
  const withSystem = hasSystem
    ? messages
    : [
        {
          role: "system" as const,
          content: "You are a helpful AI assistant named SmolLM, trained by Hugging Face",
        },
        ...messages,
      ];
  return renderChatml(withSystem);
}

function renderGemma(messages: ChatMessage[]): string {
  let firstUserPrefix = "";
  let loopMessages = messages;
  if (messages[0]?.role === "system") {
    firstUserPrefix = `${messages[0].content}\n\n`;
    loopMessages = messages.slice(1);
  }

  let out = "<bos>\n";
  for (let i = 0; i < loopMessages.length; i++) {
    const message = loopMessages[i];
    const role = message.role === "assistant" ? "model" : message.role;
    const content = i === 0 && message.role === "user"
      ? `${firstUserPrefix}${message.content.trim()}`
      : message.content.trim();
    out += `<start_of_turn>${role}\n${content}<end_of_turn>\n`;
  }
  out += "<start_of_turn>model\n";
  return out;
}

function renderGranite(messages: ChatMessage[]): string {
  const defaultSystem = "You are a helpful assistant. Please ensure responses are professional, accurate, and safe.";
  let out = "";
  const first = messages[0];
  if (first?.role === "system") {
    out += `<|start_of_role|>system<|end_of_role|>${first.content}<|end_of_text|>\n`;
  } else {
    out += `<|start_of_role|>system<|end_of_role|>${defaultSystem}<|end_of_text|>\n`;
  }

  for (const message of messages) {
    if (message.role === "system" && message === first) continue;
    out += `<|start_of_role|>${message.role}<|end_of_role|>${message.content}<|end_of_text|>\n`;
  }
  out += "<|start_of_role|>assistant<|end_of_role|>";
  return out;
}
