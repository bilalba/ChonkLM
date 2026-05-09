// Adapter from chonklm's loaded-model shape to the svenflow/gemma-webgpu
// `GemmaEngine`.

import { createGemmaEngine, type GemmaEngine } from "gemma-webgpu";
import type { LoadStepCallback, ProgressCallback } from "./cache";
import type { ChatMessage } from "./tokenizer";
import type { ModelDef } from "./registry";

export interface LoadedGemmaModel {
  runtime: "gemma-webgpu";
  def: ModelDef;
  engine: GemmaEngine;
  ep: "webgpu";
  /**
   * Count of `ChatMessage[]` entries already replayed into the engine's
   * internal history. The playground rebuilds `history` on every send; we
   * only push the delta into the engine via `addUserMessage`.
   */
  turnsReplayed: number;
}

export async function loadGemmaModel(
  model: ModelDef,
  onProgress?: ProgressCallback,
  onStep?: LoadStepCallback,
): Promise<LoadedGemmaModel> {
  if (!model.gguf) {
    throw new Error(`gemma-webgpu model ${model.id} missing 'gguf' URL`);
  }
  const url = model.gguf.startsWith("http")
    ? model.gguf
    : `${model.base}/${model.gguf}`;

  onStep?.({ step: "runtime", detail: "creating Gemma WebGPU engine" });
  const engine = await createGemmaEngine({
    model: url,
    contextLength: model.maxContext,
    onProgress: (p) => {
      onStep?.({ step: "weights", detail: p.status });
      // Translate gemma-webgpu progress → our cache.ProgressCallback shape
      // so the existing progress bar wiring keeps working.
      onProgress?.({
        url,
        loaded: p.loaded,
        total: p.total > 0 ? p.total : null,
        fromNetwork: true,
        status: p.status,
      });
    },
  });

  onStep?.({ step: "ready", detail: "Gemma WebGPU engine ready" });
  return { runtime: "gemma-webgpu", def: model, engine, ep: "webgpu", turnsReplayed: 0 };
}

export interface GemmaGenerateOptions {
  maxNewTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  repetitionPenalty?: number;
  /** Called once per generated token with the cumulative decoded string. */
  onToken?: (info: { text: string; cumulative: string }) => void;
  shouldStop?: () => boolean;
}

/**
 * Stream a reply from a loaded Gemma engine. The engine maintains its own
 * conversation history (KV-cache-friendly multi-turn), so we just hand it
 * the most recent user turn — the upstream history pushed earlier in this
 * conversation is reflected via prior `addUserMessage` + `generate` calls.
 *
 * For the playground's pattern (rebuild from `history: ChatMessage[]`
 * each turn), we replay any messages the engine hasn't seen yet, then
 * stream the assistant reply.
 */
export async function generateGemma(
  loaded: LoadedGemmaModel,
  history: ChatMessage[],
  opts: GemmaGenerateOptions = {},
): Promise<{ text: string; tokensPerSec: number; generatedTokens: number }> {
  // The playground passes the full history each turn. The engine maintains
  // its own conversation state internally (KV-cache-friendly multi-turn) —
  // we just push the delta we haven't yet replayed.
  const replay = history.slice(loaded.turnsReplayed);
  for (const msg of replay) {
    if (msg.role === "user") {
      loaded.engine.addUserMessage(msg.content);
    }
    // Assistant turns are written by the engine itself when generate()
    // finishes; we don't replay them.
  }
  loaded.turnsReplayed = history.length;

  const t0 = performance.now();
  let cumulative = "";
  let tokenCount = 0;
  for await (const piece of loaded.engine.generate({
    maxTokens: opts.maxNewTokens,
    temperature: opts.temperature,
    topP: opts.topP,
    repPenalty: opts.repetitionPenalty,
  })) {
    if (opts.shouldStop?.()) break;
    cumulative += piece;
    tokenCount += 1;
    opts.onToken?.({ text: piece, cumulative });
  }
  const elapsed = (performance.now() - t0) / 1000;
  return { text: cumulative, tokensPerSec: tokenCount / Math.max(elapsed, 1e-3), generatedTokens: tokenCount };
}

export function disposeGemma(loaded: LoadedGemmaModel): void {
  loaded.engine.dispose();
}

export function resetGemmaConversation(loaded: LoadedGemmaModel): void {
  loaded.engine.resetConversation();
  loaded.turnsReplayed = 0;
}

export function restoreGemmaConversation(loaded: LoadedGemmaModel, history: ChatMessage[]): void {
  loaded.engine.resetConversation();
  const state = loaded.engine as GemmaEngine & {
    conversationHistory?: Array<{ role: "user" | "model"; text: string }>;
  };
  if (Array.isArray(state.conversationHistory)) {
    state.conversationHistory = history
      .filter((msg) => msg.role === "user" || msg.role === "assistant")
      .map((msg) => ({
        role: msg.role === "assistant" ? "model" : "user",
        text: msg.content,
      }));
    loaded.turnsReplayed = history.length;
  }
}
