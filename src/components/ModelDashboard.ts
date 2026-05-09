import {
  DEFAULT_MODEL_ID,
  MODELS,
  getModel,
  isQ8,
  modelCacheUrls,
  modelDisplayLabel,
  modelDisplayPitch,
  modelQuant,
  type ModelDef,
} from "../lib/registry";
import { clearCache, fetchCached, inspectCachedUrls, storageEstimate } from "../lib/cache";
import type { LoadStepCallback, LoadStepEvent, PartEvent, ProgressCallback } from "../lib/cache";
import type { LoadedGemmaModel } from "../lib/gemma-runtime";
import type { LoadedLfmWgslModel } from "../lib/lfm-wgsl-runtime";
import type { LoadedLlamaWgslModel } from "../lib/llama-wgsl-runtime";
import type { LoadedGraniteWgslModel } from "../lib/granite-wgsl-runtime";
import type { LoadedGptWgslModel } from "../lib/gpt-wgsl-runtime";
import type { ChatMessage } from "../lib/tokenizer";

type AnyLoaded =
  | LoadedGemmaModel
  | LoadedLfmWgslModel
  | LoadedLlamaWgslModel
  | LoadedGraniteWgslModel
  | LoadedGptWgslModel;
type WgslLoaded =
  | LoadedLfmWgslModel
  | LoadedLlamaWgslModel
  | LoadedGraniteWgslModel
  | LoadedGptWgslModel;
type TranscriptRole = "you" | "assistant" | "completion" | "system";

interface TranscriptEntry {
  role: TranscriptRole;
  text: string;
  at: number;
}

interface StoredConversation {
  history: ChatMessage[];
  transcript: TranscriptEntry[];
  updatedAt: number;
}

interface StoredState {
  selectedId: string;
  conversations: Record<string, StoredConversation>;
}

type GenerateOpts = {
  maxNewTokens: number;
  temperature: number;
  topP: number;
  topK: number;
  repetitionPenalty: number;
  shouldStop: () => boolean;
  onToken: (info: { cumulative: string }) => void;
};

// Per-runtime adapter. Bridges the two generate shapes (gemma takes
// ChatMessage[]; wgsl runtimes take token IDs) so the dashboard's hot paths
// don't switch on runtime. Adding a new runtime = add one entry to RUNTIMES.
type RuntimeAdapter = {
  load: (def: ModelDef, onProgress?: ProgressCallback, onStep?: LoadStepCallback) => Promise<AnyLoaded>;
  generate: (loaded: AnyLoaded, history: ChatMessage[], userText: string, opts: GenerateOpts) => Promise<{ text: string; tokensPerSec: number }>;
  reset: (loaded: AnyLoaded) => void;
  restore?: (loaded: AnyLoaded, history: ChatMessage[]) => void;
};

function wgslAdapter(
  load: RuntimeAdapter["load"],
  rawGenerate: (loaded: AnyLoaded, ids: number[], opts: GenerateOpts) => Promise<{ text: string; tokensPerSec: number }>,
  reset: RuntimeAdapter["reset"],
): RuntimeAdapter {
  return {
    load,
    generate: (loaded, history, userText, opts) => {
      const l = loaded as WgslLoaded;
      const ids = l.def.chat
        ? l.tokenizer.encodeChat(history)
        : l.tokenizer.encode(userText, { addSpecialTokens: !!l.def.rawAddSpecialTokens });
      return rawGenerate(loaded, ids, opts);
    },
    reset,
  };
}

const RUNTIMES: Record<string, () => Promise<RuntimeAdapter>> = {
  "gemma-webgpu": async () => {
    const m = await import("../lib/gemma-runtime");
    return {
      load: (def, p, s) => m.loadGemmaModel(def, p, s),
      generate: (loaded, history, _userText, opts) =>
        m.generateGemma(loaded as LoadedGemmaModel, history, opts),
      reset: (loaded) => m.resetGemmaConversation(loaded as LoadedGemmaModel),
      restore: (loaded, history) =>
        m.restoreGemmaConversation(loaded as LoadedGemmaModel, history),
    };
  },
  "lfm2-webgpu": async () => {
    const m = await import("../lib/lfm-wgsl-runtime");
    return wgslAdapter(
      (def, p, s) => m.loadLfmWgslModel(def, p, s),
      (loaded, ids, opts) => m.generateLfmWgsl(loaded as LoadedLfmWgslModel, ids, opts),
      (loaded) => m.resetLfmWgslConversation(loaded as LoadedLfmWgslModel),
    );
  },
  "llama-webgpu": async () => {
    const m = await import("../lib/llama-wgsl-runtime");
    return wgslAdapter(
      (def, p, s) => m.loadLlamaWgslModel(def, p, s),
      (loaded, ids, opts) => m.generateLlamaWgsl(loaded as LoadedLlamaWgslModel, ids, opts),
      (loaded) => m.resetLlamaWgslConversation(loaded as LoadedLlamaWgslModel),
    );
  },
  "granite-webgpu": async () => {
    const m = await import("../lib/granite-wgsl-runtime");
    return wgslAdapter(
      (def, p, s) => m.loadGraniteWgslModel(def, p, s),
      (loaded, ids, opts) => m.generateGraniteWgsl(loaded as LoadedGraniteWgslModel, ids, opts),
      (loaded) => m.resetGraniteWgslConversation(loaded as LoadedGraniteWgslModel),
    );
  },
  "gpt-webgpu": async () => {
    const m = await import("../lib/gpt-wgsl-runtime");
    return wgslAdapter(
      (def, p, s) => m.loadGptWgslModel(def, p, s),
      // GPT runtime caps at 128 regardless of thinking mode — preserved
      // from the prior switch.
      (loaded, ids, opts) =>
        m.generateGptWgsl(loaded as LoadedGptWgslModel, ids, { ...opts, maxNewTokens: 128 }),
      (loaded) => m.resetGptWgslConversation(loaded as LoadedGptWgslModel),
    );
  },
};

const adapterCache = new Map<string, Promise<RuntimeAdapter>>();
function getAdapter(runtime: string): Promise<RuntimeAdapter> {
  let cached = adapterCache.get(runtime);
  if (!cached) {
    const factory = RUNTIMES[runtime];
    if (!factory) return Promise.reject(new Error(`unsupported runtime: ${runtime}`));
    cached = factory();
    adapterCache.set(runtime, cached);
  }
  return cached;
}

const STORE_KEY = "chonklm:conversations:v1";
const DEV_MODELS_KEY = "chonklm:show-dev-models:v1";
const Q8_MODELS_KEY = "chonklm:show-q8-models:v1";
const params = new URLSearchParams(window.location.search);
const devParam = params.get("dev");
if (devParam === "1" || devParam === "true") localStorage.setItem(DEV_MODELS_KEY, "1");
if (devParam === "0" || devParam === "false") localStorage.removeItem(DEV_MODELS_KEY);
const showDevModels =
  devParam === "1" ||
  devParam === "true" ||
  (devParam == null && localStorage.getItem(DEV_MODELS_KEY) === "1");
const q8Param = params.get("q8");
if (q8Param === "1" || q8Param === "true") localStorage.setItem(Q8_MODELS_KEY, "1");
if (q8Param === "0" || q8Param === "false") localStorage.removeItem(Q8_MODELS_KEY);
const showQ8Models =
  q8Param === "1" ||
  q8Param === "true" ||
  (q8Param == null && localStorage.getItem(Q8_MODELS_KEY) === "1");
const visibleModels = MODELS.filter(
  (m) => (showDevModels || !m.devOnly) && (showQ8Models || !isQ8(m)),
);
const visibleIds = new Set(visibleModels.map((m) => m.id));

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};

const dashboardSection = $("dashboard");
const modelList = $("model-list");
const storageSummary = $("storage-summary");
const selectedLabel = $("selected-label");
const selectedPitch = $("selected-pitch");
const chatPanel = $("chat-panel");
const chatLog = $("chat-log");
const chatEmpty = $("chat-empty");
const chatInput = $<HTMLTextAreaElement>("chat-input");
const chatStatus = $("chat-status");
const sendStopLink = $<HTMLAnchorElement>("send-stop-link");
const resetLink = $<HTMLAnchorElement>("reset-link");
const generatingIndicator = $("chat-generating");
const epLabel = $("ep-label");
const rateLabel = $("rate-label");
modelList.dataset.showDev = showDevModels ? "1" : "";
modelList.dataset.showQ8 = showQ8Models ? "1" : "";

let appState = loadStoredState();
const hashId = decodeURIComponent(window.location.hash.replace(/^#/, ""));
let selectedId = visibleIds.has(hashId) ? hashId : visibleIds.has(appState.selectedId) ? appState.selectedId : DEFAULT_MODEL_ID;
let loaded: AnyLoaded | null = null;
let history: ChatMessage[] = [];
let transcript: TranscriptEntry[] = [];
let stopFlag = false;
let busy = false;
let generating = false;
let seenProgress: Record<string, { loaded: number; total: number | null; fromNetwork: boolean }> = {};

function loadStoredState(): StoredState {
  const fallback: StoredState = { selectedId: DEFAULT_MODEL_ID, conversations: {} };
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw) as Partial<StoredState>;
    const conversations: Record<string, StoredConversation> = {};
    if (parsed.conversations && typeof parsed.conversations === "object") {
      for (const [id, value] of Object.entries(parsed.conversations)) {
        if (!visibleIds.has(id)) continue;
        conversations[id] = sanitizeConversation(value);
      }
    }
    return {
      selectedId: typeof parsed.selectedId === "string" ? parsed.selectedId : DEFAULT_MODEL_ID,
      conversations,
    };
  } catch {
    return fallback;
  }
}

function sanitizeConversation(value: unknown): StoredConversation {
  const v = value as Partial<StoredConversation>;
  return {
    history: Array.isArray(v.history) ? v.history.filter(isChatMessage) : [],
    transcript: Array.isArray(v.transcript) ? v.transcript.filter(isTranscriptEntry) : [],
    updatedAt: typeof v.updatedAt === "number" ? v.updatedAt : 0,
  };
}

function isChatMessage(value: unknown): value is ChatMessage {
  const v = value as ChatMessage;
  return (
    (v?.role === "system" || v?.role === "user" || v?.role === "assistant") &&
    typeof v.content === "string"
  );
}

function isTranscriptEntry(value: unknown): value is TranscriptEntry {
  const v = value as TranscriptEntry;
  return (
    (v?.role === "you" || v?.role === "assistant" || v?.role === "completion" || v?.role === "system") &&
    typeof v.text === "string" &&
    typeof v.at === "number"
  );
}

function persistState() {
  appState.selectedId = selectedId;
  appState.conversations[selectedId] = {
    history: history.slice(),
    transcript: transcript.slice(),
    updatedAt: Date.now(),
  };
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(appState));
  } catch (err) {
    console.warn("conversation save failed", err);
  }
}

function loadConversationForSelected() {
  const stored = appState.conversations[selectedId];
  history = stored?.history.slice() ?? [];
  transcript = stored?.transcript.slice() ?? [];
  renderTranscript();
}

function setChatStatus(text: string) {
  chatStatus.textContent = text;
  chatStatus.hidden = !text;
}

function getCacheState(id: string): string {
  const badge = document.querySelector<HTMLElement>(`[data-cache-id="${id}"]`);
  return badge?.dataset.state ?? "checking";
}

function updateChatInputState() {
  const def = getModel(selectedId);
  const cached = getCacheState(selectedId) === "cached";
  const loadedHere = loaded != null && loaded.def.id === selectedId;
  const ready = cached || loadedHere;
  const loadingIntoGpu = busy && !loadedHere;
  chatInput.disabled = !ready || loadingIntoGpu;
  if (loadingIntoGpu) {
    chatInput.placeholder = `loading ${modelDisplayLabel(def)}…`;
  } else if (!ready) {
    chatInput.placeholder = "download a model to start chatting";
  } else {
    chatInput.placeholder = def.chat ? "message selected model" : "start a completion";
  }
  // Single send/stop control. Becomes a stop link only while a generation
  // is in flight; loading and idle both show "send".
  sendStopLink.textContent = generating ? "stop" : "send";
  sendStopLink.dataset.mode = generating ? "stop" : "send";
  sendStopLink.classList.toggle("is-disabled", !generating && (busy || !ready));
  resetLink.classList.toggle("is-disabled", generating);
  generatingIndicator.hidden = !generating;
}

function updateSelectedUi() {
  const def = getModel(selectedId);
  selectedLabel.textContent = modelDisplayLabel(def);
  selectedPitch.textContent = modelDisplayPitch(def);
  chatInput.placeholder = def.chat ? "message selected model" : "start a completion";
  document.querySelectorAll<HTMLElement>(".model-row").forEach((row) => {
    row.dataset.selected = row.dataset.id === selectedId ? "1" : "";
  });
  document.querySelectorAll<HTMLAnchorElement>(".model-pick").forEach((a) => {
    a.classList.toggle("active", a.dataset.id === selectedId);
  });
}

function selectModel(id: string) {
  if (busy || !visibleIds.has(id)) return;
  if (id !== selectedId) {
    selectedId = id;
    loaded = null;
    rateLabel.textContent = "- tok/s";
    appState.selectedId = selectedId;
    loadConversationForSelected();
    persistState();
  }
  updateSelectedUi();
  updateChatInputState();
  setMobileView("chat");
  // Auto-load only if the bytes are already cached. We do not start a
  // download on selection; the user must click the explicit `download`
  // action for that. (When that download finishes for the still-selected
  // model, downloadModel kicks off the load on its own.)
  if ((!loaded || loaded.def.id !== selectedId) && getCacheState(selectedId) === "cached") {
    void handleLoad();
  }
}

function formatBytes(bytes?: number): string {
  if (!bytes || bytes <= 0) return "0 MB";
  if (bytes >= 1_073_741_824) return `${(bytes / 1_073_741_824).toFixed(1)} GB`;
  return `${(bytes / 1_048_576).toFixed(1)} MB`;
}

async function refreshCacheStatus() {
  storageSummary.textContent = "checking cache";
  let persisted: boolean | null = null;
  try {
    if (navigator.storage?.persisted) persisted = await navigator.storage.persisted();
  } catch {
    persisted = null;
  }

  const estimate = await storageEstimate();
  const used = estimate?.usage ? formatBytes(estimate.usage) : "usage unknown";
  const quota = estimate?.quota ? ` / ${formatBytes(estimate.quota)}` : "";
  const persistenceLabel = persisted == null ? "cache" : persisted ? "persisted cache" : "best-effort cache";
  storageSummary.textContent = `${persistenceLabel} · ${used}${quota}`;

  await Promise.all(
    visibleModels.map(async (def) => {
      const badge = document.querySelector<HTMLElement>(`[data-cache-id="${def.id}"]`);
      if (!badge) return;
      badge.textContent = "checking";
      badge.dataset.state = "checking";
      const urls = modelCacheUrls(def);
      const status = await inspectCachedUrls(urls);
      if (!status.supported) {
        badge.textContent = "cache unavailable";
        badge.dataset.state = "unsupported";
        return;
      }
      if (status.total === 0) {
        badge.textContent = "not tracked";
        badge.dataset.state = "unsupported";
        return;
      }
      if (status.cached.length === status.total) {
        badge.textContent = "cached";
        badge.dataset.state = "cached";
        return;
      }
      if (status.cached.length > 0) {
        badge.textContent = `partial ${status.cached.length}/${status.total}`;
        badge.dataset.state = "partial";
        return;
      }
      badge.textContent = "not cached";
      badge.dataset.state = "missing";
    }),
  );
  updateChatInputState();
}

async function handleLoad() {
  if (busy) return;
  busy = true;
  stopFlag = false;
  loaded = null;
  seenProgress = {};
  updateChatInputState();

  const def = getModel(selectedId);
  const defLabel = modelDisplayLabel(def);
  setChatStatus(`loading ${defLabel}…`);

  // Step + progress events arrive interleaved during a load. Track the
  // latest of each so the status line keeps both a phase ("uploading
  // layer 5 / 24") and a percent ("16%") visible simultaneously.
  //
  // Step-based progress is preferred when present: byte-based progress
  // is misleading on cached loads, where every cache-hit event reports
  // `loaded == total` and the running fraction sits at 100% from the
  // first event. Layer-step events give a real 0..N/N counter.
  let lastDetail = "";
  let lastStepPct: string | null = null;
  let lastBytePct: string | null = null;
  let lastSource = "loading";
  const pushStatus = () => {
    const parts = [`${lastSource} ${defLabel}`];
    if (lastDetail) parts.push(lastDetail);
    const pct = lastStepPct ?? lastBytePct;
    if (pct) parts.push(pct);
    setChatStatus(parts.join(" · "));
  };

  try {
    const onStep = (e: LoadStepEvent) => {
      lastDetail = e.detail || e.step;
      lastStepPct = e.progress && e.progress.total > 0
        ? `${Math.round((e.progress.current / e.progress.total) * 100)}%`
        : null;
      pushStatus();
    };
    const onProgress = (e: { url: string; loaded: number; total: number | null; fromNetwork: boolean; status?: string }) => {
      const prev = seenProgress[e.url];
      seenProgress[e.url] = {
        loaded: e.loaded,
        total: e.total ?? prev?.total ?? null,
        fromNetwork: e.fromNetwork,
      };
      let loadedSum = 0;
      let totalSum = 0;
      let unknown = false;
      for (const v of Object.values(seenProgress)) {
        loadedSum += v.loaded;
        if (v.total == null) unknown = true;
        else totalSum += v.total;
      }
      const pct = !unknown && totalSum > 0 ? Math.min(100, (loadedSum / totalSum) * 100) : null;
      lastSource = e.fromNetwork ? "downloading" : "loading";
      lastBytePct = pct != null ? `${pct.toFixed(0)}%` : null;
      pushStatus();
    };

    const adapter = await getAdapter(def.runtime);
    loaded = await adapter.load(def, onProgress, onStep);
    if (history.length > 0) adapter.restore?.(loaded, history);

    epLabel.textContent = `GPU: ${loaded.ep}`;
    setChatStatus("");
    await refreshCacheStatus();
  } catch (err: unknown) {
    console.error(err);
    const msg = err instanceof Error ? err.message : String(err);
    setChatStatus(`load failed: ${msg}`);
    loaded = null;
  } finally {
    busy = false;
    updateChatInputState();
  }
}

function scrollChat() {
  chatPanel.scrollTop = chatPanel.scrollHeight;
}

function setEmptyVisible() {
  chatEmpty.style.display = transcript.length === 0 ? "block" : "none";
}

function appendChatLine(role: string, text: string): HTMLSpanElement {
  chatEmpty.style.display = "none";
  const div = document.createElement("div");
  const isUser = role === "you";
  div.className = `chat-turn chat-turn--${isUser ? "user" : "agent"}`;
  const tag = document.createElement("span");
  tag.className = "chat-role";
  tag.textContent = role;
  const body = document.createElement("span");
  body.className = "chat-body";
  body.textContent = text;
  div.appendChild(tag);
  div.appendChild(body);
  chatLog.appendChild(div);
  scrollChat();
  return body;
}

interface ThinkingRenderer {
  update: (cumulative: string) => void;
}

function appendThinkingChatLine(role: string): ThinkingRenderer {
  chatEmpty.style.display = "none";
  const div = document.createElement("div");
  div.className = "chat-turn chat-turn--agent";
  const tag = document.createElement("span");
  tag.className = "chat-role";
  tag.textContent = role;
  div.appendChild(tag);

  const thinkingWrap = document.createElement("details");
  thinkingWrap.className = "thinking-wrap";
  thinkingWrap.open = true;
  const thinkingLabel = document.createElement("summary");
  thinkingLabel.className = "thinking-label";
  thinkingLabel.textContent = "thinking";
  const thinkingBody = document.createElement("div");
  thinkingBody.className = "thinking-body";
  thinkingWrap.appendChild(thinkingLabel);
  thinkingWrap.appendChild(thinkingBody);
  div.appendChild(thinkingWrap);

  const answerBody = document.createElement("div");
  div.appendChild(answerBody);
  chatLog.appendChild(div);
  scrollChat();

  let userToggled = false;
  let autoCollapsed = false;
  thinkingLabel.addEventListener("click", () => {
    userToggled = true;
  });

  return {
    update(cumulative: string) {
      const closeIdx = cumulative.indexOf("</think>");
      let thinking: string;
      let answer: string;
      if (closeIdx >= 0) {
        thinking = cumulative.slice(0, closeIdx);
        answer = cumulative.slice(closeIdx + "</think>".length);
      } else {
        thinking = cumulative;
        answer = "";
      }
      thinking = thinking.replace(/^\s*<think>\s*/, "").trimEnd();
      answer = answer.replace(/^\s+/, "");
      thinkingBody.textContent = thinking;
      thinkingWrap.style.display = thinking ? "block" : "none";
      if (!userToggled && !autoCollapsed && closeIdx >= 0 && answer) {
        thinkingWrap.open = false;
        autoCollapsed = true;
      }
      answerBody.textContent = answer;
      scrollChat();
    },
  };
}

function renderTranscript() {
  chatLog.textContent = "";
  const def = getModel(selectedId);
  for (const entry of transcript) {
    if ((entry.role === "assistant" || entry.role === "completion") && def.thinking) {
      const renderer = appendThinkingChatLine(entry.role);
      renderer.update(entry.text);
    } else {
      appendChatLine(entry.role, entry.text);
    }
  }
  setEmptyVisible();
}

async function handleSend() {
  if (busy) return;

  const userText = chatInput.value.trim();
  if (!userText) return;

  if (getCacheState(selectedId) !== "cached" && !(loaded && loaded.def.id === selectedId)) return;

  if (!loaded || loaded.def.id !== selectedId) {
    await handleLoad();
    if (!loaded || loaded.def.id !== selectedId) return;
  }

  busy = true;
  generating = true;
  stopFlag = false;
  chatInput.value = "";
  updateChatInputState();

  transcript.push({ role: "you", text: userText, at: Date.now() });
  appendChatLine("you", userText);
  if (loaded.def.chat) history.push({ role: "user", content: userText });
  persistState();

  const role: TranscriptRole = loaded.def.chat ? "assistant" : "completion";
  const useThinking = !!loaded.def.thinking;
  const thinkingRenderer = useThinking ? appendThinkingChatLine(role) : null;
  const assistantBody = thinkingRenderer ? null : appendChatLine(role, "");

  try {
    const onToken = ({ cumulative }: { cumulative: string }) => {
      if (thinkingRenderer) {
        thinkingRenderer.update(cumulative);
      } else if (assistantBody) {
        assistantBody.textContent = cumulative;
        scrollChat();
      }
    };

    const adapter = await getAdapter(loaded.def.runtime);
    const res = await adapter.generate(loaded, history, userText, {
      maxNewTokens: useThinking ? 512 : 128,
      temperature: loaded.def.defaultTemperature ?? 0,
      topP: loaded.def.defaultTopP ?? 1,
      topK: loaded.def.defaultTopK ?? 0,
      repetitionPenalty: loaded.def.defaultRepetitionPenalty ?? 1,
      shouldStop: () => stopFlag,
      onToken,
    });

    if (loaded.def.chat) history.push({ role: "assistant", content: res.text });
    transcript.push({ role, text: res.text, at: Date.now() });
    rateLabel.textContent = `${res.tokensPerSec.toFixed(1)} tok/s`;
    persistState();
  } catch (err: unknown) {
    console.error(err);
    const msg = err instanceof Error ? err.message : String(err);
    const errorText = `[error] ${msg}`;
    if (thinkingRenderer) {
      thinkingRenderer.update(errorText);
    } else if (assistantBody) {
      assistantBody.textContent = errorText;
    }
    if (loaded.def.chat) {
      const last = history[history.length - 1];
      if (last && last.role === "user") history.push({ role: "assistant", content: errorText });
    }
    transcript.push({ role, text: errorText, at: Date.now() });
    persistState();
  } finally {
    busy = false;
    generating = false;
    updateChatInputState();
  }
}

async function resetConversation() {
  history = [];
  transcript = [];
  renderTranscript();
  persistState();
  rateLabel.textContent = "- tok/s";
  if (loaded) {
    const adapter = await getAdapter(loaded.def.runtime);
    adapter.reset(loaded);
  }
}

document.querySelectorAll<HTMLAnchorElement>(".model-pick").forEach((a) => {
  a.addEventListener("click", (e) => {
    e.preventDefault();
    selectModel(a.dataset.id || DEFAULT_MODEL_ID);
  });
});

// Clicking anywhere in the row selects the model, except clicks on the
// per-row action links (info / download / details) which keep their own
// behavior.
document.querySelectorAll<HTMLElement>(".model-row").forEach((row) => {
  row.addEventListener("click", (e) => {
    const target = e.target as HTMLElement | null;
    if (target?.closest("[data-info-id], [data-download-id], [data-download-details-id]")) return;
    const id = row.dataset.id;
    if (!id) return;
    e.preventDefault();
    selectModel(id);
  });
});

$("refresh-cache-link").addEventListener("click", (e) => {
  e.preventDefault();
  refreshCacheStatus();
});

$("clear-cache-link").addEventListener("click", async (e) => {
  e.preventDefault();
  if (busy) return;
  await clearCache();
  partsByModel.clear();
  partsScanned.clear();
  if (currentDownloadModalId) renderDownloadParts(currentDownloadModalId);
  await refreshCacheStatus();
});

sendStopLink.addEventListener("click", (e) => {
  e.preventDefault();
  if (sendStopLink.classList.contains("is-disabled")) return;
  if (generating) {
    stopFlag = true;
    // Visual feedback: stop is one-shot — disable until the runtime returns
    // and updateChatInputState flips the link back to "send".
    sendStopLink.classList.add("is-disabled");
  } else {
    handleSend();
  }
});

resetLink.addEventListener("click", (e) => {
  e.preventDefault();
  if (generating) return;
  resetConversation();
});

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
});

function setMobileView(view: "picker" | "chat") {
  dashboardSection.dataset.mobileView = view;
  document.querySelectorAll<HTMLAnchorElement>(".mobile-tab").forEach((tab) => {
    const isActive = tab.dataset.mobileTarget === view;
    if (isActive) tab.setAttribute("aria-current", "true");
    else tab.removeAttribute("aria-current");
  });
}

document.querySelectorAll<HTMLAnchorElement>(".mobile-tab").forEach((tab) => {
  tab.addEventListener("click", (e) => {
    e.preventDefault();
    const target = tab.dataset.mobileTarget;
    if (target === "picker" || target === "chat") setMobileView(target);
  });
});

const infoOverlay = $("info-overlay");
const infoTitle = $("info-title");
const infoPitch = $("info-pitch");
const infoList = $("info-list");

function openInfo(id: string) {
  const def = MODELS.find((m) => m.id === id);
  if (!def) return;
  infoTitle.textContent = def.label;
  infoPitch.textContent = def.pitch;
  const tags: string[] = [];
  tags.push(def.chat ? "chat" : "completion");
  if (def.thinking) tags.push("thinking");
  if (def.devOnly) tags.push("dev");
  if (isQ8(def)) tags.push("q8");
  const rows: [string, string][] = [
    ["runtime", def.runtime],
    ["quantization", modelQuant(def)],
    ["tags", tags.join(" · ")],
    ["layers", String(def.layers)],
    ["vocab", def.vocab.toLocaleString()],
    ["max context", def.maxContext.toLocaleString()],
    ["gguf", `${def.base}/${def.gguf}`],
  ];
  if (def.tokenizer) rows.push(["tokenizer", `${def.base}/${def.tokenizer}`]);
  infoList.textContent = "";
  for (const [k, v] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = k;
    const dd = document.createElement("dd");
    dd.textContent = v;
    infoList.appendChild(dt);
    infoList.appendChild(dd);
  }
  infoOverlay.hidden = false;
  infoOverlay.setAttribute("aria-hidden", "false");
}

function closeInfo() {
  infoOverlay.hidden = true;
  infoOverlay.setAttribute("aria-hidden", "true");
}

infoOverlay.querySelectorAll<HTMLElement>("[data-info-dismiss]").forEach((el) => {
  el.addEventListener("click", (e) => {
    if (el.tagName === "A") e.preventDefault();
    closeInfo();
  });
});

document.addEventListener("keydown", (e) => {
  if (!infoOverlay.hidden && e.key === "Escape") closeInfo();
});

document.querySelectorAll<HTMLAnchorElement>("[data-info-id]").forEach((a) => {
  a.addEventListener("click", (e) => {
    e.preventDefault();
    const id = a.dataset.infoId;
    if (id) openInfo(id);
  });
});

const historyOverlay = $("history-overlay");
const historyList = $("history-list");
const historyEmpty = $("history-empty");

function formatHistoryTime(ts: number): string {
  if (!ts) return "—";
  const d = new Date(ts);
  const now = Date.now();
  const diff = now - ts;
  if (diff < 60_000) return "just now";
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
  if (diff < 7 * 86_400_000) return `${Math.floor(diff / 86_400_000)}d ago`;
  return d.toLocaleDateString();
}

function getConversationPreview(conv: StoredConversation): string {
  const firstUser = conv.transcript.find((t) => t.role === "you");
  const fallback = conv.transcript[0];
  const text = (firstUser ?? fallback)?.text ?? "";
  return text.replace(/\s+/g, " ").trim();
}

function renderHistoryList() {
  historyList.textContent = "";
  const entries = Object.entries(appState.conversations)
    .filter(([, conv]) => conv.transcript.length > 0 || conv.history.length > 0)
    .sort(([, a], [, b]) => (b.updatedAt || 0) - (a.updatedAt || 0));

  if (entries.length === 0) {
    historyEmpty.hidden = false;
    return;
  }
  historyEmpty.hidden = true;

  for (const [id, conv] of entries) {
    const def = MODELS.find((m) => m.id === id);
    const label = def ? modelDisplayLabel(def) : id;
    const preview = getConversationPreview(conv) || "(empty)";
    const turns = conv.transcript.length;

    const row = document.createElement("li");
    row.className = "history-row";

    const title = document.createElement("div");
    title.className = "history-row-title";
    title.textContent = label;

    const meta = document.createElement("span");
    meta.className = "history-meta";
    meta.textContent = `${turns} msg · ${formatHistoryTime(conv.updatedAt)}`;

    const previewEl = document.createElement("div");
    previewEl.className = "history-preview";
    previewEl.textContent = preview;

    const actions = document.createElement("p");
    actions.className = "history-actions";
    const openLink = document.createElement("a");
    openLink.className = "link";
    openLink.href = "#";
    openLink.textContent = def && visibleIds.has(id) ? "open" : "open (hidden model)";
    openLink.addEventListener("click", (e) => {
      e.preventDefault();
      if (!def) return;
      if (!visibleIds.has(id)) {
        if (def.devOnly) modelList.dataset.showDev = "1";
        if (isQ8(def)) modelList.dataset.showQ8 = "1";
        visibleModels.push(def);
        visibleIds.add(id);
      }
      closeHistory();
      selectModel(id);
    });
    const sep = document.createElement("span");
    sep.className = "pipe";
    sep.textContent = "|";
    const deleteLink = document.createElement("a");
    deleteLink.className = "link";
    deleteLink.href = "#";
    deleteLink.textContent = "delete";
    deleteLink.addEventListener("click", (e) => {
      e.preventDefault();
      delete appState.conversations[id];
      try {
        localStorage.setItem(STORE_KEY, JSON.stringify(appState));
      } catch (err) {
        console.warn("history delete failed", err);
      }
      if (id === selectedId) {
        history = [];
        transcript = [];
        renderTranscript();
      }
      renderHistoryList();
    });
    actions.appendChild(openLink);
    actions.appendChild(document.createTextNode(" "));
    actions.appendChild(sep);
    actions.appendChild(document.createTextNode(" "));
    actions.appendChild(deleteLink);

    row.appendChild(title);
    row.appendChild(meta);
    row.appendChild(previewEl);
    row.appendChild(actions);
    historyList.appendChild(row);
  }
}

function openHistory() {
  renderHistoryList();
  historyOverlay.hidden = false;
  historyOverlay.setAttribute("aria-hidden", "false");
}

function closeHistory() {
  historyOverlay.hidden = true;
  historyOverlay.setAttribute("aria-hidden", "true");
}

historyOverlay.querySelectorAll<HTMLElement>("[data-history-dismiss]").forEach((el) => {
  el.addEventListener("click", (e) => {
    if (el.tagName === "A") e.preventDefault();
    closeHistory();
  });
});

document.addEventListener("keydown", (e) => {
  if (!historyOverlay.hidden && e.key === "Escape") closeHistory();
});

$("history-link").addEventListener("click", (e) => {
  e.preventDefault();
  openHistory();
});

// --- download visualizer ----------------------------------------------

// Per-model part state. Insertion order = announce order
// (tokenizer files → manifest → shards), so iteration is stable for
// the modal layout. We mutate in place so live re-renders are O(parts).
const partsByModel = new Map<string, Map<string, PartEvent>>();
const partsScanned = new Set<string>();
let currentDownloadModalId: string | null = null;

const downloadOverlay = $("download-overlay");
const downloadTitle = $("download-title");
const downloadSummaryStatus = $("download-summary-status");
const downloadSummaryBytes = $("download-summary-bytes");
const downloadSummaryPct = $("download-summary-pct");
const downloadParts = $("download-parts");
const downloadEmpty = $("download-empty");

function getOrInitParts(id: string): Map<string, PartEvent> {
  let m = partsByModel.get(id);
  if (!m) {
    m = new Map();
    partsByModel.set(id, m);
  }
  return m;
}

function recordPart(id: string, e: PartEvent): void {
  const m = getOrInitParts(id);
  const existing = m.get(e.url);
  if (existing) {
    existing.loaded = e.loaded;
    existing.total = e.total;
    existing.status = e.status;
    existing.fromNetwork = e.fromNetwork;
    if (e.parent !== undefined) existing.parent = e.parent;
  } else {
    m.set(e.url, { ...e });
  }
  if (currentDownloadModalId === id) renderDownloadParts(id);
}

function formatPartBytes(n: number): string {
  if (n >= 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MiB`;
  if (n >= 1024) return `${(n / 1024).toFixed(0)} KiB`;
  return `${n} B`;
}

function isShardedDirUrl(url: string): boolean {
  const path = url.split("?")[0].split("#")[0];
  const lastSlash = path.lastIndexOf("/");
  const tail = lastSlash >= 0 ? path.slice(lastSlash + 1) : path;
  return tail.length > 0 && !tail.includes(".");
}

function renderDownloadParts(id: string): void {
  const def = MODELS.find((m) => m.id === id);
  if (!def) return;
  downloadTitle.textContent = modelDisplayLabel(def);
  const parts = partsByModel.get(id);
  downloadParts.textContent = "";
  if (!parts || parts.size === 0) {
    downloadEmpty.hidden = false;
    downloadSummaryStatus.textContent = "no parts";
    downloadSummaryBytes.textContent = "0 B / 0 B";
    downloadSummaryPct.textContent = "0%";
    return;
  }
  downloadEmpty.hidden = true;
  let totalLoaded = 0;
  let totalSize = 0;
  let unknown = false;
  let activeCount = 0;
  let cachedCount = 0;
  let doneCount = 0;
  let errorCount = 0;
  const totalCount = parts.size;
  for (const p of parts.values()) {
    totalLoaded += p.loaded;
    if (p.total == null) unknown = true;
    else totalSize += p.total;
    if (p.status === "downloading") activeCount += 1;
    else if (p.status === "cached") cachedCount += 1;
    else if (p.status === "done") doneCount += 1;
    else if (p.status === "error") errorCount += 1;

    const li = document.createElement("li");
    li.className = "download-part";
    li.dataset.status = p.status;

    const name = document.createElement("span");
    name.className = "download-part__name";
    name.textContent = p.name;

    const status = document.createElement("span");
    status.className = "download-part__status";
    status.textContent = p.status;

    const bar = document.createElement("div");
    bar.className = "download-part__bar";
    const pct = p.total && p.total > 0
      ? Math.min(100, (p.loaded / p.total) * 100)
      : (p.status === "cached" || p.status === "done" ? 100 : 0);
    bar.style.setProperty("--pct", `${pct}%`);

    const bytes = document.createElement("span");
    bytes.className = "download-part__bytes";
    if (p.total && p.total > 0) {
      bytes.textContent = `${formatPartBytes(p.loaded)} / ${formatPartBytes(p.total)}`;
    } else if (p.status === "cached" || p.status === "done") {
      bytes.textContent = formatPartBytes(p.loaded);
    } else {
      bytes.textContent = "—";
    }

    li.appendChild(name);
    li.appendChild(status);
    li.appendChild(bar);
    li.appendChild(bytes);
    downloadParts.appendChild(li);
  }

  const finishedCount = cachedCount + doneCount;
  let summaryStatus: string;
  if (errorCount > 0) summaryStatus = `${errorCount} error${errorCount === 1 ? "" : "s"}`;
  else if (finishedCount === totalCount) summaryStatus = cachedCount === totalCount ? "cached" : "ready";
  else if (activeCount > 0) summaryStatus = `downloading ${finishedCount}/${totalCount}`;
  else summaryStatus = "queued";
  downloadSummaryStatus.textContent = summaryStatus;
  downloadSummaryBytes.textContent = unknown
    ? `${formatPartBytes(totalLoaded)} / ?`
    : `${formatPartBytes(totalLoaded)} / ${formatPartBytes(totalSize)}`;
  if (!unknown && totalSize > 0) {
    const pct = Math.min(100, (totalLoaded / totalSize) * 100);
    downloadSummaryPct.textContent = `${pct.toFixed(0)}%`;
  } else if (finishedCount === totalCount && totalCount > 0) {
    downloadSummaryPct.textContent = "100%";
  } else {
    downloadSummaryPct.textContent = "—";
  }
}

function seedKnownParts(id: string): void {
  const def = MODELS.find((m) => m.id === id);
  if (!def) return;
  for (const url of modelCacheUrls(def)) {
    if (isShardedDirUrl(url)) {
      // We only know the manifest URL up front; shards appear once
      // the manifest is fetched.
      const manifestUrl = `${url.replace(/\/+$/, "")}/manifest.json`;
      recordPart(id, {
        url: manifestUrl,
        name: "manifest.json",
        loaded: 0,
        total: null,
        status: "queued",
        fromNetwork: false,
        parent: url,
      });
    } else {
      const name = url.split("/").pop() ?? url;
      recordPart(id, {
        url,
        name,
        loaded: 0,
        total: null,
        status: "queued",
        fromNetwork: false,
      });
    }
  }
}

// Cheap pass for already-cached models: hits the Cache API on every
// URL, yielding "cached" PartEvents with no network traffic.
async function scanCachedParts(id: string): Promise<void> {
  if (partsScanned.has(id)) return;
  const def = MODELS.find((m) => m.id === id);
  if (!def) return;
  partsScanned.add(id);
  try {
    await Promise.all(
      modelCacheUrls(def).map((u) =>
        fetchCached(u, undefined, (e) => recordPart(id, e)),
      ),
    );
  } catch {
    partsScanned.delete(id);
  }
}

function openDownloadDetails(id: string): void {
  const def = MODELS.find((m) => m.id === id);
  if (!def) return;
  currentDownloadModalId = id;
  const existing = partsByModel.get(id);
  if (!existing || existing.size === 0) {
    if (getCacheState(id) === "cached") {
      void scanCachedParts(id);
    } else {
      seedKnownParts(id);
    }
  }
  downloadOverlay.hidden = false;
  downloadOverlay.setAttribute("aria-hidden", "false");
  renderDownloadParts(id);
}

function closeDownloadDetails(): void {
  currentDownloadModalId = null;
  downloadOverlay.hidden = true;
  downloadOverlay.setAttribute("aria-hidden", "true");
}

downloadOverlay.querySelectorAll<HTMLElement>("[data-download-dismiss]").forEach((el) => {
  el.addEventListener("click", (e) => {
    if (el.tagName === "A") e.preventDefault();
    closeDownloadDetails();
  });
});

document.addEventListener("keydown", (e) => {
  if (!downloadOverlay.hidden && e.key === "Escape") closeDownloadDetails();
});

document.querySelectorAll<HTMLAnchorElement>("[data-download-details-id]").forEach((a) => {
  a.addEventListener("click", (e) => {
    e.preventDefault();
    const id = a.dataset.downloadDetailsId;
    if (id) openDownloadDetails(id);
  });
});

// --- download flow ----------------------------------------------------

const downloadingIds = new Set<string>();

async function downloadModel(link: HTMLAnchorElement, id: string) {
  if (downloadingIds.has(id)) return;
  const def = MODELS.find((m) => m.id === id);
  if (!def) return;
  downloadingIds.add(id);
  const original = link.textContent ?? "download";
  link.classList.add("is-disabled");
  const badge = document.querySelector<HTMLElement>(`[data-cache-id="${id}"]`);
  const urls = modelCacheUrls(def);
  // A real download is the source of truth for parts; replace any
  // placeholder seed so we don't leave stale "queued" rows behind.
  partsByModel.delete(id);
  partsScanned.delete(id);
  if (currentDownloadModalId === id) renderDownloadParts(id);
  const setLinkText = (text: string) => {
    link.textContent = text;
  };
  setLinkText("downloading…");
  try {
    // Drive the link text from the same per-part state the modal uses.
    // Until the GGUF manifest resolves and shards are announced, the
    // tokenizer + manifest totals would round to a misleading near-100%
    // (the old "67% jump" bug) — so suppress the percent in that window.
    const recompute = () => {
      const parts = partsByModel.get(id);
      if (!parts || parts.size === 0) {
        setLinkText("downloading…");
        return;
      }
      let loaded = 0;
      let total = 0;
      let unknown = false;
      let hasShard = false;
      for (const p of parts.values()) {
        loaded += p.loaded;
        if (p.total == null) unknown = true;
        else total += p.total;
        if (p.parent && p.name !== "manifest.json") hasShard = true;
      }
      if (hasShard && !unknown && total > 0) {
        const pct = Math.min(100, (loaded / total) * 100);
        setLinkText(`downloading ${pct.toFixed(0)}%`);
      } else {
        setLinkText("downloading…");
      }
    };
    await Promise.all(
      urls.map((url) =>
        fetchCached(url, undefined, (e) => {
          recordPart(id, e);
          recompute();
        }),
      ),
    );
    if (badge) {
      badge.textContent = "cached";
      badge.dataset.state = "cached";
    }
    setLinkText("cached");
    // Now that bytes are durably in the Cache API, normalize visualizer
    // state so reopening the modal shows everything as "cached" rather
    // than the transient "done" state from this session's download.
    const finishedParts = partsByModel.get(id);
    if (finishedParts) {
      for (const p of finishedParts.values()) {
        if (p.status === "done") p.status = "cached";
      }
      if (currentDownloadModalId === id) renderDownloadParts(id);
    }
    partsScanned.add(id);
    // If the model that just finished downloading is still the selected
    // one and hasn't been loaded into the GPU yet, kick off the load now
    // rather than waiting for the first send.
    if (id === selectedId && (!loaded || loaded.def.id !== selectedId) && !busy) {
      void handleLoad();
    }
  } catch (err) {
    console.error(`download failed for ${id}`, err);
    setLinkText("download failed");
    setTimeout(() => setLinkText(original), 2500);
  } finally {
    link.classList.remove("is-disabled");
    downloadingIds.delete(id);
    refreshCacheStatus();
  }
}

document.querySelectorAll<HTMLAnchorElement>("[data-download-id]").forEach((a) => {
  a.addEventListener("click", (e) => {
    e.preventDefault();
    const id = a.dataset.downloadId;
    if (id) downloadModel(a, id);
  });
});

const navWithGpu = navigator as unknown as { gpu?: { requestAdapter(): Promise<unknown> } };
if (navWithGpu.gpu) {
  navWithGpu.gpu
    .requestAdapter()
    .then((adapter: unknown) => {
      epLabel.textContent = `GPU: ${adapter ? "webgpu available" : "no adapter"}`;
    })
    .catch(() => {
      epLabel.textContent = "GPU: probe failed";
    });
} else {
  epLabel.textContent = "GPU: unavailable";
}

updateSelectedUi();
updateChatInputState();
loadConversationForSelected();
refreshCacheStatus();
