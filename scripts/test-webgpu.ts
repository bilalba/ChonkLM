// Programmatic WebGPU test harness. Boots Chromium with WebGPU enabled,
// drives /test/ via window.runInference, returns the same {ep, text,
// tokensPerSec} shape the manual playground would. Use this to A/B model
// variants without staring at a browser.
//
// HEADED BY DEFAULT. Headless Chromium on macOS falls back to a software
// adapter (SwiftShader) which doesn't expose the `shader-f16` WebGPU
// feature. Headed mode opens a real window for the run duration, giving us
// the Metal adapter with f16 support. Pass --headless to opt back into the
// software-adapter mode (useful for CI on Linux + xvfb, or for testing
// fp32-only paths quickly).
//
// Assumes the dev server is already running on http://localhost:4321/.
// (Astro's static handlers for /models and the /test page only exist in dev
// mode.) Start it with `npm run dev` in another terminal.
//
// Usage:
//   npm run test:webgpu -- <model-id> [prompt] [max-tokens]
//   npm run test:webgpu -- --list
//   npm run test:webgpu -- --matrix       # every chat-capable model
//   npm run test:webgpu -- --probe-prefix <model-id>
//   npm run test:webgpu -- --compare-lfm [prompt] [max-tokens]
//   npm run test:webgpu -- --compare-pleias [prompt] [max-tokens]
//   npm run test:webgpu -- --headless <id>     # software adapter, no window
//   npm run test:webgpu -- --clear-cache <id>  # wipe Cache API before run
//   npm run test:webgpu -- --temperature 0.8 --top-p 0.95 --top-k 50 --repetition-penalty 1.1 <id>

import { chromium, type Browser, type ConsoleMessage, type Page } from "playwright";
import { MODELS, getModel, type ModelDef } from "../src/lib/registry.ts";

declare const process: {
  env: Record<string, string | undefined>;
  argv: string[];
  exitCode?: number;
  exit(code?: number): never;
};

interface Result {
  ep: "webgpu";
  text: string;
  tokensPerSec: number;
  promptTokens: number;
  generatedTokens: number;
  elapsedSec: number;
}

interface RunOptions {
  modelId: string;
  prompt: string;
  maxTokens: number;
  headed: boolean;
  clearCache: boolean;
  generation: {
    temperature?: number;
    topP?: number;
    topK?: number;
    repetitionPenalty?: number;
  };
}

interface PrefixProbeResult {
  before: number;
  after: number;
  extendedPromptTokens: number;
  suffixTokens: number;
  advanced: number;
  reused: boolean;
}

declare global {
  interface Window {
    runInference: (
      modelId: string,
      prompt: string,
      max?: number,
      generation?: RunOptions["generation"],
    ) => Promise<Result>;
    probeWgslPrefixCache: (modelId: string) => Promise<PrefixProbeResult>;
    clearChonkCache: () => Promise<void>;
    chonklmReady: Promise<void>;
  }
}

const TEST_URL = process.env.CHONKLM_TEST_URL ?? "http://localhost:4321/test/";

async function bootBrowser(headed: boolean): Promise<Browser> {
  // Playwright's bundled Chromium ships a Dawn build that can produce
  // different numerical results on some GPUs than stable Chrome (we hit
  // this on a 16" MBP with discrete AMD Radeon Pro). Prefer the system
  // Chrome installation — same engine real users run — and fall back to
  // the bundled Chromium when Chrome isn't present.
  //
  // Pass CHONKLM_BROWSER=chromium to force the bundled build for repro.
  const channel = process.env.CHONKLM_BROWSER === "chromium" ? undefined : "chrome";
  try {
    return await chromium.launch({
      channel,
      headless: !headed,
      args: ["--enable-unsafe-webgpu"],
    });
  } catch (e) {
    if (channel) {
      console.warn(`(falling back to bundled Chromium: ${(e as Error).message})`);
      return chromium.launch({
        headless: !headed,
        args: ["--enable-unsafe-webgpu"],
      });
    }
    throw e;
  }
}

async function diagnoseGpu(page: Page): Promise<{ adapter: string | null }> {
  return page.evaluate(async () => {
    if (!("gpu" in navigator)) return { adapter: null };
    try {
      const adapter = await (navigator as Navigator & {
        gpu: { requestAdapter(): Promise<unknown> };
      }).gpu.requestAdapter();
      if (!adapter) return { adapter: null };
      const info = (adapter as { info?: { vendor?: string; architecture?: string } }).info;
      return { adapter: info ? `${info.vendor ?? "?"} / ${info.architecture ?? "?"}` : "unknown" };
    } catch (e) {
      return { adapter: `error: ${(e as Error).message}` };
    }
  });
}

async function runOne(opts: RunOptions): Promise<{
  result: Result | null;
  gpuAdapter: string | null;
  consoleErrors: string[];
  pageErrors: string[];
}> {
  const browser = await bootBrowser(opts.headed);
  const consoleErrors: string[] = [];
  const pageErrors: string[] = [];

  try {
    const ctx = await browser.newContext();
    const page = await ctx.newPage();

    page.on("console", (msg: ConsoleMessage) => {
      const text = `${msg.type()}: ${msg.text()}`;
      if (msg.type() === "error" || msg.type() === "warning") consoleErrors.push(text);
      // Forward our own debug traces verbatim — useful for tok-id traces.
      if (msg.text().startsWith("[chonklm]")) console.log(`  ${msg.text()}`);
    });
    page.on("pageerror", (err) => pageErrors.push(err.message));

    await page.goto(TEST_URL, { waitUntil: "domcontentloaded" });
    await page.waitForFunction(() => window.chonklmReady !== undefined && typeof window.runInference === "function", {
      timeout: 30_000,
    });
    await page.evaluate(() => window.chonklmReady);

    const gpuAdapter = (await diagnoseGpu(page)).adapter;

    if (opts.clearCache) {
      await page.evaluate(() => window.clearChonkCache());
      console.log(`# cache: cleared`);
    }

    const result = (await page.evaluate(
      ([id, prompt, max, generation]) => window.runInference(id, prompt, max, generation),
      [opts.modelId, opts.prompt, opts.maxTokens, opts.generation] as [
        string,
        string,
        number,
        RunOptions["generation"],
      ],
    )) as Result;

    return { result, gpuAdapter, consoleErrors, pageErrors };
  } catch (e) {
    return {
      result: null,
      gpuAdapter: null,
      consoleErrors,
      pageErrors: [...pageErrors, (e as Error).message],
    };
  } finally {
    await browser.close();
  }
}

async function runPrefixProbe(opts: {
  modelId: string;
  headed: boolean;
  clearCache: boolean;
}): Promise<{
  result: PrefixProbeResult | null;
  gpuAdapter: string | null;
  consoleErrors: string[];
  pageErrors: string[];
}> {
  const browser = await bootBrowser(opts.headed);
  const consoleErrors: string[] = [];
  const pageErrors: string[] = [];

  try {
    const ctx = await browser.newContext();
    const page = await ctx.newPage();

    page.on("console", (msg: ConsoleMessage) => {
      const text = `${msg.type()}: ${msg.text()}`;
      if (msg.type() === "error" || msg.type() === "warning") consoleErrors.push(text);
      if (msg.text().startsWith("[chonklm]")) console.log(`  ${msg.text()}`);
    });
    page.on("pageerror", (err) => pageErrors.push(err.message));

    await page.goto(TEST_URL, { waitUntil: "domcontentloaded" });
    await page.waitForFunction(() => window.chonklmReady !== undefined && typeof window.probeWgslPrefixCache === "function", {
      timeout: 30_000,
    });
    await page.evaluate(() => window.chonklmReady);

    const gpuAdapter = (await diagnoseGpu(page)).adapter;
    if (opts.clearCache) {
      await page.evaluate(() => window.clearChonkCache());
      console.log(`# cache: cleared`);
    }

    const result = await page.evaluate((id) => window.probeWgslPrefixCache(id), opts.modelId);
    return { result, gpuAdapter, consoleErrors, pageErrors };
  } catch (e) {
    return {
      result: null,
      gpuAdapter: null,
      consoleErrors,
      pageErrors: [...pageErrors, (e as Error).message],
    };
  } finally {
    await browser.close();
  }
}

function numberFlag(argv: string[], name: string): number | undefined {
  const prefix = `${name}=`;
  const inline = argv.find((a) => a.startsWith(prefix));
  if (inline) return Number(inline.slice(prefix.length));
  const i = argv.indexOf(name);
  if (i >= 0 && argv[i + 1] && !argv[i + 1].startsWith("--")) return Number(argv[i + 1]);
  return undefined;
}

function summarise(model: ModelDef, run: Awaited<ReturnType<typeof runOne>>): void {
  const want = model.preferredEp ?? "webgpu";
  const r = run.result;
  console.log(`# adapter: ${run.gpuAdapter ?? "none"}`);
  if (!r) {
    console.log(`# FAIL — no result`);
  } else {
    console.log(`# ep: ${r.ep} (wanted ${want})`);
    console.log(`# tokens: ${r.generatedTokens} · ${r.tokensPerSec.toFixed(1)} tok/s`);
    console.log(`# output:\n${r.text}`);
  }
  if (run.consoleErrors.length) {
    console.log(`# console errors (${run.consoleErrors.length}):`);
    for (const e of run.consoleErrors.slice(0, 10)) console.log(`  ${e}`);
  }
  if (run.pageErrors.length) {
    console.log(`# page errors (${run.pageErrors.length}):`);
    for (const e of run.pageErrors.slice(0, 5)) console.log(`  ${e}`);
  }
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);

  if (argv.includes("--list") || argv.includes("-l")) {
    console.log("registered models:");
    for (const m of MODELS) console.log(`  ${m.id.padEnd(40)} ${m.pitch}`);
    return;
  }

  // Default headed; --headless opts into the software-adapter path.
  const headed = !argv.includes("--headless");
  const clearCache = argv.includes("--clear-cache");
  const generation = {
    temperature: numberFlag(argv, "--temperature"),
    topP: numberFlag(argv, "--top-p"),
    topK: numberFlag(argv, "--top-k"),
    repetitionPenalty: numberFlag(argv, "--repetition-penalty"),
  };
  const skipNext = new Set<number>();
  for (const flag of ["--temperature", "--top-p", "--top-k", "--repetition-penalty"]) {
    const i = argv.indexOf(flag);
    if (i >= 0) skipNext.add(i + 1);
  }
  const probePrefixIndex = argv.indexOf("--probe-prefix");
  if (probePrefixIndex >= 0) skipNext.add(probePrefixIndex + 1);
  const positional = argv.filter((a, i) => !a.startsWith("--") && !skipNext.has(i));

  if (probePrefixIndex >= 0) {
    const id = argv[probePrefixIndex + 1] && !argv[probePrefixIndex + 1].startsWith("--")
      ? argv[probePrefixIndex + 1]
      : positional[0];
    if (!id) throw new Error("--probe-prefix requires a model id");
    const model = getModel(id);
    console.log(`# model: ${model.id}`);
    console.log(`# prefix cache probe\n`);
    const run = await runPrefixProbe({ modelId: id, headed, clearCache });
    console.log(`# adapter: ${run.gpuAdapter ?? "none"}`);
    if (!run.result) {
      console.log(`# FAIL — no result`);
      process.exitCode = 1;
    } else {
      console.log(`# before: ${run.result.before}`);
      console.log(`# after: ${run.result.after}`);
      console.log(`# extended prompt tokens: ${run.result.extendedPromptTokens}`);
      console.log(`# suffix tokens: ${run.result.suffixTokens}`);
      console.log(`# advanced: ${run.result.advanced}`);
      console.log(`# reused: ${run.result.reused}`);
      if (!run.result.reused) process.exitCode = 1;
    }
    if (run.consoleErrors.length) {
      console.log(`# console errors (${run.consoleErrors.length}):`);
      for (const e of run.consoleErrors.slice(0, 10)) console.log(`  ${e}`);
    }
    if (run.pageErrors.length) {
      console.log(`# page errors (${run.pageErrors.length}):`);
      for (const e of run.pageErrors.slice(0, 5)) console.log(`  ${e}`);
      process.exitCode = 1;
    }
    return;
  }

  if (argv.includes("--matrix")) {
    const targets = MODELS.filter((m) => m.chat);
    const prompt = positional[0] ?? "List three primary colors.";
    const max = Number(positional[1] ?? 60);
    for (const m of targets) {
      console.log(`\n=========================`);
      console.log(`# model: ${m.id}`);
      console.log(`# prompt: ${JSON.stringify(prompt)}`);
      const run = await runOne({ modelId: m.id, prompt, maxTokens: max, headed, clearCache, generation });
      summarise(m, run);
    }
    return;
  }

  if (argv.includes("--compare-lfm")) {
    const prompt = positional[0] ?? "Extract name, city, and invoice total from: Ada Lovelace, London, total $42.15.";
    const max = Number(positional[1] ?? 40);
    const targets = ["lfm2_5-350m-q4-k-m-gguf-wgsl", "lfm2_5-350m-q8-gguf-wgsl"];
    const results: Array<{ model: ModelDef; run: Awaited<ReturnType<typeof runOne>> }> = [];
    for (const id of targets) {
      const model = getModel(id);
      console.log(`\n=========================`);
      console.log(`# model: ${model.id}`);
      console.log(`# prompt: ${JSON.stringify(prompt)}`);
      const run = await runOne({ modelId: id, prompt, maxTokens: max, headed, clearCache, generation });
      summarise(model, run);
      results.push({ model, run });
    }
    const [q4, q8] = results;
    if (q4?.run.result && q8?.run.result) {
      const a = q4.run.result.tokensPerSec;
      const b = q8.run.result.tokensPerSec;
      console.log(`\n# comparison: Q8 ${b.toFixed(2)} tok/s vs Q4_K_M ${a.toFixed(2)} tok/s (${(b / a).toFixed(2)}x)`);
    } else {
      process.exitCode = 1;
    }
    return;
  }

  if (argv.includes("--compare-pleias")) {
    const prompt = positional[0] ?? "Explique en une phrase pourquoi le ciel est bleu.";
    const max = Number(positional[1] ?? 4);
    const pairs = [
      ["monad-q4-k-m-gguf-wgsl", "monad-q8-gguf-wgsl"],
      ["baguettotron-q4-k-m-gguf-wgsl", "baguettotron-q8-gguf-wgsl"],
    ];
    for (const [ortId, wgslId] of pairs) {
      const results: Array<{ model: ModelDef; run: Awaited<ReturnType<typeof runOne>> }> = [];
      for (const id of [ortId, wgslId]) {
        const model = getModel(id);
        console.log(`\n=========================`);
        console.log(`# model: ${model.id}`);
        console.log(`# prompt: ${JSON.stringify(prompt)}`);
        const run = await runOne({ modelId: id, prompt, maxTokens: max, headed, clearCache, generation });
        summarise(model, run);
        results.push({ model, run });
      }
      const [q4, q8] = results;
      if (q4?.run.result && q8?.run.result) {
        const a = q4.run.result.tokensPerSec;
        const b = q8.run.result.tokensPerSec;
        console.log(`\n# comparison ${q4.model.id}: Q8 ${b.toFixed(2)} tok/s vs Q4_K_M ${a.toFixed(2)} tok/s (${(b / a).toFixed(2)}x)`);
      } else {
        process.exitCode = 1;
      }
    }
    return;
  }

  if (argv.includes("--compare-gemma")) {
    const prompt = positional[0] ?? "List three primary colors.";
    const max = Number(positional[1] ?? 32);
    const targets = [
      "gemma-3-270m-it-q4-k-m-gguf-wgsl",
      "gemma-3-270m-it-q8-gguf-wgsl",
      "gemma-3-270m-it-q8-imported",
    ];
    const results: Array<{ model: ModelDef; run: Awaited<ReturnType<typeof runOne>> }> = [];
    for (const id of targets) {
      const model = getModel(id);
      console.log(`\n=========================`);
      console.log(`# model: ${model.id}`);
      console.log(`# prompt: ${JSON.stringify(prompt)}`);
      const run = await runOne({ modelId: id, prompt, maxTokens: max, headed, clearCache, generation });
      summarise(model, run);
      results.push({ model, run });
    }
    const [q4km, q8, importedQ8] = results;
    if (q4km?.run.result && q8?.run.result && importedQ8?.run.result) {
      const q8Speed = q8.run.result.tokensPerSec;
      const q4kmSpeed = q4km.run.result.tokensPerSec;
      const importedQ8Speed = importedQ8.run.result.tokensPerSec;
      console.log(`# comparison Gemma: Q4_K_M WebGPU ${q4kmSpeed.toFixed(2)} tok/s vs Q8 WebGPU ${q8Speed.toFixed(2)} tok/s (${(q4kmSpeed / q8Speed).toFixed(2)}x)`);
      console.log(`# comparison Gemma: shared Q8 WebGPU ${q8Speed.toFixed(2)} tok/s vs imported Q8 WebGPU ${importedQ8Speed.toFixed(2)} tok/s (${(q8Speed / importedQ8Speed).toFixed(2)}x)`);
    } else {
      process.exitCode = 1;
    }
    return;
  }

  const modelId = positional[0] ?? "smollm2-135m-instruct-q4-k-m-gguf-wgsl";
  const prompt = positional[1] ?? "List three primary colors.";
  const maxTokens = Number(positional[2] ?? 60);

  const def = getModel(modelId);
  console.log(`# model: ${def.id}`);
  console.log(`# prompt: ${JSON.stringify(prompt)}`);
  console.log(`# max tokens: ${maxTokens}\n`);

  const run = await runOne({ modelId, prompt, maxTokens, headed, clearCache, generation });
  summarise(def, run);

  if (!run.result) process.exit(1);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
