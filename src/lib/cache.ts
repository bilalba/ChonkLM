// Browser-side weight loader.
//
// Models are sharded into ≤24 MiB byte-chunks (see web/scripts/shard-gguf.mjs).
// At runtime we fetch shards individually, cap concurrency at MAX_INFLIGHT, and
// reassemble logical (offset, length) reads. Each shard is cached in the
// browser Cache API under its own URL so repeat sessions skip the network.
//
// Public surface — kept narrow on purpose so runtimes don't need to know
// anything about sharding:
//
//   fetchCached(url, onProgress)              → ArrayBuffer
//     Whole-file fetch via Cache API. Used for tokenizer.json and friends.
//
//   fetchCachedRange(url, offset, length, …)  → ArrayBuffer
//     Logical byte-range read. `url` may be either:
//       a) a sharded weights path like ".../gguf/q4_k_m" (no extension),
//          in which case we resolve manifest.json + shards under that dir.
//       b) any other URL — we fall back to a single HTTP Range request,
//          cached as one entry under `${url}?__range=offset-end`.
//     Mode (b) keeps backward compat for non-sharded assets.
//
//   inspectCachedUrls(urls)                   → CacheInspection
//     Used by the dashboard to show "fully cached" badges. For sharded
//     entries we report cached when the manifest *and* every shard are
//     present in the cache.

const CACHE_NAME = "chonklm-models-v1";

// Cap concurrent shard downloads. AGENTS-side requirement: too much parallel
// fetching slows everything down (HTTP/2 head-of-line blocking, decoder
// thrash). 4 is the sweet spot we settled on.
const MAX_INFLIGHT = 4;

export interface ProgressEvent {
  url: string;
  /** Bytes received so far for the current shard or whole-file fetch. */
  loaded: number;
  /** Total bytes for the current shard or whole-file fetch, if known. */
  total: number | null;
  /** True the first time this URL is observed (i.e. cache miss). */
  fromNetwork: boolean;
  /** Optional runtime-provided stage text, used by custom WebGPU loaders. */
  status?: string;
}

export type ProgressCallback = (e: ProgressEvent) => void;

/**
 * Per-part progress event. Fired by `fetchCached` and `prefetchSharded`
 * for every atomic file they touch (tokenizer/whole-file URL, the GGUF
 * manifest, and each individual shard). Lets the dashboard render a
 * row-per-file download visualizer without losing per-shard granularity
 * the way the aggregate `ProgressCallback` does.
 */
export interface PartEvent {
  /** Stable per-file URL. Use this as the visualizer row key. */
  url: string;
  /** Last path segment of `url`, suitable for display. */
  name: string;
  loaded: number;
  total: number | null;
  status: "queued" | "downloading" | "cached" | "done" | "error";
  fromNetwork: boolean;
  /**
   * Set on shard + manifest events to the parent shard-directory URL,
   * so the visualizer can group them under the right model entry.
   */
  parent?: string;
}

export type PartCallback = (e: PartEvent) => void;

export interface LoadStepEvent {
  step: string;
  detail?: string;
  /**
   * Optional discrete progress for this phase. Used by the dashboard to
   * derive a meaningful percent during cached loads, where the byte-based
   * progress is uninformative (each cache hit lands as `loaded == total`,
   * so the running fraction sits at 100% from the first event).
   */
  progress?: { current: number; total: number };
}

export type LoadStepCallback = (e: LoadStepEvent) => void;

function urlTail(url: string): string {
  const path = url.split("?")[0].split("#")[0];
  const lastSlash = path.lastIndexOf("/");
  return lastSlash >= 0 ? path.slice(lastSlash + 1) : path;
}

export interface CacheInspection {
  supported: boolean;
  cached: string[];
  missing: string[];
  total: number;
}

interface ShardManifest {
  version: number;
  quant?: string;
  totalSize: number;
  shardSize: number;
  shardCount: number;
  shards: ShardEntry[];
}

interface ShardEntry {
  index: number;
  name: string;
  offset: number;
  size: number;
}

export async function requestPersistence(): Promise<boolean> {
  if (typeof navigator === "undefined" || !navigator.storage?.persist) return false;
  try {
    return await navigator.storage.persist();
  } catch {
    return false;
  }
}

export async function storageEstimate(): Promise<{ usage?: number; quota?: number } | null> {
  if (typeof navigator === "undefined" || !navigator.storage?.estimate) return null;
  try {
    return await navigator.storage.estimate();
  } catch {
    return null;
  }
}

async function openCache(): Promise<Cache | null> {
  if (typeof caches === "undefined") return null;
  return caches.open(CACHE_NAME);
}

export async function inspectCachedUrls(urls: string[]): Promise<CacheInspection> {
  const uniqueUrls = Array.from(new Set(urls));
  const cache = await openCache();
  if (!cache) {
    return { supported: false, cached: [], missing: uniqueUrls, total: uniqueUrls.length };
  }

  const cached: string[] = [];
  const missing: string[] = [];
  await Promise.all(
    uniqueUrls.map(async (url) => {
      const ok = isShardedUrl(url)
        ? await isShardedUrlFullyCached(cache, url)
        : !!(await cache.match(url));
      (ok ? cached : missing).push(url);
    }),
  );

  return { supported: true, cached, missing, total: uniqueUrls.length };
}

/**
 * Cache-warm a URL. Used by both the tokenizer-loader path and the
 * dashboard's "download" button.
 *
 * For sharded URLs (e.g. ".../gguf/q4_k_m") we fetch every shard and
 * cache it under its own URL, but we don't materialize a single
 * concatenated ArrayBuffer — that could be ~hundreds of MiB of useless
 * memory pressure on low-end devices. Callers that need actual bytes
 * out of a sharded file should use `fetchCachedRange`.
 *
 * Progress events are aggregated across shards: `loaded` is cumulative
 * bytes received, `total` is the manifest's `totalSize`.
 */
export async function fetchCached(
  url: string,
  onProgress?: ProgressCallback,
  onPart?: PartCallback,
): Promise<ArrayBuffer> {
  if (isShardedUrl(url)) {
    await prefetchSharded(url, onProgress, onPart);
    return new ArrayBuffer(0);
  }
  const name = urlTail(url);
  onPart?.({ url, name, loaded: 0, total: null, status: "queued", fromNetwork: false });
  const cache = await openCache();
  if (cache) {
    const hit = await cache.match(url);
    if (hit) {
      const buf = await hit.arrayBuffer();
      onProgress?.({ url, loaded: buf.byteLength, total: buf.byteLength, fromNetwork: false });
      onPart?.({
        url,
        name,
        loaded: buf.byteLength,
        total: buf.byteLength,
        status: "cached",
        fromNetwork: false,
      });
      return buf;
    }
  }

  let res: Response;
  try {
    res = await fetch(url, { credentials: "same-origin" });
  } catch (e) {
    onPart?.({ url, name, loaded: 0, total: null, status: "error", fromNetwork: true });
    throw e;
  }
  if (!res.ok) {
    onPart?.({ url, name, loaded: 0, total: null, status: "error", fromNetwork: true });
    throw new Error(`fetch ${url}: HTTP ${res.status}`);
  }
  const total = Number(res.headers.get("content-length")) || null;

  let toCache: Response;
  let buf: ArrayBuffer;

  if (res.body && (onProgress || onPart)) {
    const [a, b] = res.body.tee();
    toCache = new Response(a, { headers: res.headers });
    buf = await readWithProgress(b, url, total, (e) => {
      onProgress?.(e);
      onPart?.({
        url,
        name,
        loaded: e.loaded,
        total: e.total,
        status: "downloading",
        fromNetwork: true,
      });
    });
  } else {
    buf = await res.arrayBuffer();
    toCache = new Response(buf, { headers: res.headers });
    onProgress?.({ url, loaded: buf.byteLength, total: buf.byteLength, fromNetwork: true });
  }
  onPart?.({
    url,
    name,
    loaded: buf.byteLength,
    total: buf.byteLength,
    status: "done",
    fromNetwork: true,
  });

  if (cache) {
    try {
      await cache.put(url, toCache);
    } catch (e) {
      console.warn(`cache.put(${url}) failed:`, e);
    }
  }

  return buf;
}

/**
 * Logical byte-range read. Routes through shards when `url` looks like
 * a sharded directory path, otherwise issues a single HTTP Range request.
 */
export async function fetchCachedRange(
  url: string,
  offset: number,
  length: number,
  onProgress?: ProgressCallback,
): Promise<ArrayBuffer> {
  if (isShardedUrl(url)) {
    return fetchShardedRange(url, offset, length, onProgress);
  }
  return fetchHttpRange(url, offset, length, onProgress);
}

// --- sharded path ---------------------------------------------------------

// We treat any URL that does not end in a recognized file extension as a
// shard directory. In practice the registry now hands us ".../gguf/q4_k_m"
// (no extension) for weights, and ".../raw/tokenizer.json" (extension) for
// metadata, so this is unambiguous.
function isShardedUrl(url: string): boolean {
  const path = url.split("?")[0].split("#")[0];
  const lastSlash = path.lastIndexOf("/");
  const tail = lastSlash >= 0 ? path.slice(lastSlash + 1) : path;
  if (!tail) return false;
  return !tail.includes(".");
}

function manifestUrlFor(shardDirUrl: string): string {
  return `${shardDirUrl.replace(/\/+$/, "")}/manifest.json`;
}

function shardUrlFor(shardDirUrl: string, shard: ShardEntry): string {
  return `${shardDirUrl.replace(/\/+$/, "")}/${shard.name}`;
}

const manifestPromises = new Map<string, Promise<ShardManifest>>();

async function getManifest(shardDirUrl: string): Promise<ShardManifest> {
  let p = manifestPromises.get(shardDirUrl);
  if (!p) {
    p = (async () => {
      const buf = await fetchCached(manifestUrlFor(shardDirUrl));
      const text = new TextDecoder().decode(buf);
      const m = JSON.parse(text) as ShardManifest;
      if (!m || typeof m.totalSize !== "number" || !Array.isArray(m.shards)) {
        throw new Error(`invalid manifest at ${manifestUrlFor(shardDirUrl)}`);
      }
      return m;
    })();
    manifestPromises.set(shardDirUrl, p);
    p.catch(() => manifestPromises.delete(shardDirUrl));
  }
  return p;
}

async function isShardedUrlFullyCached(cache: Cache, shardDirUrl: string): Promise<boolean> {
  const manifestHit = await cache.match(manifestUrlFor(shardDirUrl));
  if (!manifestHit) return false;
  let manifest: ShardManifest;
  try {
    manifest = JSON.parse(await manifestHit.text()) as ShardManifest;
  } catch {
    return false;
  }
  for (const s of manifest.shards) {
    const hit = await cache.match(shardUrlFor(shardDirUrl, s));
    if (!hit) return false;
  }
  return true;
}

async function prefetchSharded(
  shardDirUrl: string,
  onProgress?: ProgressCallback,
  onPart?: PartCallback,
): Promise<void> {
  const manifestUrl = manifestUrlFor(shardDirUrl);
  if (onPart) {
    onPart({
      url: manifestUrl,
      name: "manifest.json",
      loaded: 0,
      total: null,
      status: "queued",
      fromNetwork: false,
      parent: shardDirUrl,
    });
  }
  let manifest: ShardManifest;
  try {
    manifest = await getManifest(shardDirUrl);
  } catch (e) {
    onPart?.({
      url: manifestUrl,
      name: "manifest.json",
      loaded: 0,
      total: null,
      status: "error",
      fromNetwork: true,
      parent: shardDirUrl,
    });
    throw e;
  }
  if (onPart) {
    // Manifest is tiny (a few KB) and may have come from cache or the network;
    // we don't track its bytes individually. Mark it done so the visualizer
    // can render a finished row.
    onPart({
      url: manifestUrl,
      name: "manifest.json",
      loaded: 1,
      total: 1,
      status: "done",
      fromNetwork: false,
      parent: shardDirUrl,
    });
    // Announce all shards upfront so the visualizer renders placeholders
    // before any bytes flow.
    for (const s of manifest.shards) {
      onPart({
        url: shardUrlFor(shardDirUrl, s),
        name: s.name,
        loaded: 0,
        total: s.size,
        status: "queued",
        fromNetwork: false,
        parent: shardDirUrl,
      });
    }
  }

  const perShardLoaded = new Array<number>(manifest.shards.length).fill(0);

  const reportAggregate = (fromNetwork: boolean) => {
    if (!onProgress) return;
    let loaded = 0;
    for (const v of perShardLoaded) loaded += v;
    onProgress({ url: shardDirUrl, loaded, total: manifest.totalSize, fromNetwork });
  };

  await Promise.all(
    manifest.shards.map((s, i) =>
      fetchShardCached(
        shardUrlFor(shardDirUrl, s),
        s.size,
        (e) => {
          perShardLoaded[i] = e.loaded;
          reportAggregate(e.fromNetwork);
        },
        onPart,
        shardDirUrl,
      ),
    ),
  );
  // Cached-only short-circuit shards skip onProgress in fetchShardCached's
  // first branch — emit a final aggregate so the dashboard sees 100%.
  reportAggregate(false);
}

async function fetchShardedRange(
  shardDirUrl: string,
  offset: number,
  length: number,
  onProgress?: ProgressCallback,
): Promise<ArrayBuffer> {
  const manifest = await getManifest(shardDirUrl);
  if (offset < 0 || length < 0 || offset + length > manifest.totalSize) {
    throw new Error(
      `range out of bounds for ${shardDirUrl}: offset=${offset} length=${length} totalSize=${manifest.totalSize}`,
    );
  }

  const out = new Uint8Array(length);
  const end = offset + length;

  // Find the shard slice that intersects [offset, end). Manifest shards are
  // contiguous and ordered; binary search would help if we ever had hundreds
  // of shards, but linear scan is fine at <100 shards.
  const intersecting: ShardEntry[] = [];
  for (const s of manifest.shards) {
    const sEnd = s.offset + s.size;
    if (sEnd <= offset) continue;
    if (s.offset >= end) break;
    intersecting.push(s);
  }

  // Fetch all required shards through the global semaphore. Each shard is
  // cached under its own URL, so subsequent reads that touch the same shard
  // reuse the cache entry instead of re-fetching.
  const buffers = await Promise.all(
    intersecting.map((s) => fetchShardCached(shardUrlFor(shardDirUrl, s), s.size, onProgress)),
  );

  // Slice & paste each shard's overlapping window into the output buffer.
  for (let i = 0; i < intersecting.length; i++) {
    const s = intersecting[i];
    const buf = buffers[i];
    const sliceStart = Math.max(0, offset - s.offset);
    const sliceEnd = Math.min(s.size, end - s.offset);
    const dstOffset = s.offset + sliceStart - offset;
    out.set(new Uint8Array(buf, sliceStart, sliceEnd - sliceStart), dstOffset);
  }

  return out.buffer;
}

// --- shard fetch with concurrency limit + in-flight dedup -----------------

const inflight = new Map<string, Promise<ArrayBuffer>>();
let activeFetches = 0;
const waiters: Array<() => void> = [];

function acquire(): Promise<void> {
  if (activeFetches < MAX_INFLIGHT) {
    activeFetches += 1;
    return Promise.resolve();
  }
  return new Promise<void>((resolve) => {
    waiters.push(() => {
      activeFetches += 1;
      resolve();
    });
  });
}

function release(): void {
  activeFetches -= 1;
  const next = waiters.shift();
  if (next) next();
}

async function fetchShardCached(
  url: string,
  expectedSize: number,
  onProgress?: ProgressCallback,
  onPart?: PartCallback,
  parent?: string,
): Promise<ArrayBuffer> {
  const name = urlTail(url);
  // Cache API short-circuits the semaphore — already-cached shards are free,
  // we should never queue them behind an in-flight network shard.
  const cache = await openCache();
  if (cache) {
    const hit = await cache.match(url);
    if (hit) {
      const buf = await hit.arrayBuffer();
      onProgress?.({ url, loaded: buf.byteLength, total: buf.byteLength, fromNetwork: false });
      onPart?.({
        url,
        name,
        loaded: buf.byteLength,
        total: buf.byteLength,
        status: "cached",
        fromNetwork: false,
        parent,
      });
      return buf;
    }
  }

  // Dedup concurrent callers asking for the same shard. The promise stays in
  // the map until it settles — long enough to absorb bursty parallel reads
  // from a runtime that touches several tensors at once.
  let p = inflight.get(url);
  if (p) return p;

  p = (async () => {
    await acquire();
    try {
      let res: Response;
      try {
        res = await fetch(url, { credentials: "same-origin" });
      } catch (e) {
        onPart?.({ url, name, loaded: 0, total: expectedSize, status: "error", fromNetwork: true, parent });
        throw e;
      }
      if (!res.ok) {
        onPart?.({ url, name, loaded: 0, total: expectedSize, status: "error", fromNetwork: true, parent });
        throw new Error(`fetch ${url}: HTTP ${res.status}`);
      }

      let toCache: Response;
      let buf: ArrayBuffer;
      if (res.body && (onProgress || onPart)) {
        const [a, b] = res.body.tee();
        toCache = new Response(a, { headers: res.headers });
        buf = await readWithProgress(b, url, expectedSize, (e) => {
          onProgress?.(e);
          onPart?.({
            url,
            name,
            loaded: e.loaded,
            total: e.total,
            status: "downloading",
            fromNetwork: true,
            parent,
          });
        });
      } else {
        buf = await res.arrayBuffer();
        toCache = new Response(buf, { headers: res.headers });
        onProgress?.({ url, loaded: buf.byteLength, total: expectedSize, fromNetwork: true });
      }

      if (buf.byteLength !== expectedSize) {
        onPart?.({ url, name, loaded: buf.byteLength, total: expectedSize, status: "error", fromNetwork: true, parent });
        throw new Error(
          `shard ${url}: expected ${expectedSize} bytes, got ${buf.byteLength}`,
        );
      }

      onPart?.({
        url,
        name,
        loaded: buf.byteLength,
        total: expectedSize,
        status: "done",
        fromNetwork: true,
        parent,
      });

      if (cache) {
        try {
          await cache.put(url, toCache);
        } catch (e) {
          console.warn(`cache.put(${url}) failed:`, e);
        }
      }
      return buf;
    } finally {
      release();
      inflight.delete(url);
    }
  })();
  inflight.set(url, p);
  return p;
}

// --- legacy single-Range path (non-sharded URLs) --------------------------

async function fetchHttpRange(
  url: string,
  offset: number,
  length: number,
  onProgress?: ProgressCallback,
): Promise<ArrayBuffer> {
  const end = offset + length - 1;
  const cacheUrl = `${url}${url.includes("?") ? "&" : "?"}__range=${offset}-${end}`;
  const cache = await openCache();
  if (cache) {
    const hit = await cache.match(cacheUrl);
    if (hit) {
      const buf = await hit.arrayBuffer();
      onProgress?.({ url: cacheUrl, loaded: buf.byteLength, total: buf.byteLength, fromNetwork: false });
      return buf;
    }
  }

  const res = await fetch(url, {
    credentials: "same-origin",
    headers: { Range: `bytes=${offset}-${end}` },
  });
  if (!res.ok) throw new Error(`range fetch ${url} ${offset}-${end}: HTTP ${res.status}`);
  let buf = await res.arrayBuffer();
  if (res.status === 206) {
    if (buf.byteLength !== length) {
      throw new Error(`range fetch ${url} ${offset}-${end}: expected ${length} bytes, got ${buf.byteLength}`);
    }
  } else {
    buf = buf.slice(offset, offset + length);
  }
  onProgress?.({ url: cacheUrl, loaded: buf.byteLength, total: length, fromNetwork: true });

  if (cache) {
    try {
      await cache.put(cacheUrl, new Response(buf));
    } catch (e) {
      console.warn(`cache.put(${cacheUrl}) failed:`, e);
    }
  }

  return buf;
}

async function readWithProgress(
  stream: ReadableStream<Uint8Array>,
  url: string,
  total: number | null,
  onProgress: ProgressCallback,
): Promise<ArrayBuffer> {
  const reader = stream.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress({ url, loaded, total, fromNetwork: true });
  }
  const merged = new Uint8Array(loaded);
  let offset = 0;
  for (const c of chunks) {
    merged.set(c, offset);
    offset += c.byteLength;
  }
  return merged.buffer;
}

export async function clearCache(): Promise<void> {
  if (typeof caches === "undefined") return;
  await caches.delete(CACHE_NAME);
}

// Bounded Promise.all. Used by runtimes to fan out per-layer weight uploads
// without spawning hundreds of in-flight tensor fetches at once on tall models
// (e.g. baguettotron with 80 layers).
export async function mapWithConcurrency<T, R>(
  items: readonly T[],
  concurrency: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(items.length);
  let cursor = 0;
  const workerCount = Math.max(1, Math.min(concurrency, items.length));
  const workers = Array.from({ length: workerCount }, async () => {
    while (true) {
      const i = cursor++;
      if (i >= items.length) return;
      results[i] = await fn(items[i], i);
    }
  });
  await Promise.all(workers);
  return results;
}
