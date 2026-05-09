#!/usr/bin/env node
// Split each models/<id>/gguf/<quant>.gguf into ≤24 MiB byte-shards so the
// shards fit under Cloudflare Workers static-asset's 25 MiB per-file cap.
// This lets us ship weights without R2.
//
// Layout produced (next to the source .gguf):
//   models/<id>/gguf/<quant>/manifest.json
//   models/<id>/gguf/<quant>/0000.bin
//   models/<id>/gguf/<quant>/0001.bin
//   …
//
// Concatenating the shards in index order reproduces the source .gguf
// byte-for-byte. The manifest is the only metadata the browser-side
// loader needs.
//
// Usage (from anywhere):
//   node web/scripts/shard-gguf.mjs                # every model, every quant
//   node web/scripts/shard-gguf.mjs lfm2_5-350m    # one model
//   SHARD_SIZE=20971520 node web/scripts/shard-gguf.mjs   # custom shard size
//   FORCE=1 node web/scripts/shard-gguf.mjs        # rebuild even if up-to-date

import { createReadStream } from 'node:fs';
import { mkdir, readdir, rm, stat, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { Buffer } from 'node:buffer';

const ROOT = resolve(import.meta.dirname, '../..');
const MODELS = resolve(ROOT, 'models');

// 24 MiB. Stays under Cloudflare's 25 MiB static-asset cap with margin.
const DEFAULT_SHARD_SIZE = 24 * 1024 * 1024;
const SHARD_SIZE = Number(process.env.SHARD_SIZE || DEFAULT_SHARD_SIZE);
const FORCE = process.env.FORCE === '1';

if (!Number.isFinite(SHARD_SIZE) || SHARD_SIZE <= 0 || SHARD_SIZE > 25 * 1024 * 1024) {
  console.error(`SHARD_SIZE must be a positive integer ≤ 25 MiB; got ${SHARD_SIZE}`);
  process.exit(2);
}

const MANIFEST_VERSION = 1;

async function listModelIds() {
  const entries = await readdir(MODELS, { withFileTypes: true });
  return entries.filter((e) => e.isDirectory()).map((e) => e.name).sort();
}

async function* listQuants(modelId) {
  const ggufDir = join(MODELS, modelId, 'gguf');
  let entries;
  try {
    entries = await readdir(ggufDir, { withFileTypes: true });
  } catch (e) {
    if (e?.code === 'ENOENT') return;
    throw e;
  }
  for (const e of entries) {
    if (e.isFile() && e.name.endsWith('.gguf')) {
      const quant = e.name.replace(/\.gguf$/, '');
      yield {
        quant,
        src: join(ggufDir, e.name),
        outDir: join(ggufDir, quant),
      };
    }
  }
}

async function isUpToDate(srcStat, outDir) {
  if (FORCE) return false;
  let manifestRaw;
  try {
    const fs = await import('node:fs/promises');
    manifestRaw = await fs.readFile(join(outDir, 'manifest.json'), 'utf8');
  } catch {
    return false;
  }
  let manifest;
  try {
    manifest = JSON.parse(manifestRaw);
  } catch {
    return false;
  }
  if (manifest.version !== MANIFEST_VERSION) return false;
  if (manifest.totalSize !== srcStat.size) return false;
  if (manifest.shardSize !== SHARD_SIZE) return false;
  if (Math.floor(manifest.sourceMtimeMs) !== Math.floor(srcStat.mtimeMs)) return false;

  // Verify each shard exists and matches manifest size. Cheap cross-check —
  // catches half-deleted directories and partial writes from a killed run.
  for (const s of manifest.shards) {
    let st;
    try {
      st = await stat(join(outDir, s.name));
    } catch {
      return false;
    }
    if (st.size !== s.size) return false;
  }
  return true;
}

async function shardOne({ quant, src, outDir }) {
  const srcStat = await stat(src);
  if (await isUpToDate(srcStat, outDir)) {
    return { skipped: true, srcSize: srcStat.size, shardCount: shardCountFor(srcStat.size) };
  }

  // Wipe stale shards; otherwise a previous run with a larger size could
  // leave orphan shard files that the manifest no longer references.
  await rm(outDir, { recursive: true, force: true });
  await mkdir(outDir, { recursive: true });

  const shards = [];
  let shardIndex = 0;
  let offset = 0;

  // Read once, write each chunk to its own file. Streaming keeps peak
  // memory at one shard's worth of bytes regardless of source size.
  const readStream = createReadStream(src, { highWaterMark: 1024 * 1024 });
  let buffer = Buffer.alloc(SHARD_SIZE);
  let buffered = 0;

  const writeShard = async (size) => {
    const name = `${String(shardIndex).padStart(4, '0')}.bin`;
    const outPath = join(outDir, name);
    await writeFile(outPath, buffer.subarray(0, size));
    shards.push({ index: shardIndex, name, offset, size });
    offset += size;
    shardIndex += 1;
    buffered = 0;
  };

  for await (const chunk of readStream) {
    let cursor = 0;
    while (cursor < chunk.length) {
      const remainingInShard = SHARD_SIZE - buffered;
      const take = Math.min(remainingInShard, chunk.length - cursor);
      chunk.copy(buffer, buffered, cursor, cursor + take);
      buffered += take;
      cursor += take;
      if (buffered === SHARD_SIZE) await writeShard(SHARD_SIZE);
    }
  }
  if (buffered > 0) await writeShard(buffered);

  if (offset !== srcStat.size) {
    throw new Error(
      `sharder for ${src} wrote ${offset} bytes, source is ${srcStat.size} — refusing to write manifest`,
    );
  }

  const manifest = {
    version: MANIFEST_VERSION,
    quant,
    totalSize: srcStat.size,
    shardSize: SHARD_SIZE,
    shardCount: shards.length,
    sourceMtimeMs: srcStat.mtimeMs,
    shards,
  };
  await writeFile(
    join(outDir, 'manifest.json'),
    JSON.stringify(manifest, null, 2) + '\n',
  );

  return { skipped: false, srcSize: srcStat.size, shardCount: shards.length };
}

function shardCountFor(totalSize) {
  return Math.max(1, Math.ceil(totalSize / SHARD_SIZE));
}

function fmtMb(bytes) {
  return (bytes / 1024 / 1024).toFixed(0);
}

async function main() {
  const requestedIds = process.argv.slice(2);
  const ids = requestedIds.length ? requestedIds : await listModelIds();

  let totalBytes = 0;
  let totalShards = 0;
  for (const id of ids) {
    const idDir = join(MODELS, id);
    try {
      await stat(idDir);
    } catch {
      console.warn(`shard: skip ${id} (no such directory)`);
      continue;
    }
    let any = false;
    for await (const job of listQuants(id)) {
      any = true;
      const t0 = Date.now();
      const { skipped, srcSize, shardCount } = await shardOne(job);
      const tag = skipped ? 'up-to-date' : `${shardCount} shards in ${(Date.now() - t0) / 1000}s`;
      console.log(
        `shard: ${id.padEnd(28)} ${job.quant.padEnd(8)} ${fmtMb(srcSize).padStart(4)} MB  ${tag}`,
      );
      totalBytes += srcSize;
      totalShards += shardCount;
    }
    if (!any) console.log(`shard: ${id.padEnd(28)} (no .gguf files)`);
  }
  console.log(
    `shard: TOTAL ${fmtMb(totalBytes)} MB across ${totalShards} shards (shard size ${fmtMb(SHARD_SIZE)} MB)`,
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
