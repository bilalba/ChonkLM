// Copy lightweight tokenizer/config metadata + sharded GGUF weights from
// ../models/ into dist/models/ so Wrangler can ship everything as
// Cloudflare Workers static assets. Sharding (web/scripts/shard-gguf.mjs)
// keeps each .bin under the 25 MiB per-file cap, so we don't need R2.
//
// By default we ship Q4_K_M shards only — Q8_0 entries in the registry are
// devOnly. Set INCLUDE_Q8=1 to copy Q8_0 shards as well.

import { mkdir, copyFile, readdir, stat } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';

const ROOT = resolve(import.meta.dirname, '..');
const SRC = resolve(ROOT, '../models');
const DST = resolve(ROOT, 'dist/models');

const RAW_KEEP = new Set([
  'tokenizer.json',
  'tokenizer_config.json',
  'chat_template.jinja',
  'special_tokens_map.json',
  'generation_config.json',
]);

const REGISTRY_IDS = [
  'smollm2-135m-instruct',
  'smollm2-360m-instruct',
  'gemma-3-270m-it',
  'distilgpt2',
  'gpt2',
  'gpt2-medium',
  'granite-4.0-h-350m',
  'lfm2_5-350m',
  'monad',
  'baguettotron',
  'qwen3-0.6b',
  'openelm-270m-instruct',
];

const QUANTS = process.env.INCLUDE_Q8 === '1' ? ['q4_k_m', 'q8_0'] : ['q4_k_m'];

async function copyOne(src, dst) {
  await mkdir(dirname(dst), { recursive: true });
  await copyFile(src, dst);
}

async function copyRaw(srcDir, dstDir) {
  let bytes = 0;
  let files = 0;
  try {
    const entries = await readdir(join(srcDir, 'raw'));
    for (const name of entries) {
      if (!RAW_KEEP.has(name)) continue;
      const sp = join(srcDir, 'raw', name);
      const s = await stat(sp);
      if (!s.isFile()) continue;
      await copyOne(sp, join(dstDir, 'raw', name));
      bytes += s.size;
      files += 1;
    }
  } catch (e) {
    if (e?.code !== 'ENOENT') throw e;
  }
  return { bytes, files };
}

async function copyShardedQuant(srcDir, dstDir, quant) {
  const srcQuantDir = join(srcDir, 'gguf', quant);
  const dstQuantDir = join(dstDir, 'gguf', quant);
  let bytes = 0;
  let files = 0;
  let entries;
  try {
    entries = await readdir(srcQuantDir, { withFileTypes: true });
  } catch (e) {
    if (e?.code === 'ENOENT') return { bytes: 0, files: 0, present: false };
    throw e;
  }
  for (const e of entries) {
    if (!e.isFile()) continue;
    if (!(e.name === 'manifest.json' || e.name.endsWith('.bin'))) continue;
    const sp = join(srcQuantDir, e.name);
    const dp = join(dstQuantDir, e.name);
    const s = await stat(sp);
    await copyOne(sp, dp);
    bytes += s.size;
    files += 1;
  }
  return { bytes, files, present: files > 0 };
}

async function copyModelDir(id) {
  const srcDir = join(SRC, id);
  const dstDir = join(DST, id);
  let bytes = 0;
  let files = 0;
  const missingQuants = [];

  const raw = await copyRaw(srcDir, dstDir);
  bytes += raw.bytes;
  files += raw.files;

  for (const quant of QUANTS) {
    const r = await copyShardedQuant(srcDir, dstDir, quant);
    bytes += r.bytes;
    files += r.files;
    if (!r.present) missingQuants.push(quant);
  }

  return { id, bytes, files, missingQuants };
}

async function main() {
  await mkdir(DST, { recursive: true });
  let totalBytes = 0;
  let totalFiles = 0;
  for (const id of REGISTRY_IDS) {
    try {
      await stat(join(SRC, id));
    } catch {
      console.log(`models: skip ${id} (not on disk)`);
      continue;
    }
    const r = await copyModelDir(id);
    const mb = (r.bytes / 1024 / 1024).toFixed(0);
    const note = r.missingQuants.length
      ? ` (missing shards: ${r.missingQuants.join(', ')} — run web/scripts/shard-gguf.mjs)`
      : '';
    console.log(`models: ${id.padEnd(28)} ${mb.padStart(4)} MB  (${r.files} files)${note}`);
    totalBytes += r.bytes;
    totalFiles += r.files;
  }
  console.log(
    `models: TOTAL ${(totalBytes / 1024 / 1024).toFixed(0)} MB across ${totalFiles} files (quants: ${QUANTS.join(', ')})`,
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
