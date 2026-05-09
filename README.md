# chonklm

Local LLM inference in the browser. Every token is generated on the user's
device with WebGPU — there is no server-side inference fallback.

The site curates a handful of small open-weights language models (135M–600M
params), ships them as **GGUF**, and runs them through custom WebGPU/WGSL
runtimes. Weights are served as **≤24 MiB byte-shards** out of Cloudflare
Workers static assets so we don't need R2 or any object storage.

## Layout

```
src/         Astro frontend, custom WGSL runtimes
scripts/     Sharder + build-time copy of model metadata into dist/
worker/      Thin Cloudflare Worker that wraps the static-assets binding
public/      Favicons and brand logos
```

## Quick start

```bash
npm install
npm run dev          # Astro dev server
```

The dev server expects model artifacts on disk at a sibling `../models/`
path. **Model weights are not part of this repo** — see "Bring your own
weights" below.

## Bring your own weights

This repo ships only source code. To actually run inference you need GGUF
weights and lightweight tokenizer/config metadata under a sibling
`../models/<id>/` directory:

```
../models/<id>/
  gguf/
    q4_k_m.gguf            # original GGUF
    q4_k_m/                # shards built by scripts/shard-gguf.mjs
      manifest.json
      0000.bin
      0001.bin
      …
  raw/
    tokenizer.json
    tokenizer_config.json
    generation_config.json
    chat_template.jinja     # if the model uses one
    LICENSE                 # upstream model license
```

Convert from the upstream HF checkpoint with `llama.cpp` (or download a
prebuilt GGUF), drop it at `../models/<id>/gguf/<quant>.gguf`, then:

```bash
node scripts/shard-gguf.mjs <id>
```

This builds `<quant>/manifest.json` plus the `0000.bin`, `0001.bin`, …
byte-shards next to the original file. The `.gguf` and `.bin` files stay
out of git; only manifests and metadata get copied into `dist/` at build
time by `scripts/copy-models-to-dist.mjs` (Q4_K_M only by default; set
`INCLUDE_Q8=1` to also include Q8_0).

## How weights are served

`src/lib/cache.ts` rebuilds logical byte-ranges out of shards in the
browser, with at most 4 shards in flight at a time. After the first
successful load the page calls `navigator.storage.persist()` so the
browser keeps the Cache-API entries around.

## Deploying

```bash
npm run deploy       # astro build + wrangler deploy
```

This pushes everything in `dist/` (HTML, JS, model metadata, GGUF shards)
to Cloudflare Workers static assets via the binding configured in
`wrangler.jsonc`. No R2, no Pages.

## Adding a new model

1. Drop the GGUF (Q4_K_M and optionally Q8_0) and tokenizer/config JSON
   under `../models/<new-id>/`.
2. Run `node scripts/shard-gguf.mjs <new-id>` to build shards.
3. Add a `ModelDef` entry to `src/lib/registry.ts`. Keep `gguf`
   pointing at the shard *directory* (e.g. `gguf/q4_k_m`), not the
   `.gguf` file.
4. Make sure a runtime in `src/lib/` can actually execute it.
   Supported runtimes are `llama-webgpu`, `lfm2-webgpu`,
   `granite-webgpu`, `gemma-webgpu`, and the gpt2-style `gpt-webgpu`.

## License

Source code: MIT (see [LICENSE](LICENSE)).

Each model artifact served by this site retains its own upstream license.
Bundle the upstream `LICENSE` file under `../models/<id>/raw/` so it ships
into `dist/` alongside the tokenizer.
