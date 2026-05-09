// Cloudflare Worker entrypoint.
//
// Everything ships as static assets now. GGUF weights are sharded into
// ≤24 MiB byte-chunks by web/scripts/shard-gguf.mjs and copied into
// dist/models/ at build time, so they fit under the 25 MiB Workers
// static-asset cap and don't need R2.
//
// This Worker is therefore a thin passthrough to the static-assets
// binding. It exists only so we have a stable place to add per-request
// logic later (custom headers, redirects, etc.) without churning
// wrangler.jsonc.

interface Env {
  ASSETS: Fetcher;
}

export default {
  async fetch(req: Request, env: Env, _ctx: ExecutionContext): Promise<Response> {
    return env.ASSETS.fetch(req);
  },
};
