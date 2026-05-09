import { defineConfig } from 'astro/config';
import AstroPWA from '@vite-pwa/astro';
import { createReadStream, statSync } from 'node:fs';
import { resolve } from 'node:path';

// Static output. Cloudflare Workers serves dist/ via the Static Assets
// binding configured in wrangler.jsonc. If we later need server endpoints,
// add `@astrojs/cloudflare` and switch `output: 'server'`.

// Dev-only static handler for /models/*. The full models tree (weights
// + sharded chunks + raw metadata) lives in the sibling /models dir
// and is too large to copy into web/public/. In production the build
// pipeline copies what we ship into web/dist/models/; in dev we stream
// straight off disk with Range support.
function makeStaticHandler(name, urlPrefix, root) {
  return {
    name,
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use(urlPrefix, (req, res, next) => {
        try {
          const relPath = decodeURIComponent((req.url || '/').split('?')[0]);
          const full = resolve(root, '.' + relPath);
          if (!full.startsWith(root)) {
            res.statusCode = 403;
            res.end('forbidden');
            return;
          }
          const stat = statSync(full);
          if (!stat.isFile()) return next();

          // Range support — kept on for the gemma-webgpu dev path (which
          // fetches a single .gguf with HTTP 206) and any other streaming
          // consumer. Sharded weights don't need it; each .bin is fetched
          // whole.
          res.setHeader('accept-ranges', 'bytes');
          res.setHeader('content-type', guessContentType(full));
          res.setHeader('cache-control', 'no-cache');

          const rangeHeader = req.headers['range'];
          if (rangeHeader && /^bytes=/.test(rangeHeader)) {
            const m = /^bytes=(\d*)-(\d*)$/.exec(rangeHeader);
            if (m) {
              const start = m[1] === '' ? Math.max(0, stat.size - parseInt(m[2], 10)) : parseInt(m[1], 10);
              const end = m[2] === '' ? stat.size - 1 : Math.min(parseInt(m[2], 10), stat.size - 1);
              if (start > end || start < 0 || end >= stat.size) {
                res.statusCode = 416;
                res.setHeader('content-range', `bytes */${stat.size}`);
                res.end();
                return;
              }
              res.statusCode = 206;
              res.setHeader('content-range', `bytes ${start}-${end}/${stat.size}`);
              res.setHeader('content-length', String(end - start + 1));
              createReadStream(full, { start, end }).pipe(res);
              return;
            }
          }

          res.setHeader('content-length', String(stat.size));
          createReadStream(full).pipe(res);
        } catch {
          next();
        }
      });
    },
  };
}

const MODELS_ROOT = resolve(import.meta.dirname, '../models');
const devServeModels = makeStaticHandler('chonklm-dev-models', '/models', MODELS_ROOT);

function guessContentType(p) {
  if (p.endsWith('.mjs') || p.endsWith('.js')) return 'text/javascript; charset=utf-8';
  if (p.endsWith('.json')) return 'application/json';
  if (p.endsWith('.gguf')) return 'application/octet-stream';
  if (p.endsWith('.txt')) return 'text/plain; charset=utf-8';
  return 'application/octet-stream';
}

export default defineConfig({
  output: 'static',
  site: 'https://chonklm.com',
  trailingSlash: 'ignore',
  build: {
    format: 'directory',
  },
  integrations: [
    // Generates dist/sw.js + dist/manifest.webmanifest, precaches the
    // app shell, and registers itself from every page. The shell is the
    // small stuff (HTML/JS/CSS/SVG/fonts). Sharded GGUF weights live
    // under /models/** and are managed by web/src/lib/cache.ts; we
    // explicitly keep them out of the SW so they don't double-cache or
    // blow past the precache size cap.
    AstroPWA({
      registerType: 'autoUpdate',
      injectRegister: 'auto',
      manifest: {
        name: 'chonklm',
        short_name: 'chonklm',
        description: 'Local LLM inference in the browser. No server inference.',
        start_url: '/',
        scope: '/',
        display: 'standalone',
        background_color: '#f8f4ee',
        theme_color: '#f8f4ee',
        icons: [
          {
            src: '/pwa-icon.svg',
            sizes: 'any',
            type: 'image/svg+xml',
            purpose: 'any maskable',
          },
        ],
      },
      workbox: {
        globPatterns: ['**/*.{html,js,css,svg,webp,woff2,webmanifest,ico}'],
        globIgnores: ['**/models/**'],
        navigateFallbackDenylist: [/^\/models\//],
        maximumFileSizeToCacheInBytes: 5 * 1024 * 1024,
      },
    }),
  ],
  vite: {
    plugins: [devServeModels],
  },
});
