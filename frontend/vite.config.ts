/// <reference types="vitest/config" />
import { defineConfig } from "vite";

// `npm run dev` proxies /api and /data to a deployed frontend (RECSYS_URL, the CloudFront URL:
// terraform output frontend_url), so the app runs locally against the real API.
const backend = process.env.RECSYS_URL;

export default defineConfig({
  server: backend
    ? {
        proxy: {
          "/api": { target: backend, changeOrigin: true },
          "/data": { target: backend, changeOrigin: true },
        },
      }
    : undefined,
  build: { target: "es2022" },
  test: { environment: "jsdom" },
});
