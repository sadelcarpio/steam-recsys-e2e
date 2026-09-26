# frontend

Spec: `specs/6-recsys-frontend.md`. Human docs: `README.md`.

## Layout

- Vite + TypeScript, no UI framework. Only runtime dependency: `minisearch`.
  - `src/api.ts`: types mirroring `serving/src/steam_serving/contracts.py` (responses) and
    `inference/src/steam_inference/contracts.py` (`SearchIndex`), fetch helpers under `/api`,
    `sha256Hex` for the `POST` signing header
  - `src/search.ts`: `buildSearch` (MiniSearch, prefix + fuzzy, boosted by review count);
    `src/search.worker.ts` downloads `/data/games.json` and answers queries off the main thread
  - `src/cards.ts`: DOM rendering (`h` helper, cards, details dialog, summaries)
  - `src/liked.ts`: picked games, most recent first, in localStorage (best effort)
  - `src/router.ts`: `/` and `/u/<id>` (history API; CloudFront serves index.html for
    extensionless paths)
  - `src/main.ts`: app shell and pages
- `tests/`: vitest + jsdom. `cloudfront.test.ts` evaluates `infrastructure/cloudfront/*.js`
  (the CloudFront Functions), so frontend CI also runs on changes there.

## Invariants

- All API calls are relative (`/api/...`, `/data/...`): the app has no config and no CORS.
  CloudFront strips the prefix (`infrastructure/cloudfront/strip_prefix.js`).
- `POST` bodies must be sent exactly as hashed (`x-amz-content-sha256`), and no request may set
  `Authorization`: CloudFront's OAC signs the Function URL call.
- API text (names, descriptions, LLM explanations) is set with `textContent`, never as HTML.
- `LIMIT` (30) / `MAX_LIKED_GAMES` (100) mirror serving's `MAX_LIMIT` / `MAX_LIKED_GAMES`.
  `SEARCH_INDEX_FORMAT` must match inference's `SEARCH_INDEX_FORMAT`: change both together.
- Steam user ids stay strings (they exceed JavaScript's safe integers).

## Commands

`npm ci && npm run format:check && npm run typecheck && npm test && npm run build`
Add deps only with `npm install` (never edit the lock).

Infra: `infrastructure/frontend.tf`. Workflows: `frontend-ci.yml`, `frontend-cd.yml`.
