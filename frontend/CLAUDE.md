# frontend

Spec: `specs/6-recsys-frontend.md`. Human docs: `README.md`.

## Layout

- Vite + TypeScript, no UI framework. Only runtime dependency: `minisearch`.
  - `src/api.ts`: types mirroring `serving/src/steam_serving/contracts.py` (responses) and
    `inference/src/steam_inference/contracts.py` (`SearchIndex`), fetch helpers under `/api`,
    `sha256Hex` for the `POST` signing header
  - `src/i18n.ts`: every UI text (Spanish), genre name translations, date formatting
  - `src/search.ts`: `buildSearch` (MiniSearch, prefix + fuzzy, relevance only);
    `src/search.worker.ts` downloads `/data/games.json` and answers queries off the main thread
  - `src/cards.ts`: DOM rendering (`h` helper, cards, details dialog, summaries)
  - `src/liked.ts`: picked games, most recent first, at most `MAX_LIKED_GAMES`, in
    localStorage (best effort)
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
- The UI is Spanish: no user-visible string outside `src/i18n.ts`. API errors reach the page
  as Spanish messages by status (`ApiError.message`); the server's text is only in `detail`.
- `LIMIT` (30) mirrors serving's `MAX_LIMIT`. `MAX_LIKED_GAMES` (5) is the user tower's
  history length (training `USER_HISTORY_LENGTH`), enforced by `LikedGames`; serving accepts
  up to 100.
- `SEARCH_INDEX_FORMAT` must match inference's `SEARCH_INDEX_FORMAT`: change both together.
- Steam user ids stay strings (they exceed JavaScript's safe integers).

## Commands

`npm ci && npm run format:check && npm run typecheck && npm test && npm run build`
Add deps only with `npm install` (never edit the lock).

Infra: `infrastructure/frontend.tf`. Workflows: `frontend-ci.yml`, `frontend-cd.yml`.
