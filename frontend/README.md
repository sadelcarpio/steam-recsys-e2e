# Frontend

Static web app over the serving API, in English or Spanish: search Steam games, pick up to 20 you liked,
and get recommendations as game cards. Or open a user's precomputed recommendations by user
number at `/u/<n>` (1 = the most active reviewer; spec 13: no Steam id is stored or accepted).
No login.

| Page          | What it does                                                        | API                                          |
|---------------|---------------------------------------------------------------------|----------------------------------------------|
| `/`           | Search, pick up to 20 liked games, **Recomendar**. Each pick shows its position in the model's history (1 = most recent; a new pick becomes 1); drag a pick onto another one, or focus it and use the arrow keys, to change the order | `POST /api/recommendations`         |
| `/u/<n>`      | Demo user `n`'s recommendations (popular games for users without a list), with ← / → links to the neighbouring users. Above them, the user's last liked games (the model's input); under each LLM explanation, the model that wrote it (`modelo: <rerank_model>`). The summary above the list names the models that produced it: the retriever (`modelo de dos torres: <model_id>`), plus the ranker when one ordered the list (`retrieval: <model_id>, ranking: <ranker_id>`; commit shas shortened to 7 characters), on `/u/<n>` and on the discover page. Numbers are reassigned every pipeline run | `GET /api/users/<n>/recommendations` |
| card click    | Game details dialog + Steam store link                              | details come with the list (`details=true`)  |

## How it is served

```
Browser ──► CloudFront
              ├─ /*        → S3 recsys-frontend-<acct>      (this app; /u/... → index.html)
              ├─ /data/*   → S3 model-artifacts-<acct>/serving/search   (games.json)
              └─ /api/*    → Lambda Function URL recsys-serving          (prefix stripped)
```

- One domain for everything: no CORS, no API URL to configure in the app.
- The Function URL can stay `AWS_IAM`: CloudFront signs the calls (Origin Access Control), so
  users need no credentials, and the raw URL refuses everyone else. `POST` requests carry
  `x-amz-content-sha256` (the body's SHA-256), which CloudFront needs to sign them.
- **Search runs in the browser.** Each inference run writes `games.json` (gzipped: the ~180k
  games of the online catalog as `[appid, name, reviews]`). A Web Worker downloads it once
  and indexes it with [MiniSearch](https://lucaong.github.io/minisearch/) (prefix + typo
  tolerant, ranked by relevance only). There is no search server.
- **No adult games.** Inference leaves them out of the index, the catalog and every
  recommendation list (`inference/src/steam_inference/adult.py`).
- **English and Spanish UI.** Every text the app renders is in `src/i18n.ts`, in both
  languages (Steam's genre names are translated in Spanish). English by default; `?lang=es` /
  `?lang=en` or the header link switches it and remembers the choice (localStorage). Game
  names, descriptions and the LLM explanations (written by inference in one language,
  `EXPLANATION_LANGUAGE`: English by default) are shown as the data has them.
- Images come straight from Steam's CDN (`header_image`).

Infrastructure: `infrastructure/frontend.tf` (+ the CloudFront Functions in
`infrastructure/cloudfront/`).

## Development

```bash
npm ci
RECSYS_URL=https://<frontend_url> npm run dev   # local app, real API + search index via the deployed CloudFront
npm test                                        # vitest (jsdom), no AWS
npm run typecheck && npm run format:check
npm run build                                   # → dist/
```

## Deploy

1. *infrastructure CD* (`apply`): bucket, distribution, functions. The URL is the
   `frontend_url` output.
2. *frontend CD*: builds, syncs `dist/` to the bucket, invalidates `index.html`, smoke-tests
   the app route, `/api/health` and `/data/games.json`.
3. Search needs one inference run after the infrastructure apply (the index is written with
   the online catalog). Until then the search box says it is unavailable; user pages work.

### Custom domain

1. Request an ACM certificate for the domain **in us-east-1** (CloudFront's region) and
   validate it (DNS record).
2. Set the repository variables `FRONTEND_DOMAIN_NAME` (e.g. `recs.example.com`) and
   `FRONTEND_CERTIFICATE_ARN`, then *infrastructure CD* → `apply`.
3. Point the domain at the `frontend_cloudfront_domain` output: a `CNAME` record, or an
   alias `A`/`AAAA` record in Route 53.
