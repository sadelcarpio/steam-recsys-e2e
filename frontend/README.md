# Frontend

Static web app over the serving API, in Spanish: search Steam games, pick up to 5 you liked,
and get recommendations as game cards. Or open a Steam user's precomputed recommendations at
`/u/<steam id>`. No login.

| Page          | What it does                                                        | API                                          |
|---------------|---------------------------------------------------------------------|----------------------------------------------|
| `/`           | Search, pick up to 5 liked games (the model reads the last 5), **Recomendar** | `POST /api/recommendations`         |
| `/u/<id>`     | A user's recommendations (popular games for unknown users)          | `GET /api/users/<id>/recommendations`        |
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
- **Spanish UI.** Every text the app renders is in `src/i18n.ts`, including Steam's genre
  names. Game names, descriptions and LLM explanations are shown as the data has them.
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
