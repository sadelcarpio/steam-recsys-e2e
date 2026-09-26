## Spec 6: RecSys Frontend

A simple static web app, on top of the Spec 5 serving Lambda, to search the game catalog, pick liked games and see
recommendations as game cards. No login: anyone with the link can use it.

### Pages

| Path          | What it does                                                                                       | API call                                              |
|---------------|----------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| `/`           | Search games by name, add them to a "liked" list, get recommendations for that list               | `POST /api/recommendations`                           |
| `/u/{userId}` | Recommendations for an existing user id (typed in a box on `/` or opened directly by link)         | `GET /api/users/{userId}/recommendations?details=true` |
| (empty state) | Popular games when nothing is selected yet, or when the user is unknown                            | `GET /api/popular?details=true`                       |

Every game is shown as a card with its image (`header_image`), name, and the explanation text (LLM reason) when the
recommendation has one. Clicking a card opens the game's details (`GET /api/games/{id}`) with a link to its Steam page.

### Architecture

```
Browser ──► CloudFront
              ├─ /*        → S3 frontend bucket (private, OAC)  → built static files
              ├─ /data/*   → S3 search index (private, OAC)     → games.json.gz
              └─ /api/*    → Lambda Function URL (OAC)          → recsys-serving
```

- **Static site**: plain Vite build (framework of choice, keep it light), stored in a private S3 bucket
  `recsys-frontend-<account-id>`, served by CloudFront. Client-side routing: a CloudFront Function serves
  `index.html` for extensionless paths (not a 403/404 error response, which would also rewrite API errors).
- **API through the same distribution**: `/api/*` is routed to the serving Lambda Function URL, so there is no CORS.
  A CloudFront Function strips the `/api` prefix before it reaches the Lambda.
- **Auth**: the Function URL stays `AWS_IAM`. CloudFront signs requests with Origin Access Control (lambda origin type),
  and the function's resource policy only allows `cloudfront.amazonaws.com` from this distribution
  (`lambda:InvokeFunctionUrl` + `lambda:InvokeFunction`). The raw Function URL returns 403 to everyone else. With
  auth `NONE` CloudFront calls the URL unsigned. For
  `POST` the browser must send `x-amz-content-sha256` (hex SHA-256 of the body); CloudFront does not hash the body.
  The browser must not send its own `Authorization` header.
- **Abuse protection**: the site is public, so cap the Lambda with reserved concurrency (configurable variable), and
  cache the GET routes in CloudFront (`/popular` and `/games/*` for minutes, user recs shorter). WAF is optional,
  off by default.
- **Images**: loaded straight from Steam's CDN (`header_image`), not copied to S3.

### Game search (in the browser, no search backend)

- The batch inference pipeline (Spec 4) writes a compact search index on every run:
  `s3://model-artifacts-<account-id>/serving/search/games.json` (stored gzipped, `Content-Encoding: gzip`), served
  by CloudFront at `/data/games.json`.
- Contents: only the games in the online catalog (the ones `POST /recommendations` can use), as
  `[[app_id, name, n_reviews], ...]`. Validated with a Pydantic model when written.
- The browser downloads it once (cached), builds a MiniSearch/FlexSearch index in a Web Worker, and searches by
  name prefix + fuzzy match, ranked by `n_reviews`. No Elasticsearch/OpenSearch.

### Infrastructure & CI/CD

- Terraform in `infrastructure/frontend.tf`: S3 bucket, CloudFront distribution + OACs, CloudFront Function, Lambda
  permission, reserved concurrency. Output the CloudFront URL.
- `frontend/` directory with its own `CLAUDE.md` and `README.md`.
- `frontend-ci.yml` (on changes under `frontend/`): lint, type check, unit tests (search, request signing header,
  card rendering against sample API responses), build.
- `frontend-cd.yml` (manual, OIDC): build, `aws s3 sync` to the bucket, CloudFront invalidation.
- Inference pipeline change (search index export) comes with its own tests; update `docs/deployment.md` with the
  frontend deploy step and URL.

### Custom domain

Optional: repository variables `FRONTEND_DOMAIN_NAME` + `FRONTEND_CERTIFICATE_ARN` (ACM, us-east-1) feed the
infrastructure CD; DNS points a CNAME / alias at the distribution.

### Out of scope

User login, saving liked lists server-side.
