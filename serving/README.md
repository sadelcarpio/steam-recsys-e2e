# Serving

`recsys-serving` is a Lambda behind a **Lambda Function URL**, with no API Gateway or load
balancer. It serves two kinds of recommendations:
- **Batch:** the lists the weekly inference pipeline writes to DynamoDB. Each request does one
  `GetItem` on `game-explainable-recommendations`, plus one `BatchGetItem` on `game-details`
  when details are requested.
- **Online:** computed per request for a list of liked games that the caller sends. The
  model's user tower (exported to numpy by training) embeds the liked games, and the result is
  an exact dot product against every catalog game's embedding (published by the pipeline every
  run). There is no ANN.

```
client ─► Function URL (auth: AWS_IAM | NONE) ─► recsys-serving ─► DynamoDB game-explainable-recommendations
                                                                 ├► DynamoDB game-details
                                                                 └► S3 models/<id>/user_tower.npz (training)
                                                                    + serving/online/ catalog (inference), in memory
```

## API

All endpoints are `GET` and return JSON. Null fields are left out.

| Path | |
|---|---|
| `/users/{user_id}/recommendations` | The user's list (`source: "personalized"`). Users without one (cold start, unknown id) get the popularity list (`source: "popular"`) |
| `/popular` | The popularity list (the most reviewed-positive games of the last 90 days of reviews) |
| `/games/{game_id}` | Details of one game |
| `/health` | Liveness check, no DynamoDB call |
| `POST /recommendations` | **Online**: recommendations for the liked games in the body (`source: "online"`) |

Query parameters of the two lists:
- `limit`: 1 to `MAX_LIMIT` (30), default `DEFAULT_LIMIT` (10).
- `details`: `true` or `false`, default true. Adds each game's details.

```bash
curl "$URL/users/76561198027267313/recommendations?limit=3"
```

```json
{
  "source": "personalized",
  "user_id": "76561198027267313",
  "model_id": "a1b2c3d",
  "generated_at": "2026-09-24T17:40:00Z",
  "reranked": true,
  "rerank_model": "us.amazon.nova-2-lite-v1:0",
  "recommendations": [
    {
      "rank": 1, "game_id": 63910, "name": "King's Bounty: Crossworlds", "score": 0.4127,
      "explanation": "As a fan of strategy and RPGs like Dungeons 2, ...",
      "details": {
        "game_id": 63910, "name": "King's Bounty: Crossworlds",
        "short_description": "Crossworlds is a stand-alone add-on ...",
        "header_image": "https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/63910/header.jpg",
        "release_date": "24 Sep, 2010", "is_free": false, "price": 9.99,
        "developers": ["Katauri Interactive"], "publishers": ["1C Entertainment"],
        "genres": ["RPG", "Strategy"], "categories": ["Single-player"]
      }
    }
  ]
}
```

- `explanation` is present only on the top 5 of a reranked list (the 1000 most active reviewers).
- `score` is the two-tower cosine similarity for personalized lists, and the share of the top
  game's positive reviews for the popular list.
- `details` is omitted for a game with no `game-details` item.
- Errors return `{"error": ..., "detail": ...}` with one of these statuses:
  - 400: invalid id, limit, details value or body.
  - 404: unknown route or game, or no popular list written yet.
  - 405: any method other than GET (except `POST /recommendations`).
  - 500: an unhandled error, returned without internals.
  - 503: online model not available (no catalog published yet).
- Successful list and game responses send `Cache-Control: public, max-age=300`, because the
  data changes weekly.

## Online recommendations

For a user who has no batch list yet, such as a new user who picks a few liked games in a
frontend:

```bash
curl -X POST "$URL/recommendations" -H 'content-type: application/json' \
  -d '{"liked_game_ids": [620, 105600], "limit": 5, "details": false}'
```

```json
{
  "source": "online", "model_id": "a1b2c3d", "generated_at": "2026-09-24T17:40:00Z",
  "reranked": false,
  "recommendations": [{"rank": 1, "game_id": 241910, "name": "Goodbye Deponia", "score": 0.5755}],
  "used_game_ids": [620, 105600],
  "ignored_game_ids": []
}
```

- `liked_game_ids`: Steam appids, **most recent first**, 1 to 100 of them. The first 5 that
  are in the catalog form the model's history (`used_game_ids`), which is the same input as
  batch `games_reviewed_positive`. Games outside the catalog are listed in
  `ignored_game_ids`. Every liked game is excluded from the results.
- If none of the liked games is known, the response is the popular list minus the liked games
  (`source: "popular"`).
- There are no LLM explanations online, and responses are never cached (`no-store`).
- The model comes in two parts, both in `s3://model-artifacts-<acct>/`:
  - **User tower:** `models/<model_id>/user_tower.npz`, written by training with each model
    (frozen game table, `seen_games`, 2 MLP layers; about 13 MB).
  - **Catalog:** `serving/online/catalog.npz`, written by the inference job every run (item
    embeddings, plus the `game_id`, `game_idx` and name of each catalog row; about 14 MB).
  
  `serving/online/manifest.json` pins the catalog's S3 version and names the user tower.
  Serving refuses a pair whose `model_id`s differ.
- **Mapping.** `game_id` is mapped to `game_idx` (the `lkp_games` ids) through the catalog
  rows in the catalog. Ids beyond the model's vocabulary, or games unseen in training, become
  OOV, exactly as in torch.
- **Loading.** The first POST in a container loads both parts (about 0.1 s after the S3
  downloads). A warm container re-reads the manifest at most every `ONLINE_REFRESH_SECONDS`
  (300) and switches when it changes. It reloads the user tower only when the model changed,
  and a failed refresh keeps the current model.
- **Latency.** One request takes about 0.4 ms of numpy (5 × 64 user tower plus a
  50k × 64 dot product). `inference/tests/test_online_parity.py` checks that the result equals
  torch and the batch retrieval.

## Auth

The Function URL auth type is the Terraform variable `serving_auth_type`, chosen with the
`serving_auth_type` input of the **infrastructure CD** on every run:

- **`AWS_IAM`** (default): requests must be SigV4-signed (service `lambda`) by a principal
  allowed `lambda:InvokeFunctionUrl` and `lambda:InvokeFunction` on the function. Terraform
  creates the role `steam-recsys-serving-client`, which any principal of the account may assume.
  ```bash
  creds=$(aws sts assume-role --role-arn <serving_client_role_arn> --role-session-name demo)
  # with those credentials exported:
  curl --aws-sigv4 "aws:amz:us-east-1:lambda" --user "$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY" \
    -H "x-amz-security-token: $AWS_SESSION_TOKEN" "$URL/popular"
  ```
- **`NONE`**: public, for demos. Anyone with the URL can call it. The account's Lambda
  concurrency limit (10 by default, shared with `list-partition-game-ids`) is then the only
  throttle: `serving_reserved_concurrency` can cap it once the limit is raised.

CORS (`serving_cors_allow_origins`, default `*`, GET only) is handled by the Function URL, so
a browser frontend can call it directly.

## Configuration

Pydantic settings (`steam_serving.config.ServingSettings`), read once per container.
Precedence: env vars > SSM `/serving/<ENV_VAR>` (read only when `USE_SSM=true`, as on the
Lambda).

| Env var | Default | |
|---|---|---|
| `RECOMMENDATIONS_TABLE` | `game-explainable-recommendations` | SSM |
| `GAME_DETAILS_TABLE` | `game-details` | SSM |
| `DEFAULT_LIMIT` / `MAX_LIMIT` | 10 / 30 | List length (inference writes 30 per user) |
| `INCLUDE_DETAILS` | true | Default of the `details` parameter |
| `CACHE_MAX_AGE_SECONDS` | 300 | |
| `MODEL_ARTIFACTS_BUCKET` | – | SSM. Bucket of the online model; unset disables `POST /recommendations` (503) |
| `ONLINE_BUNDLE_PREFIX` | `serving/online` | SSM |
| `ONLINE_REFRESH_SECONDS` | 300 | Manifest re-check interval of a warm container |
| `MAX_LIKED_GAMES` | 100 | |

## Development

```bash
cd serving
uv sync
uv run pytest            # moto DynamoDB + S3, no AWS
uv run ruff check . && uv run ruff format --check .
scripts/build_lambda.sh  # build/recsys-serving.zip, ~19 MB (numpy; boto3 comes from the runtime)
```

## CI/CD

- `serving CI` (PRs and main; `serving/**`): ruff, tests, zip build.
- `serving CD` (manual): tests, then uploads the zip to `recsys-serving`. It then smoke-tests
  the function with direct invokes of `/health`, `/popular` and `POST /recommendations`, and
  prints the URL and auth type in the job summary (runbook: `docs/deployment.md` step 11).
- The contract with inference is tested in inference:
  - `inference/tests/test_serving_contract.py` serves the items that the pipeline writes;
  - `test_online_parity.py` checks training's numpy export, the catalog and this package's
    numpy code against torch.
