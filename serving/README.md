# Serving

`recsys-serving` is a Lambda behind a **Lambda Function URL**, with no API Gateway or load
balancer. It serves the recommendations that the weekly inference pipeline writes to DynamoDB.
Each request does one `GetItem` on `game-explainable-recommendations`, plus one `BatchGetItem`
on `game-details` when details are requested.

```
client ─► Function URL (auth: AWS_IAM | NONE) ─► recsys-serving ─► DynamoDB game-explainable-recommendations
                                                                 └► DynamoDB game-details
```

## API

All endpoints are `GET` and return JSON. Null fields are left out.

| Path | |
|---|---|
| `/users/{user_id}/recommendations` | The user's list (`source: "personalized"`). Users without one (cold start, unknown id) get the popularity list (`source: "popular"`) |
| `/popular` | The popularity list (the most reviewed-positive games of the last 90 days of reviews) |
| `/games/{game_id}` | Details of one game |
| `/health` | Liveness check, no DynamoDB call |

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
  - 400: invalid id, limit or details value.
  - 404: unknown route or game, or no popular list written yet.
  - 405: any method other than GET.
  - 500: an unhandled error, returned without internals.
- Successful list and game responses send `Cache-Control: public, max-age=300`, because the
  data changes weekly.

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

## Development

```bash
cd serving
uv sync
uv run pytest            # moto DynamoDB, no AWS
uv run ruff check . && uv run ruff format --check .
scripts/build_lambda.sh  # build/recsys-serving.zip (boto3 comes from the Lambda runtime)
```

## CI/CD

- `serving CI` (PRs and main; `serving/**`): ruff, tests, zip build.
- `serving CD` (manual): tests, then uploads the zip to `recsys-serving`. It then smoke-tests
  the function with direct invokes of `/health` and `/popular`, and prints the URL and auth
  type in the job summary (runbook: `docs/deployment.md` step 11).
- The contract with inference is tested in inference: `inference/tests/test_serving_contract.py`
  serves the items that the pipeline writes.
