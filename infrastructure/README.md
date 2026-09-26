# Infrastructure

Terraform for the whole project (AWS, `us-east-1`).

| Path                | Contents                                                                                                                                                                                                                                        |
|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `bootstrap/`        | Terraform state bucket `tf-state-<acct>`, deploy role `steam-recsys-github-deploy`. Reuses the account's existing GitHub OIDC provider (set `create_github_oidc_provider = true` in an account without one). Local state, applied once by hand. |
| `network.tf`        | VPC with 2 public subnets and an IGW, no NAT. Egress-only security group.                                                                                                                                                                       |
| `storage.tf`        | S3 `raw-steam-data-<acct>`, `game-partitions-<acct>` (30-day expiry). DynamoDB `game-ids-state`, `reviews-state-cursor`.                                                                                                                        |
| `data_ingestion.tf` | ECR `data-ingestion`, ECS cluster and task definitions `games-scraping` / `reviews-scraping`, Lambda `list-partition-game-ids`, SSM `/data-ingestion/*`, secret `data-ingestion/steam-api-key`, IAM.                                            |
| `etl.tf`            | S3 `processed-steam-data-<acct>` (Iceberg lake + Athena results), Glue databases `steam_{raw,staging,intermediate,marts}`, raw external tables `steam_raw.games` / `reviews`, Athena workgroup `steam-recsys-etl`, ECR `etl`, ECS task definition `dbt`, SSM `/etl/*`, IAM. CI sandbox: bucket `etl-ci-<acct>` (2-day expiry), workgroup `steam-recsys-etl-ci`, role `steam-recsys-etl-ci` (PRs + main, only `ci_*` Glue databases). |
| `training.tf`       | S3 `model-artifacts-<acct>` (versioned; models, evaluation reports, champion), ECR `training`, SageMaker execution role `steam-recsys-training` (reads the marts through Glue + S3, read/write on the artifacts bucket), SSM `/training/*` (`training_instance_type`, default `ml.m5.2xlarge`). |
| `inference.tf`      | DynamoDB `game-explainable-recommendations` (PK `user_id`; only changed users are rewritten, plus the `__popular__` fallback item) and `game-details` (PK `game_id` N, insert-only), ECR `inference`, SageMaker execution role `steam-recsys-inference` (marts read, `models/*` read, `serving/online/*` + `serving/search/games.json` write (online catalog, frontend search index), scan + writes on both tables, `bedrock:InvokeModel` on `inference_bedrock_model_id` only), SSM `/inference/*`. The processing job itself is the `Infer` state (`inference_instance_type`, default `ml.t3.xlarge`). |
| `serving.tf`        | Lambda `recsys-serving` (zip, placeholder until the serving CD) behind a Lambda Function URL, auth `serving_auth_type` (`AWS_IAM` default or `NONE`, an infrastructure CD input; `NONE` adds the two public permissions), CORS `serving_cors_allow_origins`, role `steam-recsys-serving` (GetItem / BatchGetItem, read of the online catalog `serving/online/*` and the numpy user towers `models/*/user_tower.npz`; 1 GB memory), client role `steam-recsys-serving-client` (may invoke the URL under `AWS_IAM`), SSM `/serving/*`. |
| `frontend.tf`       | S3 `recsys-frontend-<acct>` (the built app), CloudFront distribution with three origins (`/*` the app, `/data/*` the search index in `model-artifacts-<acct>/serving/search/`, `/api/*` the serving Function URL), Origin Access Controls (S3, and Lambda when `serving_auth_type = AWS_IAM`), bucket policies for the distribution, CloudFront Functions (`cloudfront/spa.js`: app routes → `index.html`; `cloudfront/strip_prefix.js`: `/api` and `/data` prefixes dropped), Lambda permissions for CloudFront, optional custom domain (`frontend_domain_name` + `frontend_certificate_arn`). Outputs `frontend_url`, `frontend_cloudfront_domain`. |
| `orchestration.tf`  | State machine `steam-recsys-pipeline` (Lambda, then parallel Distributed Maps of ECS tasks, then the `dbt` task, then `CheckChampion` → the SageMaker Processing job `Infer` or a skip) and the EventBridge schedule (Thursdays 17:00 America/Chicago).                                                               |

## Deployment

Step-by-step from an empty account (bootstrap, GitHub variables, apply, secrets, component
deploys, first run): [`docs/deployment.md`](../docs/deployment.md).

To apply from a workstation: `terraform init -backend-config="bucket=tf-state-<acct>" && terraform apply`.

## Running the pipeline manually

```bash
aws stepfunctions start-execution --state-machine-arn <pipeline_state_machine_arn> --input '{}'
# Re-run the partitions of an earlier run (idempotent):
aws stepfunctions start-execution --state-machine-arn <arn> --input '{"run_id": "<previous run_id>"}'
```

Set `schedule_enabled = false` to pause the weekly schedule. Tunables are in `variables.tf`
(`num_review_workers`, `games_per_task`, `max_reviews_per_game`, ...).

## CI/CD

- `infrastructure CI`: `fmt -check` and `validate` for both roots, on changes under `infrastructure/`.
- `infrastructure CD` (manual): `plan`, or `plan` + `apply`, through OIDC. The deploy role only
  trusts `refs/heads/main`. Input `serving_auth_type` (`AWS_IAM` / `NONE`) sets the serving
  Function URL auth on every run. Repository variables `FRONTEND_DOMAIN_NAME` /
  `FRONTEND_CERTIFICATE_ARN` (optional) set the frontend's custom domain.
- `etl CI` assumes `steam-recsys-etl-ci`, which trusts pull requests and `main` of this repo but
  can only touch the CI bucket, the CI workgroup and `ci_*` Glue databases.
- `inference CD` pushes the image and, with `run_now`, starts the `Infer` processing job once.
- `serving CD` uploads the `recsys-serving` zip and smoke-tests it.
- `training CD` / `training promote` use the deploy role to push the image and start SageMaker
  training jobs, which run as `steam-recsys-training` (passed by the deploy role).
