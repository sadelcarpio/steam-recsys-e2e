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
| `inference.tf`      | DynamoDB `game-explainable-recommendations` (PK `user_id`, TTL `expires_at`), ECR `inference`, ECS task definition `inference` (Fargate, `inference_cpu` / `inference_memory`), SSM `/inference/*`, IAM (marts read, `models/*` read, table writes, `bedrock:InvokeModel` on `inference_bedrock_model_id` only). |
| `orchestration.tf`  | State machine `steam-recsys-pipeline` (Lambda, then parallel Distributed Maps of ECS tasks, then the `dbt` task, then `CheckChampion` → the `inference` task or a skip) and the EventBridge schedule (Thursdays 17:00 America/Chicago).                                                               |

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
  trusts `refs/heads/main`.
- `etl CI` assumes `steam-recsys-etl-ci`, which trusts pull requests and `main` of this repo but
  can only touch the CI bucket, the CI workgroup and `ci_*` Glue databases.
- `inference CD` pushes the image and, with `run_now`, runs the `inference` task once.
- `training CD` / `training promote` use the deploy role to push the image and start SageMaker
  training jobs, which run as `steam-recsys-training` (passed by the deploy role).
