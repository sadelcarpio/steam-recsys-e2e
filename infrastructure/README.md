# Infrastructure

Terraform for the whole project (AWS, `us-east-1`).

| Path | Contents |
|---|---|
| `bootstrap/` | Terraform state bucket `tf-state-<acct>`, deploy role `steam-recsys-github-deploy`. Reuses the account's existing GitHub OIDC provider (set `create_github_oidc_provider = true` in an account without one). Local state, applied once by hand. |
| `network.tf` | VPC with 2 public subnets and an IGW, no NAT. Egress-only security group. |
| `storage.tf` | S3 `raw-steam-data-<acct>`, `game-partitions-<acct>` (30-day expiry). DynamoDB `game-ids-state`, `reviews-state-cursor`. |
| `data_ingestion.tf` | ECR `data-ingestion`, ECS cluster and task definitions `games-scraping` / `reviews-scraping`, Lambda `list-partition-game-ids`, SSM `/data-ingestion/*`, secret `data-ingestion/steam-api-key`, IAM. |
| `orchestration.tf` | State machine `steam-recsys-pipeline` (Lambda, then parallel Distributed Maps of ECS tasks) and the EventBridge schedule (Thursdays 17:00 America/Chicago). |

## First-time setup

```bash
# 1. Bootstrap with your own admin credentials
cd infrastructure/bootstrap && terraform init && terraform apply
# 2. In GitHub → Settings → Variables (Actions), set:
#    AWS_ROLE_ARN    = <github_deploy_role_arn output>
#    AWS_REGION      = us-east-1
#    TF_STATE_BUCKET = <tf_state_bucket output>
# 3. Run the "infrastructure CD" workflow with action=apply
# 4. Run the "data-ingestion CD" workflow (pushes the image, deploys the Lambda code)
# 5. Put the real Steam API key in place of the dummy one:
aws secretsmanager put-secret-value --secret-id data-ingestion/steam-api-key --secret-string '<key>'
```

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
