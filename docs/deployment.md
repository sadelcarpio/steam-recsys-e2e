# Deployment runbook

How to stand up the AWS infrastructure and deploy every component, from an empty account to a
running weekly pipeline. Region: `us-east-1`. `<acct>` is your AWS account id.

Commands after step 3 only need the AWS CLI: resource names are fixed, so they are looked up
directly instead of through `terraform output` (which needs the remote state initialized locally,
`terraform init -backend-config="bucket=tf-state-<acct>"`). Use `--profile <name>` or
`AWS_PROFILE` as usual.

## 0. Prerequisites

- AWS CLI v2 logged in with **your own admin credentials** (only for steps 1 and, the first time,
  3b). They never go into GitHub; CI/CD uses OIDC roles.
- Terraform >= 1.10, the GitHub CLI (`gh auth login`), Docker (only for local image builds).
- The repository on GitHub (`sadelcarpio/steam-recsys-e2e`, see `infrastructure/bootstrap/variables.tf`).

## 1. Bootstrap (once, local, admin credentials)

Creates the Terraform state bucket and the GitHub deploy role `steam-recsys-github-deploy`
(trusted only from `main`).

```bash
cd infrastructure/bootstrap
terraform init
terraform apply        # add -var create_github_oidc_provider=true if the account has no GitHub OIDC provider yet
```

## 2. GitHub repository variables (deploy)

Workflows read these as `${{ vars.NAME }}`: they are **repository variables** (GitHub → Settings
→ Secrets and variables → Actions → *Variables* tab), not secrets and not Environments. They are
resource names, not credentials.

```bash
cd infrastructure/bootstrap
gh variable set AWS_ROLE_ARN    --body "$(terraform output -raw github_deploy_role_arn)"
gh variable set TF_STATE_BUCKET --body "$(terraform output -raw tf_state_bucket)"
gh variable set AWS_REGION      --body us-east-1
```

## 3. Main infrastructure

### 3a. Normal path: `infrastructure CD` workflow (from `main`)

Actions → *infrastructure CD* → Run workflow → `action=plan`, review, then `action=apply`. The
job summary lists the Terraform outputs.

### 3b. First time a change adds infrastructure

The deploy role only trusts `main`, so new infrastructure can be applied by the workflow only
after it is on `main`. Pick one:

**Option A: direct push to `main`** (no admin credentials; the new code is tested on `main`
after the push instead of gating it):

1. Push to `main`. *etl CI* runs lint, unit tests and the image build; the Athena job is skipped
   (no `ETL_CI_*` variables yet).
2. *infrastructure CD* with `action=apply` (step 3a). This adds the `Transform` step to the
   pipeline before the `etl` image exists, so either finish steps 2-5 of this list before the
   next scheduled run (Thursday 17:00 CT) or disable the schedule `steam-recsys-weekly` in the
   EventBridge Scheduler console until step 5 (the next apply re-enables it). Otherwise that run
   scrapes normally, then fails at `Transform`. Nothing is lost: the next dbt run loads
   everything pending.
3. Set the `ETL_CI_*` variables (step 4).
4. Actions → *etl CI* → Run workflow on `main` (re-running the old run may not pick up the new
   variables). The Athena test now runs through the CI role.
5. Green: run *etl CD* (step 6). Red: fix on `main` and repeat from 4.

**Option B: local apply from the PR branch** (keeps CI as a gate before merging):

A PR's CI may need its new infrastructure first (e.g. the etl Athena test needs the CI
role/bucket/workgroup from `etl.tf`). Apply once from the branch with your admin credentials,
set the variables (step 4), re-run the PR's CI, then merge. The next CD run from `main` has
nothing to change:

```bash
cd infrastructure
terraform init -backend-config="bucket=tf-state-<acct>"
terraform plan -out tfplan && terraform apply tfplan
```

## 4. GitHub repository variables (etl CI)

Needed by the `athena-integration` job of *etl CI* (it is skipped while `ETL_CI_ROLE_ARN` is
unset). Values are Terraform outputs of step 3. After setting them, run *etl CI* manually
(Actions → *etl CI* → Run workflow, on `main`) or push to the PR:

The names are fixed, so the AWS CLI can look them up (no Terraform state needed locally):

```bash
ACCT=$(aws sts get-caller-identity --query Account --output text)
gh variable set ETL_CI_ROLE_ARN   --body "$(aws iam get-role --role-name steam-recsys-etl-ci --query Role.Arn --output text)"
gh variable set ETL_CI_BUCKET     --body "etl-ci-${ACCT}"
gh variable set ETL_CI_WORK_GROUP --body "steam-recsys-etl-ci"
```

(Or copy them from the *infrastructure CD* job summary into the Variables tab.)

| Variable | Value | Used by |
|---|---|---|
| `AWS_REGION` | `us-east-1` | all AWS workflows |
| `AWS_ROLE_ARN` | `github_deploy_role_arn` (bootstrap) | every CD workflow |
| `TF_STATE_BUCKET` | `tf_state_bucket` (bootstrap) | infrastructure CD |
| `ETL_CI_ROLE_ARN` | `etl_ci_role_arn` | etl CI (Athena test) |
| `ETL_CI_BUCKET` | `etl_ci_bucket` | etl CI (Athena test) |
| `ETL_CI_WORK_GROUP` | `etl_ci_work_group` | etl CI (Athena test) |

## 5. Secrets (out of band)

Terraform creates secrets with dummy values. Set the real ones yourself; they are never in the
repo, tfvars or GitHub:

```bash
aws secretsmanager put-secret-value --secret-id data-ingestion/steam-api-key --secret-string '<key>'
```

## 6. Deploy the components (from `main`)

Order matters only on the first deploy: the pipeline's ECS tasks run the `:latest` images.

| Workflow | Deploys |
|---|---|
| *data-ingestion CD* (`target=all`) | image `data-ingestion` (scraping tasks) + Lambda `list-partition-game-ids` code |
| *etl CD* | image `etl` (ECS task `dbt`) |
| *training CD* | image `training:<sha>` + a SageMaker training job for that commit (see step 9) |
| *inference CD* | image `inference` (SageMaker Processing job `Infer`, the pipeline's last step; see step 10) |
| *serving CD* | Lambda `recsys-serving` code (zip) + smoke test (see step 11) |

Later deploys: re-run the component's CD workflow after merging; the next pipeline run picks up
the new `:latest` image.

## 7. Run and verify

The EventBridge schedule starts the pipeline every Thursday 17:00 America/Chicago
(`schedule_enabled = false` pauses it). To run it now:

```bash
ARN=$(aws stepfunctions list-state-machines \
  --query "stateMachines[?name=='steam-recsys-pipeline'].stateMachineArn" --output text)
aws stepfunctions start-execution --state-machine-arn "$ARN" --input '{}'
```

Flow: `ListPartitionGameIds → Scrape (games + reviews) → Transform (dbt) → CheckChampion →
Infer` (`NoChampion`, a successful skip, until a model is promoted in step 9). The first run is
the backfill and takes hours (Steam rate limits). Logs: CloudWatch `/ecs/data-ingestion`,
`/ecs/etl`, `/aws/sagemaker/ProcessingJobs` (inference), `/aws/lambda/list-partition-game-ids`,
`/aws/lambda/recsys-serving`.

Check the marts in the Athena console (workgroup `steam-recsys-etl`):

```sql
select count(*) from steam_marts.interactions;
select * from steam_marts.game_features order by timestamp desc limit 10;
select count(*) from steam_marts.game_details;
```

**Weekly runs (incremental, nothing to do by hand).** After the first run, each scheduled
execution:
- scrapes only new games and new reviews (the DynamoDB cursors);
- runs dbt incrementally: only rows newer than each model's watermark, and new games get a
  `game_details` row;
- runs `Infer` with the current champion, which:
  - rewrites only the users whose recommendations changed, plus `__popular__`;
  - adds the new games to `game-details`;
  - republishes the online catalog.

The model itself is not retrained: training and promotion are manual (step 9), and the next
`Infer` picks up a new champion automatically. Re-run a component's CD only after merging a
change to it. An ETL change that adds a model or column needs no full refresh: a new
incremental model builds itself completely on its first run (a changed column of an existing
Iceberg table fails on purpose, `on_schema_change: fail`; then run step 8 with
`FULL_REFRESH`).

## 8. Run the ETL (dbt) on its own

Transforms whatever is in `raw-steam-data-<acct>` without scraping: use it for the first
backfill or after an ETL change. Every model is incremental and idempotent, so it is safe to
repeat. A second run with no new raw data writes nothing, and the next pipeline `Transform` only
picks up what arrived since.

**Never run two dbt runs at the same time** (this task and the pipeline's `Transform` step, or two
manual tasks). Both would read the same watermark and could process a batch twice. A scrape in
progress is fine: its files are loaded now or by the next run (lookback + already-loaded keys are
skipped).

```bash
SUBNETS=$(aws ec2 describe-subnets --filters "Name=tag:Name,Values=steam-recsys-public-*" \
  --query 'Subnets[].SubnetId' --output text | tr '\t' ',')
SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values=steam-recsys-egress-only \
  --query 'SecurityGroups[0].GroupId' --output text)
ARN=$(aws stepfunctions list-state-machines \
  --query "stateMachines[?name=='steam-recsys-pipeline'].stateMachineArn" --output text)

# 1. Nothing may be listed here (no pipeline execution running)
aws stepfunctions list-executions --state-machine-arn "$ARN" --status-filter RUNNING \
  --query 'executions[].name'

# 2. Start the dbt task (same network settings as the Transform step)
TASK=$(aws ecs run-task --cluster steam-recsys --task-definition dbt --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SG],assignPublicIp=ENABLED}" \
  --query 'tasks[0].taskArn' --output text)

# 3. Follow it and get the result: 0 = all models and data tests passed
aws logs tail /ecs/etl --follow            # Ctrl-C when dbt prints "Done."
aws ecs wait tasks-stopped --cluster steam-recsys --tasks "$TASK"
aws ecs describe-tasks --cluster steam-recsys --tasks "$TASK" \
  --query 'tasks[0].containers[0].exitCode'
```

Exit code 1: the log names the failed model or data test. Tables are written atomically, so
fix it and run again.

**Full rebuild** (recomputes everything exactly and keeps the lookup ids): add a container
override to step 2:

```bash
  --overrides '{"containerOverrides":[{"name":"dbt","environment":[{"name":"FULL_REFRESH","value":"true"}]}]}'
```

Check the result in the Athena console (workgroup `steam-recsys-etl`):

```sql
select count(*) from steam_marts.interactions;
select count(*) from steam_intermediate.int_games__deduplicated;
select * from steam_marts.user_features order by timestamp desc limit 10;
```

## 9. Train and promote a model

Training runs outside the weekly pipeline, on SageMaker training jobs, and needs the marts to
be populated (step 7 or 8).

**Quota (once).** New accounts have a SageMaker training quota of 0 for every instance type. Request
one instance of the type in `training_instance_type` (default `ml.m5.2xlarge`, quota code
`L-AD0A282D`; CPU quotas are usually approved within hours):

```bash
aws service-quotas request-service-quota-increase --service-code sagemaker \
  --quota-code L-AD0A282D --desired-value 1
aws service-quotas list-requested-service-quota-change-history --service-code sagemaker \
  --query 'RequestedQuotas[].[QuotaName,Status,DesiredValue]' --output table
```

**Train.** Actions → *training CD* → Run workflow (on `main`). Optional `env_overrides`, e.g.
`EPOCHS=10 BATCH_SIZE=2048`. The workflow pushes `training:<sha>` and starts
`steam-recsys-train-<sha12>-<timestamp>`. It watches the job for up to 50 min (the deploy role's
credentials last 1 h), and a longer job keeps running. The job summary shows recall@30/50/100
for warm / cold / all validation rows next to the popularity baseline. Follow a job with:

```bash
aws sagemaker list-training-jobs --name-contains steam-recsys --sort-by CreationTime \
  --query 'TrainingJobSummaries[:5].[TrainingJobName,TrainingJobStatus]' --output table
aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix <job name> --follow
ACCT=$(aws sts get-caller-identity --query Account --output text)
aws s3 cp s3://model-artifacts-${ACCT}/models/<sha>/metadata.json -
```

**Resume / extend.** A failed or stopped job leaves a per-epoch checkpoint. Re-running *training CD*
on the same commit resumes after the last finished epoch. A finished run keeps its checkpoint too:
re-run the same commit (a tag at it, under *Use workflow from*) with `EPOCHS=<more>` and only the
new epochs train, overwriting `models/<sha>/`. It works for 14 days (checkpoint lifecycle) and
until the next dbt run (new Iceberg snapshots start a fresh run).

**No quota? Train locally.** `python -m steam_training train` runs the same pipeline on a
workstation and uploads the model to the bucket (see `training/README.md`, *Development*). Then
promote it with `model_id=local-…` and `image_tag=<a pushed commit sha>`.

**GPU** (optional, once a GPU quota is granted, e.g. `ml.g4dn.xlarge`, quota code `L-3F53BF0F`):
run *training CD* with `torch_variant=cu128` and `instance_type=ml.g4dn.xlarge`.

**Promote.** Actions → *training promote* → Run workflow with `model_id` = the trained commit
sha (empty = the commit the workflow runs on). It evaluates the candidate and the current
champion on the same validation rows and writes `evaluation/<sha>/metrics.json`. When the
candidate wins, it copies the model to `models/champion/` and the report to
`evaluation/champion/metrics.json`. The first model only has to beat the popularity baseline.
To swap in a model that loses (e.g. the first one trained on the full backfill), add
`FORCE_PROMOTION=true` to `env_overrides`: it is still evaluated, and the report says it was forced.

**Roll back** the champion: the bucket is versioned, so restore the previous object versions of
`models/champion/*` and `evaluation/champion/metrics.json`, or promote an older sha again.

## 10. Recommendations (batch inference)

The pipeline's last step (`Infer`, a SageMaker Processing job on `ml.t3.xlarge`) runs every week once
`models/champion/metadata.json` exists: it scores every user of `user_features` against every
game, reranks the candidates of the 1000 most active users (>= 6 reviews) with Bedrock and
writes the users whose recommendations changed to DynamoDB `game-explainable-recommendations`
(plus the `__popular__` fallback item). It also loads the details of games missing from
DynamoDB `game-details` (the first run writes about 50k items; later runs write only new games).
The ETL must be deployed with the `game_details` mart first; without it the sync is skipped
with a warning. Details and cost:
`inference/README.md`.

**Bedrock access (once).** The default model is Amazon Nova 2 Lite through the US cross-region
inference profile (`us.amazon.nova-2-lite-v1:0`, Terraform variable
`inference_bedrock_model_id`). Serverless models are enabled on first use; for an Anthropic
model (e.g. `us.anthropic.claude-haiku-4-5-20251001-v1:0`) submit the one-time use-case form in
the Bedrock console first. Check the model answers:

```bash
aws bedrock-runtime converse --model-id us.amazon.nova-2-lite-v1:0 \
  --messages '[{"role":"user","content":[{"text":"Say ok"}]}]' --query 'output.message.content[0].text'
```

**Deploy / run now.** Actions → *inference CD* → Run workflow (on `main`). Tick `run_now` to
run the job right away (e.g. after promoting a new champion) instead of waiting for the weekly
pipeline; `env_overrides` passes container variables such as `RERANK_ENABLED=false` or
`MODEL_ID=<sha>`. Do not run it while the pipeline's `Infer` step runs (both write the same
items; harmless, but wasted Bedrock calls).

**Quota.** Processing jobs have a default quota on `ml.t3.*` (2 × `ml.t3.xlarge`), so no request
is needed. Every `ml.m5` / `ml.c5` processing quota is 0. For a non-burstable instance, request
e.g. `ml.m5.xlarge for processing job usage` and set `inference_instance_type`.

Follow a run:

```bash
aws sagemaker list-processing-jobs --name-contains steam-recsys-infer --sort-by CreationTime \
  --query 'ProcessingJobSummaries[:5].[ProcessingJobName,ProcessingJobStatus]' --output table
aws logs tail /aws/sagemaker/ProcessingJobs --log-stream-name-prefix <job name> --follow
```

**Check** the outputs:

```bash
aws dynamodb get-item --table-name game-explainable-recommendations --key '{"user_id":{"S":"<steam id>"}}'
aws dynamodb get-item --table-name game-explainable-recommendations --key '{"user_id":{"S":"__popular__"}}'
aws dynamodb scan --table-name game-details --select COUNT            # ~ the catalog size after the first run
ACCT=$(aws sts get-caller-identity --query Account --output text)
aws s3 cp s3://model-artifacts-${ACCT}/serving/online/manifest.json -  # online catalog (step 11)
```

The job's last log line (`inference done: {...}`) has the summary: users, written / unchanged /
deleted, reranked, `game_details_written`, `online_bundle`.

## 11. Serving API

`recsys-serving` is a Lambda behind a Lambda Function URL. It reads the tables of step 10, so it
returns 404 until the first inference run. API and examples: `serving/README.md`.

**Auth.** Actions → *infrastructure CD* → Run workflow → `serving_auth_type`:
- `AWS_IAM` (default): callers sign requests with SigV4. Assume
  `steam-recsys-serving-client` (output `serving_client_role_arn`) or grant
  `lambda:InvokeFunctionUrl` + `lambda:InvokeFunction` on the function.
- `NONE`: a public demo URL.

The value is applied on **every** infrastructure CD run, so choose `NONE` again on each apply
to keep the URL public. The outputs `serving_function_url` / `serving_auth_type` are in the
job summary.

**Deploy.** Actions → *serving CD* → Run workflow (on `main`). It runs the tests, uploads the
zip, invokes `/health`, `/popular` and `POST /recommendations` directly (this works under
either auth type) and prints the URL. Until the first serving CD, the function is a placeholder
that answers 503.

**Online model.** `POST /recommendations` needs two things:
- the champion's numpy user tower, which every model trained from now on gets automatically;
- the catalog that each inference run publishes to `s3://model-artifacts-<acct>/serving/online/`.

The current champion was trained before the export existed. Backfill it once (local, admin
credentials), then run *inference CD* with `run_now` or wait for the weekly run:

```bash
cd training && MODEL_ARTIFACTS_BUCKET=model-artifacts-<acct> uv run --extra cpu python -m steam_training export --model-id champion
```

Until then, inference logs `no numpy user tower` and publishes no catalog, and the endpoint
answers 503. Batch recommendations are not affected.

**Try it:**

```bash
URL=$(aws lambda get-function-url-config --function-name recsys-serving --query FunctionUrl --output text)
curl "${URL}popular?limit=3"                                   # NONE
curl "${URL}users/<steam id>/recommendations?details=false"    # unknown ids get the popular list
curl -X POST "${URL}recommendations" -H 'content-type: application/json' \
  -d '{"liked_game_ids": [620, 105600], "limit": 5}'           # online (503 before the first run)
# AWS_IAM: sign with the client role's credentials
curl --aws-sigv4 "aws:amz:us-east-1:lambda" --user "$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY" \
  -H "x-amz-security-token: $AWS_SESSION_TOKEN" "${URL}popular"
```

**Concurrency.** The account's Lambda limit is 10 concurrent executions by default, shared
with `list-partition-game-ids`. A public URL under load can throttle the pipeline's first
step. Before a public demo, request a higher `Concurrent executions` quota (Service Quotas →
AWS Lambda), then set `serving_reserved_concurrency` (e.g. 5) to cap the function.

## Updating an existing deployment

Roll a release onto an account that already runs the pipeline in this order. Skip the steps of
components the release does not touch.

1. Merge to `main`. The CIs run on the changed paths.
2. *infrastructure CD*: `plan`, review, then `apply`. Pick `serving_auth_type` on every run.
3. *data-ingestion CD* / *etl CD*: new images, used by the next `Scrape` / `Transform`.
4. Training changes only: *training CD* (new models); promote as in step 9.
5. *inference CD*: new image for the next `Infer`. With `run_now`, it also runs it immediately.
6. *serving CD*: new Lambda code.
7. Data now instead of Thursday: step 8 (ETL alone), then *inference CD* with `run_now`. Or run
   the whole pipeline (step 7).

Never start step 8 or `run_now` while a pipeline execution is running (step 8 shows how to
check).

**Release with serving, game details and the online endpoint (spec 5):** steps 1, 2 (creates
`game-details`, `recsys-serving` and its Function URL), 3 (*etl CD*: the `game_details` mart),
5, 6. Then backfill the current champion's `user_tower.npz` (step 11, *Online model*), and run
step 7 of this list. The first `Infer` afterwards:
- writes about 50k `game-details` items;
- writes `__popular__`;
- publishes the online catalog.

Verify with the checks of steps 10 and 11.

## Order for a fresh account (summary)

1 bootstrap → 2 deploy variables → 3 infrastructure → 4 etl CI variables → 5 secrets →
6 component CDs → 7 run (or 8, ETL only, to backfill from the raw data already there) →
9 SageMaker quota, train and promote (new models include `user_tower.npz`, so no backfill) →
10 inference (automatic from the next pipeline run, or *inference CD* with `run_now`) →
11 serving (*serving CD*; its auth is an *infrastructure CD* input). After that, the weekly
schedule runs everything incrementally (step 7).
