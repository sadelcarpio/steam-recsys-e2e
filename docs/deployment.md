# Deployment runbook

How to stand up the AWS infrastructure and deploy every component, from an empty account to a
running weekly pipeline. Region: `us-east-1`. `<acct>` is your AWS account id.

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

```bash
cd infrastructure
gh variable set ETL_CI_ROLE_ARN   --body "$(terraform output -raw etl_ci_role_arn)"
gh variable set ETL_CI_BUCKET     --body "$(terraform output -raw etl_ci_bucket)"
gh variable set ETL_CI_WORK_GROUP --body "$(terraform output -raw etl_ci_work_group)"
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

Later deploys: re-run the component's CD workflow after merging; the next pipeline run picks up
the new `:latest` image.

## 7. Run and verify

The EventBridge schedule starts the pipeline every Thursday 17:00 America/Chicago
(`schedule_enabled = false` pauses it). To run it now:

```bash
ARN=$(cd infrastructure && terraform output -raw pipeline_state_machine_arn)
aws stepfunctions start-execution --state-machine-arn "$ARN" --input '{}'
```

Flow: `ListPartitionGameIds → Scrape (games + reviews) → Transform (dbt)`. The first run is the
backfill and takes hours (Steam rate limits). Logs: CloudWatch `/ecs/data-ingestion`, `/ecs/etl`,
`/aws/lambda/list-partition-game-ids`.

Check the marts in the Athena console (workgroup `steam-recsys-etl`):

```sql
select count(*) from steam_marts.interactions;
select * from steam_marts.game_features order by timestamp desc limit 10;
```

Full ETL rebuild (keeps the lookup ids): run the `dbt` task once with a `FULL_REFRESH=true`
container override (ECS console → Run task, or `aws ecs run-task ... --overrides`).

## Order for a fresh account (summary)

1 bootstrap → 2 deploy variables → 3 infrastructure → 4 etl CI variables → 5 secrets →
6 component CDs → 7 run.
