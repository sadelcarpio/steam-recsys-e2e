# infrastructure

Human docs: `README.md`. A single Terraform root with one file per component, plus `bootstrap/`.

## Conventions

- IAM role names **must** start with `${var.project}-` (`steam-recsys-`). The GitHub deploy
  role can only create, modify and pass roles under that prefix (see `bootstrap/main.tf`).
- ECR repositories use the component directory name in kebab-case (`data-ingestion`).
- Component config goes in SSM `/<component-kebab>/<ENV_VAR>` and secrets go in Secrets
  Manager `<component-kebab>/<name>`. Secret values are dummy placeholders with
  `ignore_changes`, and the real values are set out of band. Never put real secrets in code
  or tfvars.
- Application code is deployed by the component CD workflows, not by Terraform. The Lambdas
  (`list-partition-game-ids`, `recsys-serving`) use a placeholder zip plus
  `ignore_changes = [filename, source_code_hash]`, and ECS task definitions use
  `:${var.scraping_image_tag}` (default `latest`).
- The deploy role has no `iam:CreatePolicy`: use inline role policies
  (`aws_iam_role_policy`), not managed policies.
- `var.serving_auth_type` comes from the infrastructure CD input (`TF_VAR_serving_auth_type`),
  so a plan without it falls back to `AWS_IAM`. A public URL (`NONE`) needs both
  `aws_lambda_permission`s (`InvokeFunctionUrl` + `InvokeFunction` via the function URL).
- `frontend.tf` owns the only bucket policy of `model-artifacts` (CloudFront reads
  `serving/search/*`): add statements there rather than a second `aws_s3_bucket_policy`. The
  CloudFront Functions (`cloudfront/*.js`, runtime `cloudfront-js-2.0`, ES5-ish) are tested by
  `frontend/tests/cloudfront.test.ts`. The `/api/*` cache policy must not key on headers (they
  would be forwarded, and the Function URL needs its own `Host`).
- The state machine definition is HCL (`local.pipeline_definition` in `orchestration.tf`):
  `ListPartitionGameIds -> Scrape -> Transform -> CheckChampion -> HasChampion -> Infer |
  NoChampion`. `CheckChampion` lists `models/champion/metadata.json` (written last by a
  promotion). New states need matching permissions on `aws_iam_role.pipeline` (RunTask on the
  task definition, PassRole on its roles).
- Raw Glue tables (`local.raw_tables` in `etl.tf`) mirror the scraper parquet schema; keep them in
  sync with `data_ingestion/src/steam_ingestion/schemas.py` and `etl/tests/integration/fixtures.py`.
- PR-scoped CI roles (e.g. `steam-recsys-etl-ci`) live next to their component and only get
  sandbox resources (`ci_*` Glue databases, CI bucket / workgroup), never prod data.
- Backend: S3 with `use_lockfile`. The bucket is passed with `-backend-config`.

## Checks

`terraform fmt -check -recursive && terraform init -backend=false && terraform validate`
(run in both `infrastructure/` and `infrastructure/bootstrap/`).
