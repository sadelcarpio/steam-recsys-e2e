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
- Application code is deployed by the component CD workflows, not by Terraform. The Lambda
  uses a placeholder zip plus `ignore_changes = [filename, source_code_hash]`, and ECS task
  definitions use `:${var.scraping_image_tag}` (default `latest`).
- The state machine definition is HCL (`local.pipeline_definition` in `orchestration.tf`).
  Later specs add states after `Scrape` (change `Scrape.End` to `Next`) and grant the needed
  permissions on `aws_iam_role.pipeline`.
- Backend: S3 with `use_lockfile`. The bucket is passed with `-backend-config`.

## Checks

`terraform fmt -check -recursive && terraform init -backend=false && terraform validate`
(run in both `infrastructure/` and `infrastructure/bootstrap/`).
