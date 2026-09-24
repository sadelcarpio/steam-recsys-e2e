# etl component: dbt on Athena (ECS task `dbt`), raw Glue tables over the scraper output, Iceberg
# data lake in `processed-steam-data-<acct>`, and the PR-scoped role + sandbox used by the etl CI
# workflow to run the dbt project on real Athena. The image is shipped by the etl CD workflow.

locals {
  etl_ssm_prefix = "/etl"
  etl_ssm_arns = [
    "arn:aws:ssm:${local.region}:${local.account_id}:parameter${local.etl_ssm_prefix}",
    "arn:aws:ssm:${local.region}:${local.account_id}:parameter${local.etl_ssm_prefix}/*",
  ]
  # Glue databases: <schema>_raw (Terraform) + the dbt layers (<schema>_<layer>).
  etl_schema    = "steam"
  etl_databases = ["raw", "staging", "intermediate", "marts"]

  athena_catalog_arn = "arn:aws:athena:${local.region}:${local.account_id}:datacatalog/AwsDataCatalog"
  glue_catalog_arn   = "arn:aws:glue:${local.region}:${local.account_id}:catalog"

  # Contract of the scraper output: data_ingestion/src/steam_ingestion/schemas.py.
  raw_tables = {
    games = [
      ["appid", "bigint"], ["name", "string"], ["type", "string"], ["required_age", "bigint"],
      ["is_free", "boolean"], ["minimum_pc_requirements", "string"],
      ["recommended_pc_requirements", "string"], ["controller_support", "string"],
      ["detailed_description", "string"], ["about_the_game", "string"],
      ["short_description", "string"], ["supported_languages", "array<string>"],
      ["header_image", "string"], ["developers", "array<string>"],
      ["publishers", "array<string>"], ["price", "double"], ["categories", "array<string>"],
      ["genres", "array<string>"], ["windows_support", "boolean"], ["mac_support", "boolean"],
      ["linux_support", "boolean"], ["release_date", "string"], ["coming_soon", "boolean"],
      ["recommendations", "bigint"], ["dlc", "array<bigint>"], ["review_score", "bigint"],
      ["review_score_desc", "string"], ["scrape_date", "date"],
    ]
    reviews = [
      ["rec_id", "bigint"], ["author_id", "bigint"], ["appid", "bigint"],
      ["playtime_forever", "bigint"], ["playtime_last_two_weeks", "bigint"],
      ["playtime_at_review", "bigint"], ["num_games_owned", "bigint"], ["num_reviews", "bigint"],
      ["last_played", "bigint"], ["language", "string"], ["review", "string"],
      ["timestamp_created", "bigint"], ["timestamp_updated", "bigint"], ["voted_up", "boolean"],
      ["votes_up", "bigint"], ["votes_funny", "bigint"], ["weighted_vote_score", "double"],
      ["comment_count", "bigint"], ["steam_purchase", "boolean"], ["received_for_free", "boolean"],
      ["written_during_early_access", "boolean"], ["primarily_steam_deck", "boolean"],
      ["scrape_date", "date"],
    ]
  }

  # Athena + Glue permissions dbt-athena needs to build views / Iceberg tables in `databases`.
  dbt_glue_actions = [
    "glue:GetDatabase", "glue:GetDatabases", "glue:CreateDatabase",
    "glue:GetTable", "glue:GetTables", "glue:CreateTable", "glue:UpdateTable",
    "glue:DeleteTable", "glue:BatchDeleteTable", "glue:GetTableVersions",
    "glue:DeleteTableVersion", "glue:BatchDeleteTableVersion",
    "glue:GetPartition", "glue:GetPartitions", "glue:BatchGetPartition",
  ]
  dbt_athena_actions = [
    "athena:StartQueryExecution", "athena:GetQueryExecution", "athena:GetQueryResults",
    "athena:StopQueryExecution", "athena:GetWorkGroup", "athena:GetQueryRuntimeStatistics",
    "athena:GetDataCatalog", "athena:GetDatabase", "athena:ListDatabases",
    "athena:GetTableMetadata", "athena:ListTableMetadata",
  ]
  s3_rw_actions = [
    "s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload",
    "s3:ListMultipartUploadParts",
  ]
}

# ---- Iceberg data lake + Athena ------------------------------------------------------------

resource "aws_s3_bucket" "processed_steam_data" {
  bucket = "processed-steam-data-${local.account_id}"
}

resource "aws_s3_bucket_lifecycle_configuration" "processed_steam_data" {
  bucket = aws_s3_bucket.processed_steam_data.id
  rule {
    id     = "expire-athena-results"
    status = "Enabled"
    filter {
      prefix = "athena-results/"
    }
    expiration {
      days = 7
    }
  }
}

resource "aws_glue_catalog_database" "etl" {
  for_each = toset(local.etl_databases)
  name     = "${local.etl_schema}_${each.key}"
}

resource "aws_glue_catalog_table" "raw" {
  for_each      = local.raw_tables
  database_name = aws_glue_catalog_database.etl["raw"].name
  name          = each.key
  table_type    = "EXTERNAL_TABLE"
  parameters = {
    EXTERNAL         = "TRUE"
    "classification" = "parquet"
  }

  storage_descriptor {
    location      = "s3://${aws_s3_bucket.raw_steam_data.bucket}/${each.key}/"
    input_format  = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"
    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
    }
    dynamic "columns" {
      for_each = each.value
      content {
        name = columns.value[0]
        type = columns.value[1]
      }
    }
  }
}

resource "aws_athena_workgroup" "etl" {
  name          = "${var.project}-etl"
  force_destroy = true
  configuration {
    enforce_workgroup_configuration = true
    engine_version {
      selected_engine_version = "Athena engine version 3"
    }
    result_configuration {
      output_location = "s3://${aws_s3_bucket.processed_steam_data.bucket}/athena-results/"
      encryption_configuration {
        encryption_option = "SSE_S3"
      }
    }
  }
}

# ---- Config (SSM Parameter Store) -----------------------------------------------------------

resource "aws_ssm_parameter" "etl" {
  for_each = {
    ATHENA_WORK_GROUP     = aws_athena_workgroup.etl.name
    ATHENA_S3_STAGING_DIR = "s3://${aws_s3_bucket.processed_steam_data.bucket}/athena-results/"
    ICEBERG_S3_DATA_DIR   = "s3://${aws_s3_bucket.processed_steam_data.bucket}/iceberg/"
    DBT_SCHEMA            = local.etl_schema
    DBT_THREADS           = tostring(var.etl_dbt_threads)
  }
  name  = "${local.etl_ssm_prefix}/${each.key}"
  type  = "String"
  value = each.value
}

# ---- ECR -----------------------------------------------------------------------------------

resource "aws_ecr_repository" "etl" {
  name                 = "etl"
  image_tag_mutability = "MUTABLE" # CD pushes :<sha> and moves :latest
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "etl" {
  repository = aws_ecr_repository.etl.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the 20 most recent images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 20 }
      action       = { type = "expire" }
    }]
  })
}

# ---- ECS: dbt ------------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "etl" {
  name              = "/ecs/etl"
  retention_in_days = var.log_retention_days
}

resource "aws_iam_role" "etl_execution" {
  name               = "${var.project}-etl-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_trust.json
}

resource "aws_iam_role_policy_attachment" "etl_execution" {
  role       = aws_iam_role.etl_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "etl_task" {
  name               = "${var.project}-etl-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_trust.json
}

data "aws_iam_policy_document" "etl_task" {
  statement {
    sid       = "Athena"
    actions   = local.dbt_athena_actions
    resources = [aws_athena_workgroup.etl.arn, local.athena_catalog_arn]
  }
  statement {
    sid     = "Glue"
    actions = local.dbt_glue_actions
    resources = concat(
      [local.glue_catalog_arn],
      [for db in aws_glue_catalog_database.etl : db.arn],
      [for db in aws_glue_catalog_database.etl : "arn:aws:glue:${local.region}:${local.account_id}:table/${db.name}/*"],
    )
  }
  statement {
    sid       = "ListBuckets"
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = [aws_s3_bucket.raw_steam_data.arn, aws_s3_bucket.processed_steam_data.arn]
  }
  statement {
    sid       = "ReadRaw"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.raw_steam_data.arn}/*"]
  }
  statement {
    sid       = "ReadWriteLake"
    actions   = local.s3_rw_actions
    resources = ["${aws_s3_bucket.processed_steam_data.arn}/*"]
  }
  statement {
    sid       = "Config"
    actions   = ["ssm:GetParametersByPath"]
    resources = local.etl_ssm_arns
  }
}

resource "aws_iam_role_policy" "etl_task" {
  role   = aws_iam_role.etl_task.id
  policy = data.aws_iam_policy_document.etl_task.json
}

resource "aws_ecs_task_definition" "dbt" {
  family                   = "dbt"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.etl_execution.arn
  task_role_arn            = aws_iam_role.etl_task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  container_definitions = jsonencode([{
    name      = "dbt"
    image     = "${aws_ecr_repository.etl.repository_url}:${var.etl_image_tag}"
    essential = true
    command   = ["python", "-m", "steam_etl"]
    # Override FULL_REFRESH=true on a manual run-task to rebuild everything but the lookups.
    environment = [{ name = "USE_SSM", value = "true" }]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.etl.name
        awslogs-region        = local.region
        awslogs-stream-prefix = "dbt"
      }
    }
  }])
}

# ---- CI sandbox: PR workflows run the dbt project on real Athena ----------------------------
# Only `ci_*` Glue databases, the CI bucket and the CI workgroup: CI never touches prod data.

resource "aws_s3_bucket" "etl_ci" {
  bucket        = "etl-ci-${local.account_id}"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "etl_ci" {
  bucket                  = aws_s3_bucket.etl_ci.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "etl_ci" {
  bucket = aws_s3_bucket.etl_ci.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Safety net for runs whose teardown did not happen (cancelled jobs).
resource "aws_s3_bucket_lifecycle_configuration" "etl_ci" {
  bucket = aws_s3_bucket.etl_ci.id
  rule {
    id     = "expire-ci-runs"
    status = "Enabled"
    filter {}
    expiration {
      days = 2
    }
  }
}

resource "aws_athena_workgroup" "etl_ci" {
  name          = "${var.project}-etl-ci"
  force_destroy = true
  configuration {
    enforce_workgroup_configuration = true
    bytes_scanned_cutoff_per_query  = 1073741824 # 1 GiB: fixtures are tiny
    engine_version {
      selected_engine_version = "Athena engine version 3"
    }
    result_configuration {
      output_location = "s3://${aws_s3_bucket.etl_ci.bucket}/athena-results/"
    }
  }
}

data "aws_iam_policy_document" "etl_ci_trust" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = ["arn:aws:iam::${local.account_id}:oidc-provider/token.actions.githubusercontent.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    # Pull requests (same-repo branches: forks get no OIDC token) and pushes to main.
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values = flatten([
        for repo in [var.github_repository, var.github_repository_immutable] : [
          "repo:${repo}:pull_request",
          "repo:${repo}:ref:refs/heads/main",
        ]
      ])
    }
  }
}

resource "aws_iam_role" "etl_ci" {
  name                 = "${var.project}-etl-ci"
  assume_role_policy   = data.aws_iam_policy_document.etl_ci_trust.json
  max_session_duration = 3600
}

data "aws_iam_policy_document" "etl_ci" {
  statement {
    sid       = "Athena"
    actions   = local.dbt_athena_actions
    resources = [aws_athena_workgroup.etl_ci.arn, local.athena_catalog_arn]
  }
  statement {
    sid     = "Glue"
    actions = concat(local.dbt_glue_actions, ["glue:DeleteDatabase"])
    resources = [
      local.glue_catalog_arn,
      "arn:aws:glue:${local.region}:${local.account_id}:database/ci_*",
      "arn:aws:glue:${local.region}:${local.account_id}:table/ci_*/*",
      "arn:aws:glue:${local.region}:${local.account_id}:userDefinedFunction/ci_*/*",
    ]
  }
  statement {
    sid       = "ListBucket"
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = [aws_s3_bucket.etl_ci.arn]
  }
  statement {
    sid       = "ReadWrite"
    actions   = local.s3_rw_actions
    resources = ["${aws_s3_bucket.etl_ci.arn}/*"]
  }
}

resource "aws_iam_role_policy" "etl_ci" {
  role   = aws_iam_role.etl_ci.id
  policy = data.aws_iam_policy_document.etl_ci.json
}
