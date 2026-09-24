# inference component: ECS task `inference`, the last step of the Step Functions pipeline (skipped
# while s3://model-artifacts-<acct>/models/champion/ is empty). Scores every user against every
# game with the champion two-tower model, reranks the top reviewers' candidates with a Bedrock
# LLM and overwrites one DynamoDB item per user. The image is shipped by the inference CD.

locals {
  inference_ssm_prefix = "/inference"
  inference_ssm_arns = [
    "arn:aws:ssm:${local.region}:${local.account_id}:parameter${local.inference_ssm_prefix}",
    "arn:aws:ssm:${local.region}:${local.account_id}:parameter${local.inference_ssm_prefix}/*",
  ]
  # Cross-region inference profiles (us. / eu. / apac. / global.) route to the base model in
  # several regions: allow the profile and the base model everywhere.
  bedrock_base_model_id = replace(var.inference_bedrock_model_id, "/^(us|eu|apac|global)\\./", "")
}

# ---- Output: recommendations table -----------------------------------------------------------

# One item per user (contract: inference/src/steam_inference/contracts.py), overwritten each
# run. No PITR: every item is regenerated weekly. The TTL clears users that stop appearing.
resource "aws_dynamodb_table" "recommendations" {
  name         = "game-explainable-recommendations"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "user_id"

  attribute {
    name = "user_id"
    type = "S"
  }

  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }
}

# ---- Config (SSM Parameter Store) -----------------------------------------------------------

resource "aws_ssm_parameter" "inference" {
  for_each = {
    MODEL_ARTIFACTS_BUCKET = aws_s3_bucket.model_artifacts.bucket
    GLUE_DATABASE          = local.marts_database
    RECOMMENDATIONS_TABLE  = aws_dynamodb_table.recommendations.name
    BEDROCK_MODEL_ID       = var.inference_bedrock_model_id
    RERANK_MAX_USERS       = tostring(var.inference_rerank_max_users)
  }
  name  = "${local.inference_ssm_prefix}/${each.key}"
  type  = "String"
  value = each.value
}

# ---- ECR -----------------------------------------------------------------------------------

resource "aws_ecr_repository" "inference" {
  name                 = "inference"
  image_tag_mutability = "MUTABLE" # CD pushes :<sha> and moves :latest
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "inference" {
  repository = aws_ecr_repository.inference.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the 20 most recent images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 20 }
      action       = { type = "expire" }
    }]
  })
}

# ---- ECS: inference ------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "inference" {
  name              = "/ecs/inference"
  retention_in_days = var.log_retention_days
}

resource "aws_iam_role" "inference_execution" {
  name               = "${var.project}-inference-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_trust.json
}

resource "aws_iam_role_policy_attachment" "inference_execution" {
  role       = aws_iam_role.inference_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "inference_task" {
  name               = "${var.project}-inference-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_trust.json
}

data "aws_iam_policy_document" "inference_task" {
  # pyiceberg reads the marts through the Glue catalog, straight from S3 (no Athena).
  statement {
    sid     = "GlueReadMarts"
    actions = ["glue:GetDatabase", "glue:GetTable", "glue:GetTables"]
    resources = [
      local.glue_catalog_arn,
      aws_glue_catalog_database.etl["marts"].arn,
      "arn:aws:glue:${local.region}:${local.account_id}:table/${local.marts_database}/*",
    ]
  }
  statement {
    sid       = "ListBuckets"
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = [aws_s3_bucket.processed_steam_data.arn, aws_s3_bucket.model_artifacts.arn]
  }
  statement {
    sid       = "ReadLake"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.processed_steam_data.arn}/iceberg/*"]
  }
  statement {
    sid       = "ReadModels"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.model_artifacts.arn}/models/*"]
  }
  statement {
    sid       = "WriteRecommendations"
    actions   = ["dynamodb:BatchWriteItem", "dynamodb:PutItem"]
    resources = [aws_dynamodb_table.recommendations.arn]
  }
  statement {
    sid     = "InvokeRerankModel"
    actions = ["bedrock:InvokeModel"]
    resources = [
      "arn:aws:bedrock:*::foundation-model/${local.bedrock_base_model_id}",
      "arn:aws:bedrock:${local.region}:${local.account_id}:inference-profile/${var.inference_bedrock_model_id}",
    ]
  }
  statement {
    sid       = "Config"
    actions   = ["ssm:GetParametersByPath"]
    resources = local.inference_ssm_arns
  }
}

resource "aws_iam_role_policy" "inference_task" {
  role   = aws_iam_role.inference_task.id
  policy = data.aws_iam_policy_document.inference_task.json
}

resource "aws_ecs_task_definition" "inference" {
  family                   = "inference"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.inference_cpu
  memory                   = var.inference_memory
  execution_role_arn       = aws_iam_role.inference_execution.arn
  task_role_arn            = aws_iam_role.inference_task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  container_definitions = jsonencode([{
    name      = "inference"
    image     = "${aws_ecr_repository.inference.repository_url}:${var.inference_image_tag}"
    essential = true
    command   = ["python", "-m", "steam_inference"]
    # Override e.g. RERANK_ENABLED=false or MODEL_ID=<sha> on a manual run-task.
    environment = [{ name = "USE_SSM", value = "true" }]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.inference.name
        awslogs-region        = local.region
        awslogs-stream-prefix = "inference"
      }
    }
  }])
}
