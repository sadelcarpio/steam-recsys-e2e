# inference component: SageMaker Processing job, the last step (`Infer`) of the Step Functions
# pipeline (skipped while s3://model-artifacts-<acct>/models/champion/ is empty). Scores every
# user against every game with the champion two-tower model, reranks the top reviewers'
# candidates with a Bedrock LLM and writes one DynamoDB item per user whose recommendations
# changed, plus the popularity fallback item, the details of new games (game-details) and the
# online catalog for serving (s3://model-artifacts-<acct>/serving/online/).
# The image is shipped by the inference CD; the job itself is defined by the `Infer`
# state (orchestration.tf, `local.inference_job`).

locals {
  inference_ssm_prefix = "/inference"
  inference_ssm_arns = [
    "arn:aws:ssm:${local.region}:${local.account_id}:parameter${local.inference_ssm_prefix}",
    "arn:aws:ssm:${local.region}:${local.account_id}:parameter${local.inference_ssm_prefix}/*",
  ]
  # Cross-region inference profiles (us. / eu. / apac. / global.) route to the base model in
  # several regions: allow the profile and the base model everywhere.
  bedrock_base_model_id = replace(var.inference_bedrock_model_id, "/^(us|eu|apac|global)\\./", "")
  # Online catalog (item embeddings + manifest) written by inference, read by serving.
  online_bundle_prefix = "serving/online"
}

# ---- Output: recommendations table -----------------------------------------------------------

# One item per user (contract: inference/src/steam_inference/contracts.py). Each run scans the
# stored content hashes, writes only the changed users and deletes the users that are gone.
# No PITR: every item can be regenerated from the marts and the champion.
resource "aws_dynamodb_table" "recommendations" {
  name         = "game-explainable-recommendations"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "user_id"

  attribute {
    name = "user_id"
    type = "S"
  }
}

# Details of every catalog game (mart game_details), read by serving. Insert-only: each run
# writes only the games missing from the table (the first run loads the whole catalog).
resource "aws_dynamodb_table" "game_details" {
  name         = "game-details"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "game_id"

  attribute {
    name = "game_id"
    type = "N"
  }
}

# ---- Config (SSM Parameter Store) -----------------------------------------------------------

resource "aws_ssm_parameter" "inference" {
  for_each = {
    MODEL_ARTIFACTS_BUCKET = aws_s3_bucket.model_artifacts.bucket
    GLUE_DATABASE          = local.marts_database
    RECOMMENDATIONS_TABLE  = aws_dynamodb_table.recommendations.name
    GAME_DETAILS_TABLE     = aws_dynamodb_table.game_details.name
    ONLINE_BUNDLE_PREFIX   = local.online_bundle_prefix
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

# ---- SageMaker execution role (processing job) --------------------------------------------

resource "aws_iam_role" "inference" {
  name               = "${var.project}-inference"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_trust.json
}

data "aws_iam_policy_document" "inference" {
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
    sid       = "PublishOnlineBundle"
    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.model_artifacts.arn}/${local.online_bundle_prefix}/*"]
  }
  statement {
    sid       = "SyncRecommendations"
    actions   = ["dynamodb:Scan", "dynamodb:BatchWriteItem"]
    resources = [aws_dynamodb_table.recommendations.arn, aws_dynamodb_table.game_details.arn]
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
    sid       = "EcrAuth"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    sid       = "EcrPull"
    actions   = ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"]
    resources = [aws_ecr_repository.inference.arn]
  }
  statement {
    sid = "Logs"
    actions = [
      "logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams",
    ]
    resources = ["arn:aws:logs:${local.region}:${local.account_id}:log-group:/aws/sagemaker/ProcessingJobs*"]
  }
  statement {
    sid       = "Metrics"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"]
    condition {
      test     = "StringLike"
      variable = "cloudwatch:namespace"
      values   = ["/aws/sagemaker/*"]
    }
  }
  statement {
    sid       = "Config"
    actions   = ["ssm:GetParametersByPath"]
    resources = local.inference_ssm_arns
  }
}

resource "aws_iam_role_policy" "inference" {
  role   = aws_iam_role.inference.id
  policy = data.aws_iam_policy_document.inference.json
}
