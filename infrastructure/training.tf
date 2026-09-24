# training component: two-tower model trained and evaluated by SageMaker training jobs (started
# manually by the training CD / promote workflows, outside the Step Functions pipeline). Model
# artifacts are plain S3 objects in `model-artifacts-<acct>` (layout: training/README.md).

locals {
  training_ssm_prefix = "/training"
  training_ssm_arns = [
    "arn:aws:ssm:${local.region}:${local.account_id}:parameter${local.training_ssm_prefix}",
    "arn:aws:ssm:${local.region}:${local.account_id}:parameter${local.training_ssm_prefix}/*",
  ]
  marts_database = aws_glue_catalog_database.etl["marts"].name
}

# ---- Model artifacts -----------------------------------------------------------------------

resource "aws_s3_bucket" "model_artifacts" {
  bucket = "model-artifacts-${local.account_id}"
}

# Versioned: overwriting models/champion/ keeps the previous champion recoverable.
resource "aws_s3_bucket_versioning" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "model_artifacts" {
  bucket     = aws_s3_bucket.model_artifacts.id
  depends_on = [aws_s3_bucket_versioning.model_artifacts]
  rule {
    id     = "expire-old-versions"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }
  # Checkpoints are deleted when training finishes; this only clears abandoned runs.
  rule {
    id     = "expire-abandoned-checkpoints"
    status = "Enabled"
    filter {
      prefix = "checkpoints/"
    }
    expiration {
      days = 14
    }
  }
  # SageMaker's own (empty) job outputs: the jobs write their artifacts under models/.
  rule {
    id     = "expire-sagemaker-output"
    status = "Enabled"
    filter {
      prefix = "sagemaker/"
    }
    expiration {
      days = 30
    }
  }
}

# ---- Config (SSM Parameter Store) -----------------------------------------------------------

resource "aws_ssm_parameter" "training" {
  for_each = {
    MODEL_ARTIFACTS_BUCKET    = aws_s3_bucket.model_artifacts.bucket
    GLUE_DATABASE             = local.marts_database
    SAGEMAKER_ROLE_ARN        = aws_iam_role.training.arn
    TRAINING_IMAGE_REPOSITORY = aws_ecr_repository.training.repository_url
    INSTANCE_TYPE             = var.training_instance_type
  }
  name  = "${local.training_ssm_prefix}/${each.key}"
  type  = "String"
  value = each.value
}

# ---- ECR -----------------------------------------------------------------------------------

resource "aws_ecr_repository" "training" {
  name                 = "training"
  image_tag_mutability = "MUTABLE" # CD pushes :<sha> and moves :latest
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "training" {
  repository = aws_ecr_repository.training.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the 30 most recent images (promotion runs a model's own :<sha>)"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 30 }
      action       = { type = "expire" }
    }]
  })
}

# ---- SageMaker execution role --------------------------------------------------------------

data "aws_iam_policy_document" "sagemaker_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }
  }
}

resource "aws_iam_role" "training" {
  name               = "${var.project}-training"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_trust.json
}

data "aws_iam_policy_document" "training" {
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
    sid       = "ReadWriteArtifacts"
    actions   = local.s3_rw_actions
    resources = ["${aws_s3_bucket.model_artifacts.arn}/*"]
  }
  statement {
    sid       = "EcrAuth"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    sid       = "EcrPull"
    actions   = ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"]
    resources = [aws_ecr_repository.training.arn]
  }
  statement {
    sid = "Logs"
    actions = [
      "logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams",
    ]
    resources = ["arn:aws:logs:${local.region}:${local.account_id}:log-group:/aws/sagemaker/TrainingJobs*"]
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
    resources = local.training_ssm_arns
  }
}

resource "aws_iam_role_policy" "training" {
  role   = aws_iam_role.training.id
  policy = data.aws_iam_policy_document.training.json
}
