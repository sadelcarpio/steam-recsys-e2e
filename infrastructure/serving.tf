# serving component: Lambda `recsys-serving` behind a Lambda Function URL (no API Gateway / LB).
# Reads the recommendations table (+ game-details) written by the inference pipeline, and runs
# the user tower in numpy for POST /recommendations: the model's models/<id>/user_tower.npz
# (training) over the current catalog embeddings in serving/online/ (inference). The auth
# type is `var.serving_auth_type` (infrastructure CD input): AWS_IAM (SigV4, callers need
# lambda:InvokeFunctionUrl + lambda:InvokeFunction, e.g. via the steam-recsys-serving-client
# role) or NONE (public demo). The code is shipped by the serving CD (zip).

locals {
  serving_ssm_prefix = "/serving"
  serving_ssm_arns = [
    "arn:aws:ssm:${local.region}:${local.account_id}:parameter${local.serving_ssm_prefix}",
    "arn:aws:ssm:${local.region}:${local.account_id}:parameter${local.serving_ssm_prefix}/*",
  ]
  serving_public = var.serving_auth_type == "NONE"
}

# ---- Config (SSM Parameter Store) -----------------------------------------------------------

resource "aws_ssm_parameter" "serving" {
  for_each = {
    RECOMMENDATIONS_TABLE  = aws_dynamodb_table.recommendations.name
    GAME_DETAILS_TABLE     = aws_dynamodb_table.game_details.name
    MODEL_ARTIFACTS_BUCKET = aws_s3_bucket.model_artifacts.bucket
    ONLINE_BUNDLE_PREFIX   = local.online_bundle_prefix
  }
  name  = "${local.serving_ssm_prefix}/${each.key}"
  type  = "String"
  value = each.value
}

# ---- Lambda --------------------------------------------------------------------------------

resource "aws_iam_role" "serving" {
  name               = "${var.project}-serving"
  assume_role_policy = data.aws_iam_policy_document.lambda_trust.json
}

data "aws_iam_policy_document" "serving" {
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.serving.arn}:*"]
  }
  statement {
    sid       = "ReadRecommendations"
    actions   = ["dynamodb:GetItem"]
    resources = [aws_dynamodb_table.recommendations.arn]
  }
  statement {
    sid       = "ReadGameDetails"
    actions   = ["dynamodb:GetItem", "dynamodb:BatchGetItem"]
    resources = [aws_dynamodb_table.game_details.arn]
  }
  # Online model: the catalog + manifest (inference) and each model's numpy user tower (training).
  statement {
    sid       = "ReadOnlineCatalog"
    actions   = ["s3:GetObject", "s3:GetObjectVersion"]
    resources = ["${aws_s3_bucket.model_artifacts.arn}/${local.online_bundle_prefix}/*"]
  }
  statement {
    sid       = "ReadUserTowers"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.model_artifacts.arn}/models/*/user_tower.npz"]
  }
  # Without ListBucket, S3 answers GetObject on a missing key with AccessDenied instead of
  # NoSuchKey, so "no catalog published yet" (a 503) became a 500.
  statement {
    sid       = "ListModelArtifacts"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.model_artifacts.arn]
  }
  statement {
    sid       = "Config"
    actions   = ["ssm:GetParametersByPath"]
    resources = local.serving_ssm_arns
  }
}

resource "aws_iam_role_policy" "serving" {
  role   = aws_iam_role.serving.id
  policy = data.aws_iam_policy_document.serving.json
}

resource "aws_cloudwatch_log_group" "serving" {
  name              = "/aws/lambda/recsys-serving"
  retention_in_days = var.log_retention_days
}

# Placeholder so the function exists before the first CD run; CD replaces the code.
data "archive_file" "serving_placeholder" {
  type        = "zip"
  output_path = "${path.module}/.build/serving_placeholder.zip"
  source {
    filename = "steam_serving/handler.py"
    content  = "def handler(event, context):\n    return {'statusCode': 503, 'body': '{\"error\": \"code not deployed yet: run the serving CD workflow\"}'}\n"
  }
}

resource "aws_lambda_function" "serving" {
  function_name    = "recsys-serving"
  role             = aws_iam_role.serving.arn
  runtime          = "python3.12"
  architectures    = ["x86_64"]
  handler          = "steam_serving.handler.handler"
  filename         = data.archive_file.serving_placeholder.output_path
  source_code_hash = data.archive_file.serving_placeholder.output_base64sha256
  # 1 GB: the online bundle (~28 MB) stays in memory, and CPU scales with memory (numpy).
  memory_size = 1024
  timeout     = 15
  # Caps the cost of a public URL / site. Needs an account limit above it + 10 (this account has
  # 400; new accounts start at 10: docs/deployment.md, step 11). -1 = unreserved.
  reserved_concurrent_executions = var.serving_reserved_concurrency

  environment {
    variables = { USE_SSM = "true" }
  }

  depends_on = [aws_cloudwatch_log_group.serving]

  lifecycle {
    ignore_changes = [filename, source_code_hash]
  }
}

resource "aws_lambda_function_url" "serving" {
  function_name      = aws_lambda_function.serving.function_name
  authorization_type = var.serving_auth_type

  cors {
    allow_origins = var.serving_cors_allow_origins
    allow_methods = ["GET", "POST"]
    allow_headers = ["authorization", "content-type", "x-amz-date", "x-amz-security-token", "x-amz-content-sha256"]
    max_age       = 3600
  }
}

# ---- Public access (auth type NONE) ---------------------------------------------------------

# A public Function URL needs both statements: lambda:InvokeFunctionUrl for the URL and
# lambda:InvokeFunction restricted to calls through a Function URL.
resource "aws_lambda_permission" "serving_public_url" {
  count                  = local.serving_public ? 1 : 0
  statement_id           = "FunctionUrlAllowPublicAccess"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.serving.function_name
  principal              = "*"
  function_url_auth_type = "NONE"
}

resource "aws_lambda_permission" "serving_public_invoke" {
  count                    = local.serving_public ? 1 : 0
  statement_id             = "FunctionUrlAllowPublicInvoke"
  action                   = "lambda:InvokeFunction"
  function_name            = aws_lambda_function.serving.function_name
  principal                = "*"
  invoked_via_function_url = true
}

# ---- IAM access (auth type AWS_IAM) ---------------------------------------------------------

# Callers sign requests (SigV4, service `lambda`) with credentials allowed to invoke the URL:
# assume the client role below (any principal of this account that may call sts:AssumeRole on
# it), or grant the same statements to another principal. Inline policy: the GitHub deploy role
# may only manage steam-recsys-* roles, not managed policies (bootstrap/).
data "aws_iam_policy_document" "serving_invoke" {
  statement {
    sid       = "InvokeServingUrl"
    actions   = ["lambda:InvokeFunctionUrl"]
    resources = [aws_lambda_function.serving.arn]
    condition {
      test     = "StringEquals"
      variable = "lambda:FunctionUrlAuthType"
      values   = ["AWS_IAM"]
    }
  }
  statement {
    sid       = "InvokeServing"
    actions   = ["lambda:InvokeFunction"]
    resources = [aws_lambda_function.serving.arn]
  }
}

data "aws_iam_policy_document" "serving_client_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::${local.account_id}:root"]
    }
  }
}

resource "aws_iam_role" "serving_client" {
  name               = "${var.project}-serving-client"
  assume_role_policy = data.aws_iam_policy_document.serving_client_trust.json
}

resource "aws_iam_role_policy" "serving_client" {
  role   = aws_iam_role.serving_client.id
  policy = data.aws_iam_policy_document.serving_invoke.json
}
