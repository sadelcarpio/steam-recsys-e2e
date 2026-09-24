# data_ingestion component: Lambda `list-partition-game-ids` + ECS tasks `games-scraping` and
# `reviews-scraping` (one shared image). Code/images are shipped by the data-ingestion CD
# workflow; Terraform owns everything else.

locals {
  ingestion_ssm_prefix = "/data-ingestion"
  ingestion_ssm_arns = [
    "arn:aws:ssm:${local.region}:${local.account_id}:parameter${local.ingestion_ssm_prefix}",
    "arn:aws:ssm:${local.region}:${local.account_id}:parameter${local.ingestion_ssm_prefix}/*",
  ]
  ingestion_state_tables = [
    aws_dynamodb_table.game_ids_state.arn,
    aws_dynamodb_table.reviews_state_cursor.arn,
  ]
  scraping_tasks = {
    games   = { name = "games-scraping", module = "steam_ingestion.games_scraping" }
    reviews = { name = "reviews-scraping", module = "steam_ingestion.reviews_scraping" }
  }
}

# ---- Config (SSM Parameter Store) + secrets -------------------------------------------------

resource "aws_ssm_parameter" "ingestion" {
  for_each = {
    RAW_BUCKET               = aws_s3_bucket.raw_steam_data.bucket
    PARTITIONS_BUCKET        = aws_s3_bucket.game_partitions.bucket
    GAME_IDS_TABLE           = aws_dynamodb_table.game_ids_state.name
    REVIEWS_CURSOR_TABLE     = aws_dynamodb_table.reviews_state_cursor.name
    STEAM_API_KEY_SECRET_ID  = aws_secretsmanager_secret.steam_api_key.name
    NUM_REVIEW_WORKERS       = tostring(var.num_review_workers)
    GAMES_PER_TASK           = tostring(var.games_per_task)
    MAX_REVIEWS_PER_GAME     = tostring(var.max_reviews_per_game)
    REQUEST_INTERVAL_SECONDS = tostring(var.request_interval_seconds)
  }
  name  = "${local.ingestion_ssm_prefix}/${each.key}"
  type  = "String"
  value = each.value
}

resource "aws_secretsmanager_secret" "steam_api_key" {
  name                    = "data-ingestion/steam-api-key"
  description             = "Steam Web API key used by list-partition-game-ids (GetAppList)."
  recovery_window_in_days = 7
}

# Dummy value; set the real key out-of-band:
#   aws secretsmanager put-secret-value --secret-id data-ingestion/steam-api-key --secret-string '<key>'
resource "aws_secretsmanager_secret_version" "steam_api_key" {
  secret_id     = aws_secretsmanager_secret.steam_api_key.id
  secret_string = "REPLACE_ME"
  lifecycle {
    ignore_changes = [secret_string]
  }
}

# ---- ECR -----------------------------------------------------------------------------------

resource "aws_ecr_repository" "data_ingestion" {
  name                 = "data-ingestion"
  image_tag_mutability = "MUTABLE" # CD pushes :<sha> and moves :latest
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "data_ingestion" {
  repository = aws_ecr_repository.data_ingestion.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the 20 most recent images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 20 }
      action       = { type = "expire" }
    }]
  })
}

# ---- Lambda: list-partition-game-ids -------------------------------------------------------

data "aws_iam_policy_document" "lambda_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "list_partition_game_ids" {
  name               = "${var.project}-list-partition-game-ids"
  assume_role_policy = data.aws_iam_policy_document.lambda_trust.json
}

data "aws_iam_policy_document" "list_partition_game_ids" {
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.list_partition_game_ids.arn}:*"]
  }
  statement {
    sid = "GameIdsState"
    actions = [
      "dynamodb:Scan", "dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:BatchWriteItem",
    ]
    resources = [aws_dynamodb_table.game_ids_state.arn]
  }
  statement {
    sid       = "ReviewTotals"
    actions   = ["dynamodb:Scan"]
    resources = [aws_dynamodb_table.reviews_state_cursor.arn]
  }
  statement {
    sid       = "PartitionsList"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.game_partitions.arn]
  }
  statement {
    sid       = "PartitionsWrite"
    actions   = ["s3:PutObject", "s3:DeleteObject"]
    resources = ["${aws_s3_bucket.game_partitions.arn}/*"]
  }
  statement {
    sid       = "Config"
    actions   = ["ssm:GetParametersByPath"]
    resources = local.ingestion_ssm_arns
  }
  statement {
    sid       = "SteamApiKey"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.steam_api_key.arn]
  }
}

resource "aws_iam_role_policy" "list_partition_game_ids" {
  role   = aws_iam_role.list_partition_game_ids.id
  policy = data.aws_iam_policy_document.list_partition_game_ids.json
}

resource "aws_cloudwatch_log_group" "list_partition_game_ids" {
  name              = "/aws/lambda/list-partition-game-ids"
  retention_in_days = var.log_retention_days
}

# Placeholder so the function exists before the first CD run; CD replaces the code.
data "archive_file" "lambda_placeholder" {
  type        = "zip"
  output_path = "${path.module}/.build/lambda_placeholder.zip"
  source {
    filename = "steam_ingestion/list_partition_game_ids/handler.py"
    content  = "def handler(event, context):\n    raise RuntimeError('code not deployed yet: run the data-ingestion CD workflow')\n"
  }
}

resource "aws_lambda_function" "list_partition_game_ids" {
  function_name    = "list-partition-game-ids"
  role             = aws_iam_role.list_partition_game_ids.arn
  runtime          = "python3.12"
  architectures    = ["x86_64"]
  handler          = "steam_ingestion.list_partition_game_ids.handler.handler"
  filename         = data.archive_file.lambda_placeholder.output_path
  source_code_hash = data.archive_file.lambda_placeholder.output_base64sha256
  memory_size      = 1024
  timeout          = 900

  environment {
    variables = { USE_SSM = "true" }
  }

  depends_on = [aws_cloudwatch_log_group.list_partition_game_ids]

  lifecycle {
    ignore_changes = [filename, source_code_hash]
  }
}

# ---- ECS: games-scraping / reviews-scraping ------------------------------------------------

resource "aws_ecs_cluster" "main" {
  name = var.project
}

resource "aws_cloudwatch_log_group" "scraping" {
  name              = "/ecs/data-ingestion"
  retention_in_days = var.log_retention_days
}

data "aws_iam_policy_document" "ecs_tasks_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "scraping_execution" {
  name               = "${var.project}-scraping-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_trust.json
}

resource "aws_iam_role_policy_attachment" "scraping_execution" {
  role       = aws_iam_role.scraping_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "scraping_task" {
  name               = "${var.project}-scraping-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_trust.json
}

data "aws_iam_policy_document" "scraping_task" {
  statement {
    sid       = "ReadPartitions"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.game_partitions.arn}/*"]
  }
  statement {
    sid       = "ListRaw"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.raw_steam_data.arn]
  }
  statement {
    sid       = "WriteRaw"
    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.raw_steam_data.arn}/*"]
  }
  statement {
    sid = "ScrapeState"
    actions = [
      "dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem",
      "dynamodb:BatchGetItem", "dynamodb:BatchWriteItem",
    ]
    resources = local.ingestion_state_tables
  }
  statement {
    sid       = "Config"
    actions   = ["ssm:GetParametersByPath"]
    resources = local.ingestion_ssm_arns
  }
}

resource "aws_iam_role_policy" "scraping_task" {
  role   = aws_iam_role.scraping_task.id
  policy = data.aws_iam_policy_document.scraping_task.json
}

resource "aws_ecs_task_definition" "scraping" {
  for_each                 = local.scraping_tasks
  family                   = each.value.name
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 512
  memory                   = 2048
  execution_role_arn       = aws_iam_role.scraping_execution.arn
  task_role_arn            = aws_iam_role.scraping_task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  container_definitions = jsonencode([{
    name      = each.value.name
    image     = "${aws_ecr_repository.data_ingestion.repository_url}:${var.scraping_image_tag}"
    essential = true
    command   = ["python", "-m", each.value.module]
    # PARTITION_KEY is injected per task by the Step Functions Distributed Map.
    environment = [{ name = "USE_SSM", value = "true" }]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.scraping.name
        awslogs-region        = local.region
        awslogs-stream-prefix = each.value.name
      }
    }
  }])
}
