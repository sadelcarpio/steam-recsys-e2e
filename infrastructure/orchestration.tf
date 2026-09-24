# Step Functions pipeline (EventBridge Scheduler -> state machine). Later specs append the
# dbt / inference steps after `Scrape`.
#
# Input: {} (run_id defaults to the execution name) or {"run_id": "<id>"} to re-run a
# previous run's partitions idempotently.

locals {
  state_machine_name = "${var.project}-pipeline"
  state_machine_arn  = "arn:aws:states:${local.region}:${local.account_id}:stateMachine:${local.state_machine_name}"

  scrape_maps = {
    games = {
      state           = "ScrapeGames"
      task            = local.scraping_tasks.games.name
      max_concurrency = var.max_games_tasks
    }
    reviews = {
      state           = "ScrapeReviews"
      task            = local.scraping_tasks.reviews.name
      max_concurrency = var.num_review_workers
    }
  }

  # One Distributed Map per scraper over the partition files the Lambda wrote for this run;
  # each item (S3 object) becomes one Fargate task with its own public IP.
  scrape_branches = [for kind, m in local.scrape_maps : {
    StartAt = m.state
    States = {
      (m.state) = {
        Type = "Map"
        ItemReader = {
          Resource = "arn:aws:states:::s3:listObjectsV2"
          Parameters = {
            Bucket     = aws_s3_bucket.game_partitions.bucket
            "Prefix.$" = "States.Format('${kind}/{}/', $.run_id)"
          }
        }
        ItemSelector = {
          "partition_key.$" = "$$.Map.Item.Value.Key"
        }
        MaxConcurrency = m.max_concurrency
        ItemProcessor = {
          ProcessorConfig = { Mode = "DISTRIBUTED", ExecutionType = "STANDARD" }
          StartAt         = "Run${m.state}Task"
          States = {
            "Run${m.state}Task" = {
              Type     = "Task"
              Resource = "arn:aws:states:::ecs:runTask.sync"
              Parameters = {
                LaunchType     = "FARGATE"
                Cluster        = aws_ecs_cluster.main.arn
                TaskDefinition = aws_ecs_task_definition.scraping[kind].arn_without_revision
                NetworkConfiguration = {
                  AwsvpcConfiguration = {
                    Subnets        = aws_subnet.public[*].id
                    SecurityGroups = [aws_security_group.egress_only.id]
                    AssignPublicIp = "ENABLED"
                  }
                }
                Overrides = {
                  ContainerOverrides = [{
                    Name        = m.task
                    Environment = [{ Name = "PARTITION_KEY", "Value.$" = "$.partition_key" }]
                  }]
                }
                PropagateTags = "TASK_DEFINITION"
              }
              # Re-running a partition is safe: output parts never overwrite and cursors only
              # move forward. Wait out a possible Steam IP throttle before retrying.
              Retry = [
                {
                  ErrorEquals     = ["ECS.AmazonECSException", "ECS.AccessDeniedException"]
                  IntervalSeconds = 30
                  MaxAttempts     = 3
                  BackoffRate     = 2
                },
                {
                  ErrorEquals     = ["States.TaskFailed"]
                  IntervalSeconds = 300
                  MaxAttempts     = 1
                },
              ]
              ResultPath = null
              End        = true
            }
          }
        }
        End = true
      }
    }
  }]

  pipeline_definition = {
    Comment = "Steam RecSys batch pipeline: ingestion"
    StartAt = "ResolveRunId"
    States = {
      ResolveRunId = {
        Type    = "Choice"
        Choices = [{ Variable = "$.run_id", IsPresent = true, Next = "ListPartitionGameIds" }]
        Default = "DefaultRunId"
      }
      DefaultRunId = {
        Type       = "Pass"
        Parameters = { "run_id.$" = "$$.Execution.Name" }
        Next       = "ListPartitionGameIds"
      }
      ListPartitionGameIds = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.list_partition_game_ids.arn
          Payload      = { "run_id.$" = "$.run_id" }
        }
        ResultSelector = { "result.$" = "$.Payload" }
        ResultPath     = "$.partitions"
        Retry = [{
          ErrorEquals     = ["Lambda.ServiceException", "Lambda.AWSLambdaException", "Lambda.SdkClientException", "Lambda.TooManyRequestsException", "States.Timeout"]
          IntervalSeconds = 10
          MaxAttempts     = 3
          BackoffRate     = 2
        }]
        Next = "Scrape"
      }
      Scrape = {
        Type       = "Parallel"
        Branches   = local.scrape_branches
        ResultPath = null
        End        = true
      }
    }
  }
}

resource "aws_cloudwatch_log_group" "pipeline" {
  name              = "/aws/vendedlogs/states/${local.state_machine_name}"
  retention_in_days = var.log_retention_days
}

data "aws_iam_policy_document" "states_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["states.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "pipeline" {
  name               = "${var.project}-pipeline"
  assume_role_policy = data.aws_iam_policy_document.states_trust.json
}

data "aws_iam_policy_document" "pipeline" {
  statement {
    sid       = "InvokeLambda"
    actions   = ["lambda:InvokeFunction"]
    resources = [aws_lambda_function.list_partition_game_ids.arn, "${aws_lambda_function.list_partition_game_ids.arn}:*"]
  }
  statement {
    sid       = "RunScrapingTasks"
    actions   = ["ecs:RunTask"]
    resources = [for td in aws_ecs_task_definition.scraping : "${td.arn_without_revision}:*"]
  }
  statement {
    sid       = "TrackScrapingTasks"
    actions   = ["ecs:StopTask", "ecs:DescribeTasks"]
    resources = ["arn:aws:ecs:${local.region}:${local.account_id}:task/${aws_ecs_cluster.main.name}/*"]
  }
  statement {
    sid       = "PassTaskRoles"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.scraping_task.arn, aws_iam_role.scraping_execution.arn]
  }
  statement {
    sid       = "EcsSyncRule"
    actions   = ["events:PutTargets", "events:PutRule", "events:DescribeRule"]
    resources = ["arn:aws:events:${local.region}:${local.account_id}:rule/StepFunctionsGetEventsForECSTaskRule"]
  }
  statement {
    sid       = "ReadPartitionListing"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.game_partitions.arn]
  }
  statement {
    sid       = "DistributedMapChildren"
    actions   = ["states:StartExecution"]
    resources = [local.state_machine_arn]
  }
  statement {
    sid     = "DistributedMapChildrenTracking"
    actions = ["states:DescribeExecution", "states:StopExecution"]
    resources = [
      "arn:aws:states:${local.region}:${local.account_id}:execution:${local.state_machine_name}/*",
      "arn:aws:states:${local.region}:${local.account_id}:execution:${local.state_machine_name}:*",
    ]
  }
  statement {
    sid = "Logging"
    actions = [
      "logs:CreateLogDelivery", "logs:GetLogDelivery", "logs:UpdateLogDelivery",
      "logs:DeleteLogDelivery", "logs:ListLogDeliveries", "logs:PutResourcePolicy",
      "logs:DescribeResourcePolicies", "logs:DescribeLogGroups",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "pipeline" {
  role   = aws_iam_role.pipeline.id
  policy = data.aws_iam_policy_document.pipeline.json
}

resource "aws_sfn_state_machine" "pipeline" {
  name       = local.state_machine_name
  role_arn   = aws_iam_role.pipeline.arn
  definition = jsonencode(local.pipeline_definition)

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.pipeline.arn}:*"
    include_execution_data = false
    level                  = "ERROR"
  }

  depends_on = [aws_iam_role_policy.pipeline]
}

# ---- EventBridge Scheduler -----------------------------------------------------------------

data "aws_iam_policy_document" "scheduler_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["scheduler.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }
  }
}

resource "aws_iam_role" "scheduler" {
  name               = "${var.project}-scheduler"
  assume_role_policy = data.aws_iam_policy_document.scheduler_trust.json
}

resource "aws_iam_role_policy" "scheduler" {
  role = aws_iam_role.scheduler.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "states:StartExecution"
      Resource = aws_sfn_state_machine.pipeline.arn
    }]
  })
}

resource "aws_scheduler_schedule" "pipeline" {
  name                         = "${var.project}-weekly"
  description                  = "Weekly Steam ingestion -> recommendations pipeline."
  state                        = var.schedule_enabled ? "ENABLED" : "DISABLED"
  schedule_expression          = var.schedule_expression
  schedule_expression_timezone = var.schedule_timezone

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = aws_sfn_state_machine.pipeline.arn
    role_arn = aws_iam_role.scheduler.arn
    input    = jsonencode({})
  }
}
