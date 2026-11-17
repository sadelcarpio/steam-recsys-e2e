# Training Step Function (spec 9, phase 7): retriever -> ranker -> pair evaluation -> approval
# -> promotion -> inference, as one execution. Started by hand (or a disabled-by-default monthly
# schedule); the weekly data pipeline (orchestration.tf) is separate.
#
#   aws stepfunctions start-execution --state-machine-arn <arn> --name <name ≤ 40 chars> \
#     --input '{"retriever_image_tag": "<sha>-cu128", "retriever_instance_type": "ml.g4dn.xlarge"}'
#
# Every input key is optional (defaults: local.training_pipeline_defaults). Ids follow the
# execution name: retriever `ct-<name>`, ranker `ct-<name>-rank`, the dry-run report
# evaluation/pairs/runs/<name>.json. The approval step writes its task token to
# evaluation/pairs/runs/<name>.token and waits (7 days at most) for *training approve*.

locals {
  training_pipeline_name = "${var.project}-training"
  training_runs_prefix   = "evaluation/pairs/runs"
  training_image         = aws_ecr_repository.training.repository_url

  # Execution input overrides these (States.JsonMerge, shallow: an env map replaces the
  # default map, so pass the whole recipe when changing it).
  training_pipeline_defaults = {
    image_tag               = var.training_pipeline_image_tag
    retriever_image_tag     = var.training_pipeline_image_tag
    retriever_instance_type = var.training_instance_type
    cpu_instance_type       = var.training_pipeline_cpu_instance_type
    run_inference           = true
    # the champion's recipe (A3 len 20 attention + A4 tag profile), on time-based windows
    retriever_env = {
      HOLDOUT_WEEKS         = "6"
      TEST_WEEKS            = "2"
      HISTORY_LENGTH        = "20"
      HISTORY_ENCODER       = "attention"
      USER_TAG_PROFILE      = "true"
      EPOCHS                = "3"
      BATCH_SIZE            = "4096"
      LEARNING_RATE         = "0.005"
      MINED_NEGATIVES       = "false"
      USER_TABLE_SYNC_STEPS = "1000"
    }
    ranker_env = {}
    pair_env   = {}
  }

  # Set by the state machine; they win over the input's env maps.
  training_job_env = {
    USE_SSM                = "true"
    AWS_REGION             = local.region
    AWS_DEFAULT_REGION     = local.region
    MODEL_ARTIFACTS_BUCKET = aws_s3_bucket.model_artifacts.bucket
  }

  training_job_base = {
    RoleArn           = aws_iam_role.training.arn
    OutputDataConfig  = { S3OutputPath = "s3://${aws_s3_bucket.model_artifacts.bucket}/sagemaker/ct/" }
    StoppingCondition = { MaxRuntimeInSeconds = var.training_pipeline_max_runtime_seconds }
    Tags              = [{ Key = "Project", Value = var.project }]
  }

  training_job_retry = [
    {
      ErrorEquals     = ["SageMaker.AmazonSageMakerException", "ThrottlingException"]
      IntervalSeconds = 60
      MaxAttempts     = 3
      BackoffRate     = 2
    },
  ]

  # One SageMaker training job of the training image: `mode` = steam_training entry point,
  # `step` = job name suffix, `image` / `instance` / `env` = state paths.
  training_steps = {
    TrainRetriever = { mode = "train", step = "train", image = "$.config.retriever_image_tag", instance = "$.config.retriever_instance_type", env = "$.env.retriever", next = "TrainRanker" }
    TrainRanker    = { mode = "rank", step = "rank", image = "$.config.image_tag", instance = "$.config.cpu_instance_type", env = "$.env.ranker", next = "EvaluatePairs" }
    EvaluatePairs  = { mode = "pair", step = "eval", image = "$.config.image_tag", instance = "$.config.cpu_instance_type", env = "$.env.evaluate", next = "ReadDecision" }
    ApplyPromotion = { mode = "pair", step = "apply", image = "$.config.image_tag", instance = "$.config.cpu_instance_type", env = "$.env.apply", next = "RunInference?" }
  }

  training_pipeline_definition = {
    Comment = "Retriever -> ranker -> pair evaluation (dry run) -> approval -> promotion -> inference (spec 9)."
    StartAt = "Defaults"
    States = merge(
      {
        Defaults = {
          Type       = "Pass"
          Result     = local.training_pipeline_defaults
          ResultPath = "$.defaults"
          Next       = "Configure"
        }
        Configure = {
          Type       = "Pass"
          Parameters = { "config.$" = "States.JsonMerge($.defaults, $$.Execution.Input, false)" }
          Next       = "Ids"
        }
        Ids = {
          Type = "Pass"
          Parameters = {
            "retriever_id.$" = "States.Format('ct-{}', $$.Execution.Name)"
            "ranker_id.$"    = "States.Format('ct-{}-rank', $$.Execution.Name)"
            "report_key.$"   = "States.Format('${local.training_runs_prefix}/{}.json', $$.Execution.Name)"
            "token_key.$"    = "States.Format('${local.training_runs_prefix}/{}.token', $$.Execution.Name)"
          }
          ResultPath = "$.ids"
          Next       = "BaseEnv"
        }
        # the reserved settings of each job (they override the input's env maps)
        BaseEnv = {
          Type = "Pass"
          Parameters = {
            retriever = merge(local.training_job_env, {
              "MODEL_ID.$"   = "$.ids.retriever_id"
              "GIT_COMMIT.$" = "$.config.retriever_image_tag"
            })
            ranker = merge(local.training_job_env, {
              "MODEL_ID.$"     = "$.ids.ranker_id"
              "RETRIEVER_ID.$" = "$.ids.retriever_id"
            })
            evaluate = merge(local.training_job_env, {
              "RETRIEVER_ID.$"    = "$.ids.retriever_id"
              "RANKER_ID.$"       = "$.ids.ranker_id"
              "PAIR_DRY_RUN"      = "true"
              "PAIR_REPORT_KEY.$" = "$.ids.report_key"
            })
            apply = merge(local.training_job_env, {
              "APPLY_REPORT.$" = "$.ids.report_key"
            })
          }
          ResultPath = "$.base"
          Next       = "Env"
        }
        Env = {
          Type = "Pass"
          Parameters = {
            "retriever.$" = "States.JsonMerge($.config.retriever_env, $.base.retriever, false)"
            "ranker.$"    = "States.JsonMerge($.config.ranker_env, $.base.ranker, false)"
            "evaluate.$"  = "States.JsonMerge($.config.pair_env, $.base.evaluate, false)"
            "apply.$"     = "$.base.apply"
          }
          ResultPath = "$.env"
          Next       = "TrainRetriever"
        }
        # the dry-run report: `chosen` is the pair that would be promoted (null: none)
        ReadDecision = {
          Type     = "Task"
          Resource = "arn:aws:states:::aws-sdk:s3:getObject"
          Parameters = {
            Bucket  = aws_s3_bucket.model_artifacts.bucket
            "Key.$" = "$.ids.report_key"
          }
          ResultSelector = { "report.$" = "States.StringToJson($.Body)" }
          ResultPath     = "$.decision"
          Next           = "WouldPromote?"
        }
        "WouldPromote?" = {
          Type = "Choice"
          Choices = [{
            Variable = "$.decision.report.chosen"
            IsNull   = false
            Next     = "AwaitApproval"
          }]
          Default = "NothingToPromote"
        }
        NothingToPromote = { Type = "Succeed", Comment = "The challenger pairs lost: the serving pair stays." }
        # The task token goes to S3; *training approve* (GitHub, prod environment) sends it back
        # with success (promote) or failure (reject).
        AwaitApproval = {
          Type           = "Task"
          Resource       = "arn:aws:states:::aws-sdk:s3:putObject.waitForTaskToken"
          TimeoutSeconds = 7 * 24 * 3600
          Parameters = {
            Bucket   = aws_s3_bucket.model_artifacts.bucket
            "Key.$"  = "$.ids.token_key"
            "Body.$" = "$$.Task.Token"
          }
          ResultPath = "$.approval"
          Catch = [{
            ErrorEquals = ["States.ALL"]
            ResultPath  = "$.approval_error"
            Next        = "NotApproved"
          }]
          Next = "ApplyPromotion"
        }
        NotApproved = { Type = "Succeed", Comment = "Rejected or timed out: nothing promoted." }
        "RunInference?" = {
          Type = "Choice"
          Choices = [{
            Variable      = "$.config.run_inference"
            BooleanEquals = true
            Next          = "RunInference"
          }]
          Default = "Done"
        }
        RunInference = {
          Type     = "Task"
          Resource = "arn:aws:states:::sagemaker:createProcessingJob.sync"
          Parameters = merge(local.inference_job, {
            "ProcessingJobName.$" = "States.Format('steam-recsys-infer-{}', States.UUID())"
          })
          Retry      = local.training_job_retry
          ResultPath = null
          Next       = "Done"
        }
        Done = { Type = "Succeed" }
      },
      {
        for state, s in local.training_steps : state => {
          Type     = "Task"
          Resource = "arn:aws:states:::sagemaker:createTrainingJob.sync"
          Parameters = merge(local.training_job_base, {
            "TrainingJobName.$" = "States.Format('${var.project}-ct-{}-${s.step}', $$.Execution.Name)"
            AlgorithmSpecification = {
              "TrainingImage.$"   = "States.Format('${local.training_image}:{}', ${s.image})"
              TrainingInputMode   = "File"
              ContainerEntrypoint = ["python", "-m", "steam_training", s.mode]
            }
            ResourceConfig = {
              "InstanceType.$" = s.instance
              InstanceCount    = 1
              VolumeSizeInGB   = 30
            }
            "Environment.$" = s.env
          })
          Retry      = local.training_job_retry
          ResultPath = null
          Next       = s.next
        }
      },
    )
  }
}

resource "aws_cloudwatch_log_group" "training_pipeline" {
  name              = "/aws/vendedlogs/states/${local.training_pipeline_name}"
  retention_in_days = 30
}

resource "aws_iam_role" "training_pipeline" {
  name               = "${var.project}-training-pipeline"
  assume_role_policy = data.aws_iam_policy_document.states_trust.json
}

data "aws_iam_policy_document" "training_pipeline" {
  statement {
    sid       = "RunTrainingJobs"
    actions   = ["sagemaker:CreateTrainingJob", "sagemaker:DescribeTrainingJob", "sagemaker:StopTrainingJob", "sagemaker:AddTags"]
    resources = ["arn:aws:sagemaker:${local.region}:${local.account_id}:training-job/${var.project}-ct-*"]
  }
  statement {
    sid       = "RunInferenceJob"
    actions   = ["sagemaker:CreateProcessingJob", "sagemaker:DescribeProcessingJob", "sagemaker:StopProcessingJob", "sagemaker:AddTags"]
    resources = ["arn:aws:sagemaker:${local.region}:${local.account_id}:processing-job/steam-recsys-infer-*"]
  }
  statement {
    sid       = "PassJobRoles"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.training.arn, aws_iam_role.inference.arn]
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["sagemaker.amazonaws.com"]
    }
  }
  statement {
    sid     = "SageMakerSyncRules"
    actions = ["events:PutTargets", "events:PutRule", "events:DescribeRule"]
    resources = [
      "arn:aws:events:${local.region}:${local.account_id}:rule/StepFunctionsGetEventsForSageMakerTrainingJobsRule",
      "arn:aws:events:${local.region}:${local.account_id}:rule/StepFunctionsGetEventsForSageMakerProcessingJobsRule",
    ]
  }
  statement {
    sid       = "DecisionAndApproval"
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${aws_s3_bucket.model_artifacts.arn}/${local.training_runs_prefix}/*"]
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

resource "aws_iam_role_policy" "training_pipeline" {
  role   = aws_iam_role.training_pipeline.id
  policy = data.aws_iam_policy_document.training_pipeline.json
}

resource "aws_sfn_state_machine" "training_pipeline" {
  name       = local.training_pipeline_name
  role_arn   = aws_iam_role.training_pipeline.arn
  definition = jsonencode(local.training_pipeline_definition)

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.training_pipeline.arn}:*"
    include_execution_data = false
    level                  = "ERROR"
  }

  depends_on = [aws_iam_role_policy.training_pipeline]
}

# ---- Optional monthly schedule (continuous training, phase 8 builds on it) -----------------

resource "aws_iam_role" "training_scheduler" {
  name               = "${var.project}-training-scheduler"
  assume_role_policy = data.aws_iam_policy_document.scheduler_trust.json
}

resource "aws_iam_role_policy" "training_scheduler" {
  role = aws_iam_role.training_scheduler.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "states:StartExecution"
      Resource = aws_sfn_state_machine.training_pipeline.arn
    }]
  })
}

resource "aws_scheduler_schedule" "training_pipeline" {
  name                         = "${var.project}-training-monthly"
  description                  = "Monthly retriever + ranker retraining with an approval before promotion (spec 9)."
  state                        = var.training_pipeline_schedule_enabled ? "ENABLED" : "DISABLED"
  schedule_expression          = var.training_pipeline_schedule_expression
  schedule_expression_timezone = var.schedule_timezone

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = aws_sfn_state_machine.training_pipeline.arn
    role_arn = aws_iam_role.training_scheduler.arn
    input    = jsonencode({})
  }
}
