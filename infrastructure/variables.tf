variable "region" {
  type    = string
  default = "us-east-1"
}

variable "project" {
  description = "Prefix for shared resources (IAM roles must start with it: see bootstrap/)."
  type        = string
  default     = "steam-recsys"
}

variable "vpc_cidr" {
  type    = string
  default = "10.20.0.0/16"
}

variable "log_retention_days" {
  type    = number
  default = 14
}

# ---- data_ingestion ----------------------------------------------------------------------

variable "scraping_image_tag" {
  description = "Tag of the data-ingestion image run by the scraping tasks (pushed by CD)."
  type        = string
  default     = "latest"
}

variable "num_review_workers" {
  description = "Review partitions written by the Lambda = reviews tasks run in parallel."
  type        = number
  default     = 10
}

variable "games_per_task" {
  description = "Max new appids per games-scraping task (2 req/game at ~200 req / 5 min per IP => ~7 h)."
  type        = number
  default     = 8000
}

variable "max_games_tasks" {
  description = "Max concurrent games-scraping tasks (only matters for the initial backfill)."
  type        = number
  default     = 20
}

variable "max_reviews_per_game" {
  description = "Newest reviews fetched per game per run; 0 = full history."
  type        = number
  default     = 2000
}

variable "request_interval_seconds" {
  type    = number
  default = 1.5
}

# ---- orchestration -----------------------------------------------------------------------

variable "schedule_expression" {
  description = "EventBridge Scheduler expression for the pipeline (Thursdays 17:00)."
  type        = string
  default     = "cron(0 17 ? * THU *)"
}

variable "schedule_timezone" {
  type    = string
  default = "America/Chicago"
}

variable "schedule_enabled" {
  type    = bool
  default = true
}

# ---- etl -----------------------------------------------------------------------------------

variable "etl_image_tag" {
  description = "Tag of the etl image run by the dbt task (pushed by CD)."
  type        = string
  default     = "latest"
}

variable "etl_dbt_threads" {
  description = "Concurrent Athena queries per dbt run."
  type        = number
  default     = 4
}

variable "github_repository" {
  description = "owner/repo whose PR workflows may assume the etl CI role (same as bootstrap/)."
  type        = string
  default     = "sadelcarpio/steam-recsys-e2e"
}

variable "github_repository_immutable" {
  description = "Same repo in GitHub's immutable OIDC subject form: owner@<owner_id>/repo@<repo_id>."
  type        = string
  default     = "sadelcarpio@70857703/steam-recsys-e2e@1384979753"
}

# ---- training ------------------------------------------------------------------------------

variable "training_instance_type" {
  description = "SageMaker instance of the training / promote jobs (needs a training job quota)."
  type        = string
  default     = "ml.m5.2xlarge"
}

# ---- inference -----------------------------------------------------------------------------

variable "inference_image_tag" {
  description = "Tag of the inference image run by the pipeline's Infer job (pushed by CD)."
  type        = string
  default     = "latest"
}

variable "inference_instance_type" {
  description = "SageMaker Processing instance of the inference job (ml.t3.xlarge: 4 vCPU / 16 GB, default quota 2)."
  type        = string
  default     = "ml.t3.xlarge"
}

variable "inference_max_runtime_seconds" {
  description = "The inference job is stopped (and the pipeline fails) after this long."
  type        = number
  default     = 14400
}

variable "inference_bedrock_model_id" {
  description = "Bedrock model (or cross-region inference profile) that reranks the candidates."
  type        = string
  default     = "us.amazon.nova-2-lite-v1:0"
}

variable "inference_rerank_max_users" {
  description = "Users reranked by the LLM per run (the most active ones); bounds the Bedrock cost."
  type        = number
  default     = 1000
}
