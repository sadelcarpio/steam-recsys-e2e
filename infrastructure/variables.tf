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
  description = "Max new appids per games-scraping task (~200 req / 5 min per IP => ~3.5 h)."
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
