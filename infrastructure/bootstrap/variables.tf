variable "region" {
  type    = string
  default = "us-east-1"
}

variable "project" {
  type    = string
  default = "steam-recsys"
}

variable "github_repository" {
  description = "owner/repo allowed to assume the deploy role."
  type        = string
  default     = "sadelcarpio/steam-recsys-e2e"
}

variable "create_github_oidc_provider" {
  description = "false when the account already has the token.actions.githubusercontent.com provider."
  type        = bool
  default     = false
}
