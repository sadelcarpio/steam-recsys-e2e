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

variable "github_repository_immutable" {
  description = "Same repo in GitHub's immutable OIDC subject form: owner@<owner_id>/repo@<repo_id>."
  type        = string
  default     = "sadelcarpio@70857703/steam-recsys-e2e@1384979753"
}

variable "create_github_oidc_provider" {
  description = "false when the account already has the token.actions.githubusercontent.com provider."
  type        = bool
  default     = false
}
