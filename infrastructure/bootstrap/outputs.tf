output "tf_state_bucket" {
  description = "Set as the TF_STATE_BUCKET GitHub repository variable."
  value       = aws_s3_bucket.tf_state.bucket
}

output "github_deploy_role_arn" {
  description = "Set as the AWS_ROLE_ARN variable of the `prod` GitHub environment."
  value       = aws_iam_role.github_deploy.arn
}

output "github_plan_role_arn" {
  description = "Set as the AWS_PLAN_ROLE_ARN GitHub repository variable."
  value       = aws_iam_role.github_plan.arn
}
