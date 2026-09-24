output "raw_steam_data_bucket" {
  value = aws_s3_bucket.raw_steam_data.bucket
}

output "game_partitions_bucket" {
  value = aws_s3_bucket.game_partitions.bucket
}

output "data_ingestion_ecr_repository_url" {
  value = aws_ecr_repository.data_ingestion.repository_url
}

output "list_partition_game_ids_function" {
  value = aws_lambda_function.list_partition_game_ids.function_name
}

output "pipeline_state_machine_arn" {
  value = aws_sfn_state_machine.pipeline.arn
}

output "processed_steam_data_bucket" {
  value = aws_s3_bucket.processed_steam_data.bucket
}

output "etl_ecr_repository_url" {
  value = aws_ecr_repository.etl.repository_url
}

output "etl_ci_role_arn" {
  description = "Set as the ETL_CI_ROLE_ARN GitHub repository variable."
  value       = aws_iam_role.etl_ci.arn
}

output "etl_ci_bucket" {
  description = "Set as the ETL_CI_BUCKET GitHub repository variable."
  value       = aws_s3_bucket.etl_ci.bucket
}

output "etl_ci_work_group" {
  description = "Set as the ETL_CI_WORK_GROUP GitHub repository variable."
  value       = aws_athena_workgroup.etl_ci.name
}

output "model_artifacts_bucket" {
  value = aws_s3_bucket.model_artifacts.bucket
}

output "training_ecr_repository_url" {
  value = aws_ecr_repository.training.repository_url
}

output "training_role_arn" {
  description = "SageMaker execution role of the training / promote jobs."
  value       = aws_iam_role.training.arn
}

output "inference_ecr_repository_url" {
  value = aws_ecr_repository.inference.repository_url
}

output "recommendations_table" {
  value = aws_dynamodb_table.recommendations.name
}

output "public_subnet_ids" {
  description = "Subnets for a manual `aws ecs run-task` (dbt / inference)."
  value       = aws_subnet.public[*].id
}

output "egress_only_security_group_id" {
  value = aws_security_group.egress_only.id
}
