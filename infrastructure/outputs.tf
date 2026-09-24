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
