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
