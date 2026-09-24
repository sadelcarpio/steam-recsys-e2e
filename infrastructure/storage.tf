# ---- S3 ------------------------------------------------------------------------------------

resource "aws_s3_bucket" "raw_steam_data" {
  bucket = "raw-steam-data-${local.account_id}"
}

resource "aws_s3_bucket" "game_partitions" {
  bucket = "game-partitions-${local.account_id}"
}

locals {
  data_buckets = {
    raw_steam_data  = aws_s3_bucket.raw_steam_data.id
    game_partitions = aws_s3_bucket.game_partitions.id
    processed_data  = aws_s3_bucket.processed_steam_data.id
    model_artifacts = aws_s3_bucket.model_artifacts.id
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  for_each                = local.data_buckets
  bucket                  = each.value
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  for_each = local.data_buckets
  bucket   = each.value
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Partition files are per-run scratch data.
resource "aws_s3_bucket_lifecycle_configuration" "game_partitions" {
  bucket = aws_s3_bucket.game_partitions.id
  rule {
    id     = "expire-run-partitions"
    status = "Enabled"
    filter {}
    expiration {
      days = 30
    }
  }
}

# ---- DynamoDB scrape state -----------------------------------------------------------------

resource "aws_dynamodb_table" "game_ids_state" {
  name         = "game-ids-state"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "appid"

  attribute {
    name = "appid"
    type = "N"
  }

  point_in_time_recovery {
    enabled = true
  }
}

resource "aws_dynamodb_table" "reviews_state_cursor" {
  name         = "reviews-state-cursor"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "appid"

  attribute {
    name = "appid"
    type = "N"
  }

  point_in_time_recovery {
    enabled = true
  }
}
