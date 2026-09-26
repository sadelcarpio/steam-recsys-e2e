# frontend component: static web app (frontend/) on a private S3 bucket behind CloudFront. One
# distribution, three origins:
#   /*       -> S3 recsys-frontend-<acct> (the built app; extensionless paths -> /index.html)
#   /data/*  -> S3 model-artifacts-<acct>/serving/search (the game search index from inference)
#   /api/*   -> the recsys-serving Function URL (prefix stripped)
# Same origin for everything: no CORS. S3 is read through Origin Access Control. The Function URL
# is signed through OAC when its auth type is AWS_IAM (then only this distribution may call it);
# with NONE CloudFront calls it unsigned. The app is uploaded by the frontend CD.

locals {
  frontend_custom_domain = var.frontend_domain_name != ""
  # AWS managed policies (stable ids).
  cache_policy_caching_optimized = "658327ea-f89d-4fab-a63d-7e88639e58f6"
  origin_request_all_except_host = "b689b0a8-53d0-40ab-baf2-68738e2966ac" # AllViewerExceptHostHeader
  serving_url_host               = split("/", aws_lambda_function_url.serving.function_url)[2]
}

# ---- Bucket --------------------------------------------------------------------------------

resource "aws_s3_bucket" "frontend" {
  bucket = "recsys-frontend-${local.account_id}"
}

resource "aws_s3_bucket_public_access_block" "frontend" {
  bucket                  = aws_s3_bucket.frontend.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "frontend" {
  bucket = aws_s3_bucket.frontend.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

data "aws_iam_policy_document" "frontend_bucket" {
  statement {
    sid       = "CloudFrontRead"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.frontend.arn}/*"]
    principals {
      type        = "Service"
      identifiers = ["cloudfront.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "AWS:SourceArn"
      values   = [aws_cloudfront_distribution.frontend.arn]
    }
  }
}

resource "aws_s3_bucket_policy" "frontend" {
  bucket = aws_s3_bucket.frontend.id
  policy = data.aws_iam_policy_document.frontend_bucket.json
}

# Only the search index of model-artifacts is readable by the distribution.
data "aws_iam_policy_document" "model_artifacts_bucket" {
  statement {
    sid       = "CloudFrontReadSearchIndex"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.model_artifacts.arn}/${local.search_index_prefix}/*"]
    principals {
      type        = "Service"
      identifiers = ["cloudfront.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "AWS:SourceArn"
      values   = [aws_cloudfront_distribution.frontend.arn]
    }
  }
}

resource "aws_s3_bucket_policy" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id
  policy = data.aws_iam_policy_document.model_artifacts_bucket.json
}

# ---- Origin access -------------------------------------------------------------------------

resource "aws_cloudfront_origin_access_control" "frontend_s3" {
  name                              = "${var.project}-frontend-s3"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

resource "aws_cloudfront_origin_access_control" "frontend_lambda" {
  name                              = "${var.project}-frontend-lambda"
  origin_access_control_origin_type = "lambda"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

# Both statements, as for a public URL: InvokeFunctionUrl for the URL, InvokeFunction for the
# function behind it. Harmless while the auth type is NONE.
resource "aws_lambda_permission" "serving_cloudfront_url" {
  statement_id  = "AllowCloudFrontInvokeFunctionUrl"
  action        = "lambda:InvokeFunctionUrl"
  function_name = aws_lambda_function.serving.function_name
  principal     = "cloudfront.amazonaws.com"
  source_arn    = aws_cloudfront_distribution.frontend.arn
}

resource "aws_lambda_permission" "serving_cloudfront_invoke" {
  statement_id  = "AllowCloudFrontInvokeFunction"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.serving.function_name
  principal     = "cloudfront.amazonaws.com"
  source_arn    = aws_cloudfront_distribution.frontend.arn
}

# ---- Edge functions (viewer request) -------------------------------------------------------

# App routes (/u/123) have no file extension: serve the app, which routes in the browser.
# Unlike a 404 -> index.html error response, this leaves /api and /data errors untouched.
resource "aws_cloudfront_function" "frontend_spa" {
  name    = "${var.project}-frontend-spa"
  runtime = "cloudfront-js-2.0"
  publish = true
  code    = file("${path.module}/cloudfront/spa.js")
}

# /api/users/1/recommendations -> /users/1/recommendations, /data/games.json -> /games.json
resource "aws_cloudfront_function" "frontend_strip_prefix" {
  name    = "${var.project}-frontend-strip-prefix"
  runtime = "cloudfront-js-2.0"
  publish = true
  code    = file("${path.module}/cloudfront/strip_prefix.js")
}

# ---- Cache ---------------------------------------------------------------------------------

# Cached as long as the origin's Cache-Control says (nothing without one), keyed by path + query
# string only. No headers in the key: they would be forwarded, and the Function URL and S3 need
# their own Host.
resource "aws_cloudfront_cache_policy" "frontend_origin_controlled" {
  name        = "${var.project}-frontend-origin-controlled"
  min_ttl     = 0
  default_ttl = 0
  max_ttl     = 86400
  parameters_in_cache_key_and_forwarded_to_origin {
    enable_accept_encoding_gzip   = true
    enable_accept_encoding_brotli = true
    cookies_config {
      cookie_behavior = "none"
    }
    headers_config {
      header_behavior = "none"
    }
    query_strings_config {
      query_string_behavior = "all"
    }
  }
}

# ---- Distribution --------------------------------------------------------------------------

resource "aws_cloudfront_distribution" "frontend" {
  enabled             = true
  comment             = "${var.project} frontend"
  default_root_object = "index.html"
  price_class         = "PriceClass_100"
  http_version        = "http2and3"
  aliases             = local.frontend_custom_domain ? [var.frontend_domain_name] : []

  origin {
    origin_id                = "app"
    domain_name              = aws_s3_bucket.frontend.bucket_regional_domain_name
    origin_access_control_id = aws_cloudfront_origin_access_control.frontend_s3.id
  }

  origin {
    origin_id                = "search"
    domain_name              = aws_s3_bucket.model_artifacts.bucket_regional_domain_name
    origin_path              = "/${local.search_index_prefix}"
    origin_access_control_id = aws_cloudfront_origin_access_control.frontend_s3.id
  }

  origin {
    origin_id                = "api"
    domain_name              = local.serving_url_host
    origin_access_control_id = local.serving_public ? null : aws_cloudfront_origin_access_control.frontend_lambda.id
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    target_origin_id       = "app"
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["GET", "HEAD"]
    cached_methods         = ["GET", "HEAD"]
    compress               = true
    cache_policy_id        = local.cache_policy_caching_optimized
    function_association {
      event_type   = "viewer-request"
      function_arn = aws_cloudfront_function.frontend_spa.arn
    }
  }

  ordered_cache_behavior {
    path_pattern           = "/data/*"
    target_origin_id       = "search"
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["GET", "HEAD"]
    cached_methods         = ["GET", "HEAD"]
    compress               = false # stored gzipped
    cache_policy_id        = aws_cloudfront_cache_policy.frontend_origin_controlled.id
    function_association {
      event_type   = "viewer-request"
      function_arn = aws_cloudfront_function.frontend_strip_prefix.arn
    }
  }

  # GETs are cached as the Lambda's Cache-Control says (query string in the key); POST
  # /recommendations answers no-store.
  ordered_cache_behavior {
    path_pattern             = "/api/*"
    target_origin_id         = "api"
    viewer_protocol_policy   = "redirect-to-https"
    allowed_methods          = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods           = ["GET", "HEAD"]
    compress                 = true
    cache_policy_id          = aws_cloudfront_cache_policy.frontend_origin_controlled.id
    origin_request_policy_id = local.origin_request_all_except_host
    function_association {
      event_type   = "viewer-request"
      function_arn = aws_cloudfront_function.frontend_strip_prefix.arn
    }
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = !local.frontend_custom_domain
    acm_certificate_arn            = local.frontend_custom_domain ? var.frontend_certificate_arn : null
    ssl_support_method             = local.frontend_custom_domain ? "sni-only" : null
    minimum_protocol_version       = local.frontend_custom_domain ? "TLSv1.2_2021" : null
  }

  lifecycle {
    precondition {
      condition     = !local.frontend_custom_domain || var.frontend_certificate_arn != ""
      error_message = "frontend_domain_name needs frontend_certificate_arn (an ACM certificate in us-east-1)."
    }
  }
}
