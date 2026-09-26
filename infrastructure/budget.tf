# Account-wide monthly cost budget: an email when actual spend passes 80% of the limit, or the
# forecast passes 100%. Only a warning: nothing is stopped. The frontend / serving caps are
# serving_reserved_concurrency and CloudFront caching.

resource "aws_budgets_budget" "monthly" {
  count        = var.budget_alert_email != "" ? 1 : 0
  name         = "${var.project}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.budget_monthly_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_alert_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_alert_email]
  }
}
