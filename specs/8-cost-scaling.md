## Spec 8: Cost behaviour when scaling up

How the monthly bill moves as data, users, experiments and traffic grow, what stays flat, and the guardrails to put in
place before any of it happens. Prices are us-east-1 list prices (approximate, check the AWS pricing pages before
relying on them), before tax.

### Baseline (measured, September 2026)

| What | Cost |
|---|---|
| Initial backfill pipeline run (~25 h, one-time) | ECS $11.8 + VPC public IPv4 $2.0 + Athena $0.2 ≈ **$14** |
| Inference processing job (`ml.t3.xlarge`, 23 min – 3.5 h) | $0.10 – 0.70 / run |
| Bedrock rerank (Nova 2 Lite, 1000 users) | ~$2 / run (~$0.002 / reranked user) |
| DynamoDB recs write (~2M items, ~1.7 KB each) | ~$1.5 – 2.5 for a full write, less for changed users only |
| Training job (`ml.g4dn.xlarge`, 27 – 53 min billable) | $0.35 – 0.65 / job |
| Serving path (CloudFront, S3, Lambda, DynamoDB reads) at demo traffic | ~$0 (inside the always-free tiers) |

Steady state with the weekly schedule and the MLOps roadmap (shadow eval, weekly retrain, monitoring, domain):
**~$20 – 35 / month**, **~$40 – 75 / month** with heavy experimentation.

### How each driver scales

| Driver | Grows with | Shape | Notes |
|---|---|---|---|
| Scraping (ECS Fargate) | New reviews / games per week | Linear | Backfills are one-time spikes. Each task pays $0.005/h for its public IPv4. Fargate Spot ≈ -70% (cursors make tasks resumable). |
| dbt / Athena | Bytes scanned | Linear, small | $5 / TB. Incremental models and Iceberg partition pruning keep it at cents. |
| S3 + Iceberg storage | Total data kept | Linear, small | $0.023 / GB-month. Expire old Iceberg snapshots, model versions and catalog versions. |
| Retrieval (processing job) | Users × catalog games | Linear in users | Brute-force matmul on CPU. At ~10× users, move to a larger instance (quota: only `ml.t3.*` today). |
| Shadow evaluation | Challengers × weekly runs | +1 retrieval pass each | Offline only (metrics to S3). Never a second DynamoDB write or rerank. |
| Bedrock rerank | Reranked users per run | Linear | The main variable cost. Always cap it with `RERANK_MAX_USERS`. |
| DynamoDB writes | Changed users per run × item size | Linear | $0.625 / M write units (1 KB each). Write only changed users. |
| DynamoDB storage | Users with stored recs | Flat, then linear | 25 GB always free (~14M users at today's item size), then $0.25 / GB-month. |
| Training | Interactions × epochs × runs | Linear | ~$0.74 / GPU-hour. Quota is **1** `ml.g4dn.xlarge`: more experiments queue up rather than cost more at once. Managed spot ≈ -60%. |
| Monitoring | Metrics, alarms, log volume | Step | CloudWatch: $0.30 / custom metric-month, $0.10 / alarm, $0.50 / GB of logs ingested. Metrics written to S3/Athena are ~free. |
| Serving traffic | Uncached API requests | Linear | ≈ **$3.5 / 1M requests** beyond the free tiers (see below). |
| Search index downloads | Page loads × 2.5 MB | Linear, **steep** | CloudFront egress $0.085 / GB after 1 TB free: 1M fresh page loads ≈ 2.5 TB ≈ $130. |

#### Serving, per 1M uncached API requests (after the free tiers)

| Piece | Cost |
|---|---|
| CloudFront HTTPS requests + CloudFront Function | ~$1.00 + $0.10 |
| Lambda requests + compute (1 GB, ~50 ms) | ~$0.20 + $0.85 |
| DynamoDB reads (recs item + batch of details, ~11 read units) | ~$1.40 |
| **Total** | **~$3.5** |

Free every month: CloudFront 10M requests / 1 TB, Lambda 1M requests / 400k GB-s, DynamoDB 25 GB storage.

### Step changes (sudden jumps to plan for)

- **Anything always-on:** a SageMaker real-time endpoint ($40 – 150+ / month each), a managed MLflow tracking server,
  a NAT Gateway (~$33 / month + data), Studio / notebook instances left running, OpenSearch. The design stays
  batch + serverless so none of these are needed.
- **Quotas:** GPU training (1 instance) and processing (`ml.t3.*` only) cap how much you can spend per hour and also
  how far the pipeline can scale. Raising them unlocks both.
- **Free tiers ending:** DynamoDB storage past 25 GB, CloudFront past 1 TB / 10M requests.

### Abuse: the serving path is the only piece strangers can trigger

The pipeline only costs what we schedule. The public site can be made to cost money by anyone ("denial of wallet").

| Vector | What happens today | Worst case if sustained |
|---|---|---|
| Flood of API requests (e.g. 100k / min) | Reserved concurrency 30 throttles the excess with 429s (throttled invocations are free), and real users get 429s too. Served requests still cost Lambda + DynamoDB, and **every** request costs CloudFront. | Hundreds of dollars a day; thousands a month |
| Direct calls to the Function URL | Auth is currently `NONE`, so the raw URL skips CloudFront (and any future WAF) entirely. | Lambda/DynamoDB part of the above |
| Repeated downloads of `/data/games.json` (2.5 MB) | Served from the CloudFront cache, so the cost is only egress. | The 1 TB free tier is gone in minutes at 100k / min, then ~$20 / min. **The largest vector.** |
| Enumerating Steam ids / scraping the recs | Allowed, since it is public data behind a public API. | Costs as above; the data is public anyway |
| Cache busting with random query strings | Blocked: only `limit` / `details` are in the cache key. | none |
| Large `POST` bodies | Capped by Pydantic (`liked_game_ids` ≤ 1000, `max_liked_games`). | none |

### Guardrails (do before scaling up or advertising the site)

1. **Function URL back to `AWS_IAM`**, as Spec 6 intended: CloudFront signs with OAC and the raw URL returns 403.
2. **CloudFront flat-rate plan** (Free / Pro, if it covers this distribution's origins; check at adoption time): WAF
   and DDoS protection are bundled, and going over the allowance throttles instead of billing. That turns the
   worst case into "slow" instead of "expensive". Otherwise use **AWS WAF** with a per-IP rate-based rule (~$6 /
   month + $0.60 / M requests), plus a rule for `/data/*`.
3. **DynamoDB `on_demand_throughput` max read/write units** on the serving tables: a hard cap on the read cost.
4. **Budget alert** (`infrastructure/budget.tf`, set `budget_alert_email`) plus **Cost Anomaly Detection** (free).
   Optionally add a kill switch: budget → SNS → Lambda that sets the serving reserved concurrency to 0.
5. **Pipeline caps**: `RERANK_MAX_USERS`, a limit on shadow challengers, lifecycle rules on S3 / ECR, Iceberg
   snapshot expiry, and log retention (already set).
6. **Spot capacity** for scraping (Fargate Spot) and training (managed spot) once jobs checkpoint.
