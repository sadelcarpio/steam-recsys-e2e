## Spec 11: Serving observability (structured logs, metrics, traces, dashboard)

### Problem

The `recsys-serving` Lambda logs plain text plus one JSON line per request (`method`, `path`,
`status`, `ms`). Nothing answers: which stage of a slow request was slow (DynamoDB, the cold
start S3 load of the online bundle, the numpy scoring)? What are p50 / p95 / p99 per route? How
often does a user fall back to the popularity list? How much does a cold start cost?

This spec instruments the current path. It comes before spec 12 (online ranking + LLM rerank),
so the baseline is measured before the online path gets more stages, and every new stage
arrives already instrumented.

### Behaviour

**Request trace (hand-made, `steam_serving.obs`)**

- A `RequestTrace` per invocation: `span(name, **attrs)` context manager (`perf_counter`,
  nesting recorded as `parent`), `annotate(**attrs)` for request-level attributes. No global
  state beyond the current trace (a `ContextVar`).
- Spans on the current path: `route` (parse + validation), `dynamo.recommendations`,
  `dynamo.games` (BatchGetItem), `online.load` (`BundleLoader` refresh, with `reloaded`,
  `catalog_mb`), `online.embed`, `online.score` (matmul + top K), `serialize`.
- Request attributes: `route` as a **template** (`/users/{id}/recommendations`, never the raw
  path: metric cardinality), `status`, `source` (personalized / popular / online),
  `fallback_reason` (`unknown_user`, `no_known_liked_games`, none), `liked_count`,
  `ignored_count`, `limit`, `details`, `model_id`, `cold_start`.
- At the end of `App.handle`, **one** JSON line per request (`"type": "request"`) with
  `request_id` (`context.aws_request_id`), `trace_id` (from `_X_AMZN_TRACE_ID`, when set),
  the attributes, total `ms` and the spans. It replaces the current summary line. Unhandled
  errors also carry `error_type`.
- The response carries `x-request-id`, so a frontend error can be looked up in the logs.

**Structured logs**

- A JSON formatter for every stdlib log record (`level`, `logger`, `message`, `request_id`,
  exception text), replacing the text format of `configure_logging`. The Lambda's `log_format`
  stays `Text` (the formatter is the point of the exercise, and the EMF lines below must reach
  CloudWatch untouched).

**Metrics (CloudWatch Embedded Metric Format)**

- The request line carries an `_aws` EMF block, namespace `SteamRecsys/Serving`, so CloudWatch
  extracts metrics from the log line: no API call, no latency added.
- Metrics: `Latency` (ms) and `Requests` by `Route`; `Errors` (5xx) by `Route`; `Fallbacks`
  by `Reason`; `StageLatency` (ms) by `Stage`; `ColdStart` (count); `ModelLoadSeconds`.
- Dimensions are kept small (each combination is a billed custom metric, ~$0.30 / month):
  ~5 routes, ~7 stages, 2 reasons → under 25 metrics. The README states the bound.

**Traces (X-Ray)**

- `tracing_config { mode = "Active" }` on the Lambda (+ `AWSXRayDaemonWriteAccess` on its role):
  init / invocation segments with no code.
- `TRACING_ENABLED` (default true): `aws-xray-sdk` patches boto3 (DynamoDB, S3, later
  Bedrock calls appear as subsegments) and each `span` is mirrored as a subsegment with its
  attrs as metadata (`route`, `source` as annotations, so traces are searchable). Off: no
  import, spans only go to the log line.
- Overhead is measured, not assumed: `scripts/loadtest.py` against the deployed URL with tracing
  on and off; p50 / p95 and init duration of both go in the README.

**Dashboard, saved queries, alarms (Terraform, `infrastructure/observability.tf`)**

- `aws_cloudwatch_dashboard` `recsys-serving`: requests and latency p50 / p95 / p99 per route;
  `StageLatency` p95 per stage; 4xx / 5xx; fallbacks by reason; cold starts and init duration;
  Lambda concurrency and throttles; DynamoDB consumed reads and throttles of both tables; a Logs
  Insights widget with the slowest requests of the last hour (with their spans).
- `aws_cloudwatch_query_definition` (saved queries): slowest requests with spans, errors with
  request id, latency percentiles per route, cold start vs warm.
- Alarms: p95 latency of `POST /recommendations` above `serving_p95_alarm_ms` (default 1500)
  for 3 of 5 minutes, and 5xx rate above 5%, to an SNS topic; `alarm_email` (default empty = no
  subscription).
- No Grafana (a managed workspace needs IAM Identity Center and costs per editor; self-hosted is
  not serverless). The dashboard is IaC like the rest.

**Load generator (`serving/scripts/loadtest.py`)**

A bounded traffic generator (default 200 requests, concurrency 5, a mix of known users, unknown
users and liked-game lists from the search index), signed for `AWS_IAM` URLs or plain for
CloudFront. Prints client-side percentiles; the server-side numbers come from the dashboard.
Never run from CI.

### Out of scope

OpenTelemetry / ADOT (the Lambda layer costs hundreds of ms of cold start; revisit if a second
backend is wanted), MLflow tracing (GenAI-oriented; the LLM stage of spec 12 logs its own
usage), frontend RUM (CloudWatch RUM), CloudFront access logs, tracing of the batch pipeline.

### Tests / docs / infra

- Unit tests: span timing and nesting with a fake clock; the request line shape for each route,
  4xx, 5xx and the fallbacks; route templating (no ids in `route`); a valid EMF block (metric
  names declared in `_aws.CloudWatchMetrics` exist at top level, dimensions are strings);
  `x-request-id` on every response; JSON formatter (exception text, request id); tracing off →
  `aws_xray_sdk` never imported; tracing on with a fake recorder → one subsegment per span.
- The existing contract and handler tests still pass.
- `uv add aws-xray-sdk` in `serving` (the Lambda zip via `scripts/build_lambda.sh`).
- New setting `TRACING_ENABLED` in `ServingSettings` + SSM (`infrastructure/serving.tf`); new
  variables `serving_p95_alarm_ms`, `alarm_email` (`docs/deployment.md`).
- `serving/README.md` (what is measured, the dashboard, the saved queries, the overhead
  numbers, how to run the load test), `serving/CLAUDE.md` (invariants: one request line per
  invocation, route templates only, every new stage gets a span).
