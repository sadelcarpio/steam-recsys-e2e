## Spec 4: Inference Pipeline

The inference Pipeline should be the last step in the Step Function workflow,
only runnable if there is a deployed model already (deployed via @.github/workflows/training-cd.yml or either locally).
If a model isn't found on the correspondent artifact bucket, this step should be skipped.

Inference Pipeline should:

- Treat the `game_features` and `user_features` Iceberg tables as source of truth.
- Read the latest features from a User, and rank against all Games (passed through item tower with its latest features
  as well) (currently not using ANN since latency is
  not critical on this step and Two-Tower model is intentionally lightweight). Keep a configurable "K" top candidates.
- Leverage AWS Bedrock (investigate a free tier for or very cheap LLM that can be integrated easily) to rank top "K (20-
  30)" items with a prompt including the user games' history, along with an explanation for each ranked item,
  or at least a reduced set (say top "N (5 - 10)").
- Write the top K results (including the top N item explanations) to a DynamoDB table
  (`game-explainable-recommendations`)
  keyed by user id. overwritten on each run of the inference pipeline.

### Infrastructure:
