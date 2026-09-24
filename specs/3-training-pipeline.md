## Spec 3: Training Pipeline Implementation
This specs' main objective is to create the training pipeline for a two-tower model, using the processed interactions 
table from the etl step (Iceberg). THe training Pipeline, deployed in Amazon SageMaker, should:
- Implement a Two-Tower model leveraging the following features from the interactions table:
    * User Tower: games_reviewed_positive (deliberately not including user id due to sparsity)
    * Item Tower: game_idx, game_is_free, game_developers, game_publishers, game_genres, game_categories, game_reviews_ratio
    * Label: is_positive
  Use In-batch negative sampling as well as hard negatives
  Simple model, User tower -> pooling -> MLP. Item Tower -> embeddings pooling -> MLP adding numerical features
- Load all the embedding tables as a torch embedding in memory for fast lookups (developers, publishers, genres, categories, game_idx)
- For the user tower, use the same games embedding table as the game_idx embeddings, but updating it per epoch. This is very important to avoid 
unstable gradients.
- Split the dataset temporally, leaving approximately 10% o the data for validation (no test for simplicity)
- Build the evaluation check, defining metric being recall @ K. configurable K, try values 30 - 50 - 100 for fast iteration
- If quota allows, train on GPU instance.
- keep model artifacts as simple S3 objects, refer to @CLAUDE.md for the promotion strategy.
- The Pipeline should roughly be: Dataset Split -> DataLoader / Collate -> Training -> Evaluation.

### Infrastructure:
- S3 bucket for the model artifacts
- CD manual trigger for training, CD manual trigger for promotion as well, which would run only evaluation on a given model.
