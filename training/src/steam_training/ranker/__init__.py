"""Second stage (spec 9): a LightGBM ranker that re-orders a retriever's top candidates.

- `features`: the feature definitions (pure numpy; shared with inference and the serving port)
- `data`: query groups after a retriever's cutoff, popularity and game tables
- `model`: flat ranking matrices, LightGBM `lambdarank`, ranking metrics
- `pipeline`: `python -m steam_training rank` (train, report, gate, save)
"""
