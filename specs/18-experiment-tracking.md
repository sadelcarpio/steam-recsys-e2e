# Experiment Tracking on training runs
Currently since we are on a heavy experimenting stage, it is practical to have a way to track all of the experiments and fairly compare them across our important 
metrics of success. FOr training runs, offline metrics like hitrate@k and recall@k (global and calculated across users activity percentiles), coverage, 
and sanity checks like embedding collapse, lift over popularity baseline, as well as per epoch progress and lineage metadata (commit id / tag, snapshot id of the dataset, features used/
signature). This is a first implementation for experimenting and having informed training telemetry in order to choose a champion based on our existing analysis and goal 
which is having a good retrieval model, so far.

Need to analyze , for local experiments:
- Spin up an mlflow tracking server (preferrably Docker - Docker COmpose based with state persistence)
- Integrate the tracking of the desired metrics, checks, time series and plots if necessary on the training runs
- be able to tag a champion model and a challenger, so far manually
For a remote server, come up with the simplest and cheapest solution (sagemaker mlflow, ecs + db, or just keep local backed by S3)
