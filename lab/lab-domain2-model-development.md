# Lab Practice — Domain 2: Model Development
## MLA-C01 | Hands-On Simulations | 6 Labs

> **Format:** Each lab gives you a realistic scenario, setup context, step-by-step tasks, and verification checklists. Labs are designed for SageMaker Studio or notebook environments.

---

## Lab 1: XGBoost Built-in Algorithm — Training, Tuning, and Evaluation

### Scenario
Train a binary classification model on a tabular dataset using SageMaker's built-in XGBoost algorithm. Learn container URIs, hyperparameter configuration, and evaluation metrics.

### Tasks

**Step 1 — Prepare data in RecordIO-Protobuf or CSV**
```python
import sagemaker
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import boto3

session = sagemaker.Session()
bucket  = session.default_bucket()
role    = sagemaker.get_execution_role()
region  = session.boto_region_name

# Generate synthetic binary classification dataset
X, y = make_classification(n_samples=5000, n_features=20, n_informative=10, random_state=42)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
df.insert(0, "label", y)   # XGBoost with CSV expects label as FIRST column

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv("train.csv", index=False, header=False)   # no header for SageMaker XGBoost CSV
val_df.to_csv(  "val.csv",   index=False, header=False)

train_s3 = session.upload_data("train.csv", bucket, "lab2-xgb/train")
val_s3   = session.upload_data("val.csv",   bucket, "lab2-xgb/val")
print(f"Train: {train_s3}\nVal:   {val_s3}")
```

**Step 2 — Retrieve XGBoost container URI**
```python
from sagemaker import image_uris

xgboost_image = image_uris.retrieve(
    framework="xgboost",
    region=region,
    version="1.7-1"
)
print("Container URI:", xgboost_image)
# Expected: 683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1
```

**Step 3 — Configure and launch training job**
```python
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

estimator = Estimator(
    image_uri=xgboost_image,
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path=f"s3://{bucket}/lab2-xgb/output/",
    base_job_name="lab2-xgboost"
)

estimator.set_hyperparameters(
    objective="binary:logistic",   # binary classification
    num_round=150,
    eta=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    early_stopping_rounds=10,
    scale_pos_weight=4,            # if positive class is 4x underrepresented
)

estimator.fit({
    "train":      TrainingInput(train_s3, content_type="text/csv"),
    "validation": TrainingInput(val_s3,   content_type="text/csv"),
}, wait=True)
```

**Step 4 — Review training metrics**
```python
from sagemaker.analytics import TrainingJobAnalytics

job_name = estimator.latest_training_job.name
metrics  = TrainingJobAnalytics(job_name).dataframe()
print(metrics[["timestamp", "metric_name", "value"]].tail(10))

# Plot validation AUC curve
import matplotlib.pyplot as plt
val_auc = metrics[metrics["metric_name"] == "validation:auc"]
plt.plot(val_auc["timestamp"], val_auc["value"])
plt.title("Validation AUC over training rounds")
plt.xlabel("Time"); plt.ylabel("AUC")
plt.show()
```

**Step 5 — Handle class imbalance experiment**
```python
# Compare scale_pos_weight vs no weighting
estimator_no_weight = Estimator(
    image_uri=xgboost_image, role=role,
    instance_count=1, instance_type="ml.m5.xlarge",
    output_path=f"s3://{bucket}/lab2-xgb/output/",
    base_job_name="lab2-xgb-noweight"
)
estimator_no_weight.set_hyperparameters(
    objective="binary:logistic",
    num_round=150, eta=0.1, max_depth=6,
    eval_metric="auc",
    # No scale_pos_weight — model will be biased toward majority class
)
# Compare AUC and F1 scores between the two runs
print("With scale_pos_weight: better recall on minority class")
print("Without: higher accuracy but poor F1 on minority class")
```

### Verification Checklist
- [ ] Training job completes with `Completed` status
- [ ] Validation AUC improves over training rounds (not flat or decreasing)
- [ ] Early stopping fires before `num_round=150` (check `[N] validation-auc` logs)
- [ ] Model artifact `.tar.gz` appears in the S3 output path
- [ ] `scale_pos_weight` experiment shows improved minority-class recall

---

## Lab 2: Hyperparameter Optimization (HPO) — Bayesian Strategy

### Scenario
Run an HPO job on the XGBoost model from Lab 1. Use Bayesian optimization to efficiently search the hyperparameter space and find the best configuration within a budget of 10 training jobs.

### Tasks

**Step 1 — Define hyperparameter ranges**
```python
from sagemaker.tuner import (
    HyperparameterTuner,
    ContinuousParameter,
    IntegerParameter
)

hyperparameter_ranges = {
    "eta":              ContinuousParameter(0.01, 0.3),       # learning rate
    "max_depth":        IntegerParameter(3, 10),
    "subsample":        ContinuousParameter(0.5, 1.0),
    "colsample_bytree": ContinuousParameter(0.5, 1.0),
    "min_child_weight": IntegerParameter(1, 10),
    "num_round":        IntegerParameter(50, 300),
}

# Fixed (non-tuned) hyperparameters
estimator.set_hyperparameters(
    objective="binary:logistic",
    eval_metric="auc",
    early_stopping_rounds=10,
)
```

**Step 2 — Create and launch the tuner**
```python
tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name="validation:auc",
    objective_type="Maximize",
    hyperparameter_ranges=hyperparameter_ranges,
    max_jobs=10,                       # total training jobs
    max_parallel_jobs=2,               # concurrent jobs (Bayesian: keep low)
    strategy="Bayesian",               # Bayesian learns from prior results
    early_stopping_type="Auto",        # stop clearly underperforming jobs early
    base_tuning_job_name="lab2-hpo"
)

tuner.fit({
    "train":      TrainingInput(train_s3, content_type="text/csv"),
    "validation": TrainingInput(val_s3,   content_type="text/csv"),
}, wait=False)

print(f"HPO job started: {tuner.latest_tuning_job.name}")
```

**Step 3 — Monitor and retrieve best job**
```python
# Check status (run after some time)
tuner.wait()  # blocks until complete

# Get best job
best_job = tuner.best_training_job()
print(f"Best job: {best_job}")

# Get all results sorted by metric
results = tuner.analytics().dataframe()
results_sorted = results.sort_values("FinalObjectiveValue", ascending=False)
print(results_sorted[["TrainingJobName", "FinalObjectiveValue",
                       "eta", "max_depth", "subsample"]].head(5))
```

**Step 4 — Warm Start HPO**
```python
from sagemaker.tuner import WarmStartConfig, WarmStartTypes

# Re-use knowledge from the previous HPO job
warm_start_config = WarmStartConfig(
    WarmStartTypes.TRANSFER_LEARNING,       # reuse ALL previous learning
    parents={tuner.latest_tuning_job.name}
)

# Narrow the search space based on best results
new_ranges = {
    "eta":              ContinuousParameter(0.05, 0.2),   # narrowed
    "max_depth":        IntegerParameter(4, 8),            # narrowed
    "subsample":        ContinuousParameter(0.7, 1.0),    # narrowed
    "colsample_bytree": ContinuousParameter(0.7, 1.0),
    "min_child_weight": IntegerParameter(1, 5),
    "num_round":        IntegerParameter(100, 250),
}

tuner_warm = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name="validation:auc",
    objective_type="Maximize",
    hyperparameter_ranges=new_ranges,
    max_jobs=5,                         # fewer jobs needed — warm start helps
    max_parallel_jobs=2,
    strategy="Bayesian",
    warm_start_config=warm_start_config,
    base_tuning_job_name="lab2-hpo-warm"
)
```

### Verification Checklist
- [ ] HPO job shows 10 child training jobs in SageMaker console
- [ ] Best training job has higher AUC than the manually-tuned model from Lab 1
- [ ] `max_parallel_jobs=2` (not equal to max_jobs — Bayesian degrades to random if parallel = total)
- [ ] Warm Start job completes faster and achieves better results than cold start
- [ ] Results DataFrame shows all 10 jobs with their hyperparameter configurations

---

## Lab 3: SageMaker Experiments — Tracking Model Development

### Scenario
You're testing three different algorithms on the same dataset. Use SageMaker Experiments to systematically track, compare, and select the best model.

### Tasks

**Step 1 — Create an Experiment**
```python
from sagemaker.experiments.run import Run
import sagemaker

session = sagemaker.Session()

experiment_name = "loan-approval-comparison"

# Run 1: XGBoost
with Run(
    experiment_name=experiment_name,
    run_name="xgboost-v1",
    sagemaker_session=session
) as run:
    run.log_parameter("algorithm",    "xgboost")
    run.log_parameter("eta",          0.1)
    run.log_parameter("max_depth",    6)
    run.log_parameter("num_round",    150)

    # Simulate training — in real scenario, call estimator.fit()
    # and log metrics from job analytics
    run.log_metric("train:auc",       0.91, step=100)
    run.log_metric("validation:auc",  0.87, step=100)
    run.log_metric("f1_score",        0.83)
    run.log_metric("training_time_s", 45)
    print("Logged XGBoost run")
```

**Step 2 — Log Linear Learner run**
```python
with Run(
    experiment_name=experiment_name,
    run_name="linear-learner-v1",
    sagemaker_session=session
) as run:
    run.log_parameter("algorithm",    "linear_learner")
    run.log_parameter("learning_rate", 0.01)
    run.log_parameter("epochs",        20)

    run.log_metric("train:accuracy",      0.78)
    run.log_metric("validation:accuracy", 0.76)
    run.log_metric("f1_score",            0.74)
    run.log_metric("training_time_s",     12)
    print("Logged Linear Learner run")
```

**Step 3 — Log Random Forest (SKLearn) run**
```python
with Run(
    experiment_name=experiment_name,
    run_name="random-forest-v1",
    sagemaker_session=session
) as run:
    run.log_parameter("algorithm",     "random_forest")
    run.log_parameter("n_estimators",  200)
    run.log_parameter("max_depth",     10)

    run.log_metric("train:accuracy",      0.94)
    run.log_metric("validation:accuracy", 0.85)
    run.log_metric("f1_score",            0.82)
    run.log_metric("training_time_s",     67)
    print("Logged Random Forest run")
```

**Step 4 — Compare runs programmatically**
```python
from sagemaker.experiments.experiment import Experiment

exp = Experiment.load(experiment_name, sagemaker_session=session)

# List all runs
for run_trial in exp.list_trials():
    print(run_trial.trial_name)

# In SageMaker Studio: Experiments → select runs → Add to Chart
# Generates side-by-side comparison of all metrics

print("""
Comparison Summary:
  XGBoost:      val AUC=0.87, F1=0.83, 45s training  ← Best balance
  Linear:       val acc=0.76, F1=0.74, 12s training  ← Fastest but lowest quality
  RandomForest: val acc=0.85, F1=0.82, 67s training  ← Similar to XGBoost, slower
  
Decision: XGBoost wins on AUC. Register it in Model Registry.
""")
```

### Verification Checklist
- [ ] All 3 runs appear in the Experiment in SageMaker Studio
- [ ] Each run has parameters and metrics logged correctly
- [ ] Studio Experiments view shows a side-by-side metric chart
- [ ] XGBoost run shows highest validation AUC

---

## Lab 4: SageMaker Debugger — Detecting Training Anomalies

### Scenario
A deep learning training job shows unstable loss. Use Debugger to detect exploding gradients and automatically stop the job before wasting compute.

### Tasks

**Step 1 — Configure Debugger rules and hooks**
```python
from sagemaker.debugger import Rule, DebuggerHookConfig, CollectionConfig
from sagemaker.debugger import rule_configs

debugger_hook = DebuggerHookConfig(
    s3_output_path=f"s3://{bucket}/lab4-debugger/tensors/",
    collection_configs=[
        CollectionConfig(
            name="weights",
            parameters={"save_interval": "100"}
        ),
        CollectionConfig(
            name="gradients",
            parameters={"save_interval": "100"}
        ),
        CollectionConfig(
            name="losses",
            parameters={"save_interval": "50"}
        ),
    ]
)

debugger_rules = [
    Rule.sagemaker(rule_configs.exploding_gradient()),      # detects gradient explosion
    Rule.sagemaker(rule_configs.loss_not_decreasing()),     # detects stalled training
    Rule.sagemaker(rule_configs.overfit()),                 # detects train/val divergence
    Rule.sagemaker(rule_configs.vanishing_gradient()),      # detects vanishing gradients
]
```

**Step 2 — Add StopTraining action**
```python
from sagemaker.debugger import DebuggerHookConfig, ProfilerConfig
from sagemaker.debugger import CollectionConfig, Rule, rule_configs
from sagemaker.debugger import ProfilerRule

# Auto-stop training when gradient explodes
stop_training_rule = Rule.sagemaker(
    base_config=rule_configs.exploding_gradient(),
    rule_parameters={"tensor_regex": ".*gradient"},
    actions=sagemaker.debugger.actions.StopTraining()   # kills the job automatically
)
```

**Step 3 — Attach to a PyTorch estimator**
```python
from sagemaker.pytorch import PyTorch

pytorch_estimator = PyTorch(
    entry_point="train_nn.py",
    role=role,
    instance_type="ml.p3.2xlarge",
    instance_count=1,
    framework_version="2.0",
    py_version="py310",
    rules=debugger_rules,
    debugger_hook_config=debugger_hook,
    profiler_config=ProfilerConfig(
        system_monitor_interval_millis=500      # collect GPU/CPU metrics every 500ms
    )
)
# pytorch_estimator.fit({"train": train_s3})
```

**Step 4 — Check rule evaluations**
```python
# After training starts (or is stopped by Debugger):
import boto3

sm_client = boto3.client("sagemaker")
job_name   = pytorch_estimator.latest_training_job.name

response = sm_client.describe_training_job(TrainingJobName=job_name)
for rule_eval in response["DebugRuleEvaluationStatuses"]:
    print(f"Rule: {rule_eval['RuleConfigurationName']} → {rule_eval['RuleEvaluationStatus']}")
    if rule_eval.get("StatusDetails"):
        print(f"  Details: {rule_eval['StatusDetails']}")

# If exploding_gradient fires → training job status = "Stopped"
```

**Step 5 — Analyze saved tensors locally**
```python
from smdebug.trials import create_trial

trial = create_trial(f"s3://{bucket}/lab4-debugger/tensors/")

# Get gradient norms over time
for step in trial.steps():
    for tensor_name in trial.tensor_names(regex=".*gradient"):
        tensor = trial.tensor(tensor_name).value(step)
        grad_norm = float((tensor ** 2).sum() ** 0.5)
        if grad_norm > 1000:
            print(f"Step {step} | {tensor_name}: gradient norm = {grad_norm:.2f} ← EXPLODING")
```

### Verification Checklist
- [ ] Debugger collections (weights, gradients, losses) appear as S3 tensors
- [ ] Rule evaluations return `IssuesFound` or `NoIssuesFound`
- [ ] StopTraining action fires and training job transitions to `Stopped`
- [ ] Profiler report shows GPU utilization timeline

---

## Lab 5: SageMaker JumpStart — Fine-Tuning a Pre-Trained BERT Model

### Scenario
Your company needs a text classification model to categorize customer support tickets. Instead of training from scratch, fine-tune a pre-trained BERT model using SageMaker JumpStart.

### Tasks

**Step 1 — Prepare fine-tuning dataset**
```python
# JumpStart BERT expects CSV with columns: [sentence, label]
import pandas as pd

fine_tune_data = pd.DataFrame({
    "sentence": [
        "My account cannot be accessed after password reset",    # label: account_issue
        "I was charged twice for the same order",                # label: billing
        "The product arrived damaged and I need a replacement",  # label: shipping
        "I cannot find the download link for my software",       # label: technical
        # ... 200+ examples total
    ],
    "label": [0, 1, 2, 3]   # integer class indices
})

# Map labels
label_map = {0: "account_issue", 1: "billing", 2: "shipping", 3: "technical"}

fine_tune_data.to_csv("fine_tune_train.csv", index=False, header=False)
s3_fine_tune = session.upload_data("fine_tune_train.csv", bucket, "lab5-jumpstart/data")
```

**Step 2 — Launch JumpStart fine-tuning**
```python
from sagemaker.jumpstart.estimator import JumpStartEstimator

# BERT Base Uncased model
estimator = JumpStartEstimator(
    model_id="huggingface-tc-bert-base-uncased",       # text classification
    model_version="*",                                  # latest version
    role=role,
    instance_type="ml.p3.2xlarge",
    instance_count=1,
)

estimator.set_hyperparameters(
    epochs=3,
    learning_rate=2e-5,             # standard BERT fine-tune LR
    batch_size=32,
    train_only_top_layer=False,     # fine-tune all layers
)

estimator.fit(
    {"training": s3_fine_tune},
    wait=True
)
print("Fine-tuning complete")
```

**Step 3 — Deploy and test**
```python
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",   # CPU inference is fine for BERT classification
)

# Test prediction
import json
response = predictor.predict(
    {"inputs": "I need help resetting my password"},
    initial_args={"ContentType": "application/json"}
)
print("Prediction:", response)
# Expected: [{"label": "LABEL_0", "score": 0.94}] → maps to account_issue
```

**Step 4 — Verify fine-tuning vs. zero-shot baseline**
```python
# Register test accuracy in Experiments
with Run(experiment_name="bert-fine-tune-eval", run_name="bert-fine-tuned") as run:
    test_inputs = [
        "My subscription was billed incorrectly",
        "Package shows delivered but I haven't received it",
    ]
    for text in test_inputs:
        result = predictor.predict({"inputs": text})
        run.log_metric("confidence", result[0]["score"])
        print(f"Text: {text[:50]} → Label: {result[0]['label']} ({result[0]['score']:.2f})")
```

### Verification Checklist
- [ ] Fine-tuning training job completes with `Completed` status
- [ ] Model artifact exists in S3 output path
- [ ] Endpoint shows `InService` status
- [ ] Predictions return label + confidence score
- [ ] Fine-tuned model outperforms zero-shot BERT on domain-specific examples

---

## Lab 6: Distributed Training — Data Parallelism with SageMaker

### Scenario
Your training dataset is too large to fit in single-GPU memory and training is slow. Use SageMaker's distributed training with data parallelism across 4 GPUs.

### Tasks

**Step 1 — Write a distributed training script**

Save as `train_distributed.py`:
```python
# train_distributed.py
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os, json

def get_distributed_config():
    """SageMaker injects these env vars on distributed training nodes."""
    hosts      = json.loads(os.environ.get("SM_HOSTS", '["algo-1"]'))
    host_rank  = int(os.environ.get("SM_CURRENT_HOST_INDEX", 0) or 0)
    num_gpus   = int(os.environ.get("SM_NUM_GPUS", 1))
    return hosts, host_rank, num_gpus

def setup_ddp(rank, world_size):
    dist.init_process_group(
        backend="nccl",          # NCCL for GPU-to-GPU communication
        rank=rank,
        world_size=world_size
    )

def train(args):
    hosts, host_rank, num_gpus = get_distributed_config()
    world_size = len(hosts) * num_gpus
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    setup_ddp(rank=host_rank * num_gpus + local_rank, world_size=world_size)

    device = torch.device(f"cuda:{local_rank}")
    model  = MyModel().to(device)
    model  = DDP(model, device_ids=[local_rank])   # wrap in DDP

    # DistributedSampler partitions data across GPUs automatically
    dataset = MyDataset(os.environ["SM_CHANNEL_TRAIN"])
    sampler = DistributedSampler(dataset, num_replicas=world_size,
                                  rank=host_rank * num_gpus + local_rank)
    loader  = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)             # shuffle consistently across epochs
        for batch in loader:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()

        if local_rank == 0:                  # only rank 0 logs metrics
            print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

    # Only rank 0 saves the model
    if local_rank == 0:
        model_path = os.path.join(os.environ["SM_MODEL_DIR"], "model.pt")
        torch.save(model.module.state_dict(), model_path)
        print(f"Model saved to {model_path}")
```

**Step 2 — Launch distributed training job**
```python
from sagemaker.pytorch import PyTorch

distributed_estimator = PyTorch(
    entry_point="train_distributed.py",
    role=role,
    instance_type="ml.p3.8xlarge",    # 4 V100 GPUs per instance
    instance_count=1,                  # 1 instance × 4 GPUs = 4-way data parallelism
    framework_version="2.0",
    py_version="py310",
    distribution={
        "torch_distributed": {"enabled": True}   # SageMaker manages DDP setup
    },
    hyperparameters={
        "epochs":     10,
        "lr":         3e-4,
        "batch_size": 64,
    }
)

distributed_estimator.fit({"train": train_s3, "val": val_s3})
```

**Step 3 — Scale to multi-node (model parallelism)**
```python
# For models too large for single GPU: use SageMaker Model Parallelism
mp_estimator = PyTorch(
    entry_point="train_model_parallel.py",
    role=role,
    instance_type="ml.p3.16xlarge",    # 8 GPUs per instance
    instance_count=2,                   # 2 nodes × 8 GPUs = 16 GPUs total
    framework_version="2.0",
    py_version="py310",
    distribution={
        "smdistributed": {
            "modelparallel": {
                "enabled": True,
                "parameters": {
                    "microbatches":   4,
                    "placement_strategy": "cluster",
                    "optimize":       "speed",
                    "partitions":     2,          # split model across 2 partitions
                    "pipeline":       "interleaved"
                }
            }
        }
    }
)
print("Model parallel: splits model layers across GPU partitions")
print("Data parallel:  replicates model, splits data batches across GPUs")
```

**Step 4 — Verify distributed training efficiency**
```python
# Check CloudWatch metrics after training
import boto3
from datetime import datetime, timedelta

cw = boto3.client("cloudwatch")
job_name = distributed_estimator.latest_training_job.name

# SageMaker publishes GPU utilization metrics
response = cw.get_metric_statistics(
    Namespace="/aws/sagemaker/TrainingJobs",
    MetricName="GPUUtilization",
    Dimensions=[{"Name": "Host", "Value": f"{job_name}/algo-1"}],
    StartTime=datetime.utcnow() - timedelta(hours=2),
    EndTime=datetime.utcnow(),
    Period=60,
    Statistics=["Average"]
)
for point in response["Datapoints"][-5:]:
    print(f"GPU Utilization: {point['Average']:.1f}%")
# Target: >80% GPU utilization indicates efficient distribution
```

### Verification Checklist
- [ ] Training job shows multiple host log streams (algo-1, algo-2 for multi-node)
- [ ] Only rank-0 logs training metrics (no duplicate metric lines)
- [ ] Model saved once to `/opt/ml/model/` (not duplicated per GPU)
- [ ] GPU utilization >70% during training (check CloudWatch)
- [ ] Training wall-clock time decreases proportionally with GPU count

---

## Summary — Domain 2 Lab Skills Matrix

| Lab | Service / Concept | Skills Practiced |
|-----|------------------|-----------------|
| 1 | XGBoost built-in | Container URIs, CSV format, hyperparameters, class imbalance |
| 2 | SageMaker HPO | Bayesian vs random, parameter ranges, Warm Start, parallel jobs |
| 3 | SageMaker Experiments | Run logging, metric tracking, multi-algorithm comparison |
| 4 | SageMaker Debugger | Rules, tensor collections, StopTraining action, tensor analysis |
| 5 | JumpStart | BERT fine-tuning, pre-trained model selection, inference |
| 6 | Distributed Training | DDP setup, DistributedSampler, data parallelism vs model parallelism |

### Common Mistakes to Avoid
- **XGBoost CSV**: Label must be the **first** column — unlike sklearn convention
- **HPO Bayesian**: Setting `max_parallel_jobs == max_jobs` defeats Bayesian optimization; it degrades to random search
- **HPO Warm Start**: `IDENTICAL_DATA_AND_ALGORITHM` requires the exact same dataset and algorithm; use `TRANSFER_LEARNING` when changing hyperparameter ranges
- **Debugger**: Collections must be declared before training starts — cannot add collections to a running job
- **Distributed**: Without `sampler.set_epoch(epoch)`, all epochs shuffle identically — leads to poor training
- **JumpStart latest version**: Use `model_version="*"` to pin to latest, but note version changes may affect results
