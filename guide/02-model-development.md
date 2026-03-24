# Domain 2: ML Model Development
## MLA-C01 Study Guide — 26% of Exam

---

## Table of Contents
1. [Choosing the Right ML Approach](#1-choosing-the-right-ml-approach)
2. [SageMaker Built-In Algorithms](#2-sagemaker-built-in-algorithms)
3. [SageMaker Training Jobs](#3-sagemaker-training-jobs)
4. [Custom Training Containers](#4-custom-training-containers)
5. [Hyperparameter Tuning (HPO)](#5-hyperparameter-tuning-hpo)
6. [SageMaker Experiments](#6-sagemaker-experiments)
7. [SageMaker Debugger](#7-sagemaker-debugger)
8. [SageMaker JumpStart](#8-sagemaker-jumpstart)
9. [Distributed Training](#9-distributed-training)
10. [Model Evaluation & Selection](#10-model-evaluation--selection)
11. [Key Facts & Exam Tips](#11-key-facts--exam-tips)

---

## 1. Choosing the Right ML Approach

### Problem Type → Algorithm Decision Tree

```
SUPERVISED LEARNING
├── Classification
│   ├── Binary        → XGBoost, Linear Learner, KNN
│   ├── Multi-class   → XGBoost, Linear Learner, DeepAR (sequence)
│   └── Image         → Image Classification (ResNet), Object Detection
├── Regression
│   ├── Tabular       → XGBoost, Linear Learner, DeepAR
│   └── Time-Series   → DeepAR, CNN-QR, Prophet (SageMaker Canvas)
└── NLP
    ├── Text Classif. → BlazingText, HuggingFace (via JumpStart)
    ├── Seq2Seq       → Sequence-to-Sequence algorithm
    └── Embeddings    → Object2Vec, BlazingText (Word2Vec mode)

UNSUPERVISED LEARNING
├── Clustering        → K-Means, IP Insights (anomaly)
├── Dimensionality    → PCA (SageMaker built-in)
├── Topic Modeling    → LDA, NTM (Neural Topic Model)
└── Anomaly Detection → Random Cut Forest (RCF), IP Insights

REINFORCEMENT LEARNING
└── Continuous control → SageMaker RL (RLlib, Coach)
```

### When NOT to Use ML
- Rule-based logic covers all cases
- Dataset is too small (< 100 samples)
- Model interpretability is legally required and simpler models achieve adequate performance
- Latency requirements can't be met even with optimization

---

## 2. SageMaker Built-In Algorithms

### Algorithm Reference Table

| Algorithm | Task | Input Format | Key Hyperparameters |
|-----------|------|-------------|---------------------|
| **XGBoost** | Classification, Regression | CSV, LibSVM, Parquet | `num_round`, `max_depth`, `eta`, `subsample`, `scale_pos_weight` |
| **Linear Learner** | Classification, Regression | RecordIO-Protobuf, CSV | `predictor_type`, `num_models`, `learning_rate`, `l1`, `wd` |
| **KNN** | Classification, Regression | RecordIO-Protobuf | `k`, `feature_dim`, `sample_size`, `predictor_type` |
| **K-Means** | Clustering | RecordIO-Protobuf | `k`, `feature_dim`, `mini_batch_size`, `init_method` |
| **PCA** | Dimensionality Reduction | RecordIO-Protobuf | `num_components`, `algorithm_mode` (regular/randomized) |
| **Random Cut Forest** | Anomaly Detection | RecordIO-Protobuf, CSV | `num_trees`, `num_samples_per_tree`, `feature_dim` |
| **IP Insights** | Anomaly Detection | CSV | `num_entity_vectors`, `vector_dim` |
| **DeepAR** | Time-Series Forecasting | JSON Lines | `context_length`, `prediction_length`, `num_cells`, `num_layers` |
| **BlazingText** | Text Classification, Word2Vec | Text | `mode` (Word2Vec, classification), `vector_dim` |
| **Object2Vec** | Embeddings / Classification | JSON Lines | `enc_dim`, `mlp_dim`, `output_layer` |
| **NTM** | Topic Modeling | RecordIO-Protobuf | `num_topics`, `encoder_layers`, `epochs` |
| **LDA** | Topic Modeling | RecordIO-Protobuf | `num_topics`, `alpha0` |
| **Seq2Seq** | Machine Translation | RecordIO-Protobuf | `num_layers_encoder`, `num_layers_decoder`, `dropout` |
| **Image Classification** | Image Classification | RecordIO (JPEG) | `num_classes`, `num_training_samples`, `base_net` |
| **Object Detection** | Object Detection | RecordIO (JPEG) | `num_classes`, `base_network`, `optimizer` |
| **Semantic Segmentation** | Pixel Segmentation | RecordIO (JPEG) | `num_classes`, `backbone`, `optimizer` |

### XGBoost Deep Dive (Most Common on Exam)

```python
from sagemaker.estimator import Estimator
import sagemaker

# Get built-in XGBoost image
image_uri = sagemaker.image_uris.retrieve(
    framework="xgboost",
    region=region,
    version="1.7-1"
)

xgb = Estimator(
    image_uri=image_uri,
    instance_type='ml.m5.xlarge',
    instance_count=1,
    output_path='s3://my-bucket/model-artifacts/',
    role=role
)

xgb.set_hyperparameters(
    objective='binary:logistic',     # loss function
    num_round=100,                   # number of trees
    max_depth=6,                     # tree depth
    eta=0.3,                         # learning rate
    gamma=0,                         # min loss reduction for split
    min_child_weight=1,             # min sum of weights in leaf
    subsample=0.8,                  # row sampling
    colsample_bytree=0.8,           # column sampling per tree
    scale_pos_weight=10,            # for imbalanced datasets
    eval_metric='auc',              # evaluation metric
    early_stopping_rounds=10       # stop if no improvement
)

xgb.fit({
    'train': sagemaker.inputs.TrainingInput(
        's3://my-bucket/data/train/', content_type='csv'
    ),
    'validation': sagemaker.inputs.TrainingInput(
        's3://my-bucket/data/val/', content_type='csv'
    )
})
```

**XGBoost Objective Functions:**

| Objective | Use Case |
|-----------|---------|
| `binary:logistic` | Binary classification (outputs probability) |
| `binary:hinge` | Binary classification (outputs 0/1) |
| `multi:softmax` | Multi-class (outputs class index) |
| `multi:softprob` | Multi-class (outputs probabilities) |
| `reg:squarederror` | Regression (MSE) |
| `reg:tweedie` | Regression for count data |
| `rank:pairwise` | Ranking problems |

### Linear Learner Deep Dive

Linear Learner is unique because it **trains multiple models in parallel** and selects the best one.

```python
from sagemaker.linear_learner import LinearLearner

ll = LinearLearner(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    predictor_type='binary_classifier',  # 'regressor', 'multiclass_classifier'
    num_models=32,           # trains 32 models with different hyperparams
    binary_classifier_model_selection_criteria='accuracy',
    positive_example_weight_mult='balanced'  # handles class imbalance
)
```

### DeepAR Time-Series Deep Dive

DeepAR is a **probabilistic forecasting** algorithm using RNNs.

```python
from sagemaker.estimator import Estimator

deepar_image = sagemaker.image_uris.retrieve("forecasting-deepar", region)

deepar = Estimator(
    image_uri=deepar_image,
    role=role,
    instance_count=1,
    instance_type='ml.m5.2xlarge'
)

deepar.set_hyperparameters(
    time_freq='D',              # daily frequency
    prediction_length=30,       # forecast 30 days ahead
    context_length=90,          # use 90 days of history
    epochs=300,
    num_cells=40,
    num_layers=3,
    dropout_rate=0.05,
    likelihood='gaussian',      # 'negative-binomial', 'student-T', 'deterministic-L1'
    num_eval_samples=100        # samples for probabilistic predictions
)
```

**DeepAR Input Format (JSON Lines):**
```json
{"start": "2023-01-01", "target": [100, 110, 95, 120, 105], "cat": [1], "dynamic_feat": [[1.2, 0.8, 1.1, 0.9, 1.0]]}
{"start": "2023-01-01", "target": [50, 55, 48, 60, 52], "cat": [2]}
```

---

## 3. SageMaker Training Jobs

### Training Job Anatomy

```
Training Job
    │
    ├── Input: S3 data path → INSTANCE (/opt/ml/input/)
    ├── Code: S3 script/container → INSTANCE
    ├── Compute: EC2 instance type (ml.p3.2xlarge, ml.m5.xlarge, etc.)
    ├── Output: Model artifacts → S3 (/opt/ml/model/ → output_path)
    └── Logs: CloudWatch Logs
```

### Container Directory Structure

```
/opt/ml/
├── input/
│   ├── config/
│   │   ├── hyperparameters.json   # hyperparameters
│   │   └── resourceconfig.json   # cluster info
│   └── data/
│       ├── train/                 # training channel
│       └── validation/            # validation channel
├── model/                         # output: model artifacts go here
├── output/
│   ├── data/                     # additional output
│   └── failure                   # failure message file
└── code/                         # your script
```

### Instance Type Selection Guide

| Use Case | Recommended Instance | Notes |
|---------|---------------------|-------|
| Small tabular ML (XGBoost, Linear) | `ml.m5.xlarge` – `ml.m5.4xlarge` | Cost-effective CPU |
| Deep learning training | `ml.p3.2xlarge` (V100) or `ml.p4d.24xlarge` (A100) | GPU required |
| Training with large memory | `ml.r5.xlarge` – `ml.r5.4xlarge` | Memory-optimized |
| Distributed training | `ml.p3.16xlarge` or multi-node `ml.p3.2xlarge` | Use distributed training libraries |
| Inference (real-time) | `ml.c5.xlarge` – `ml.c5.4xlarge` | CPU-optimized for inference |
| Inference (GPU) | `ml.g4dn.xlarge` | Cost-effective GPU inference |

### Managed Spot Training (Cost Reduction)

```python
estimator = Estimator(
    ...,
    use_spot_instances=True,
    max_wait=7200,          # max time including queue wait (seconds)
    max_run=3600,           # max training time (seconds)
    checkpoint_s3_uri='s3://my-bucket/checkpoints/'  # REQUIRED for resumption
)
```

> **Exam Tip:** Spot training can save up to **90% of training costs**. Always specify `checkpoint_s3_uri` so training can resume after spot interruption.

### Custom Training Script (Script Mode)

```python
# train.py
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Hyperparameters from SageMaker
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=None)
    
    # SageMaker environment variables
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    
    args = parser.parse_args()
    
    # Load data
    train_df = pd.read_csv(os.path.join(args.train, 'train.csv'))
    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label']
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, os.path.join(args.model_dir, 'model.joblib'))
    print(f"Training accuracy: {accuracy_score(y_train, model.predict(X_train))}")
```

```python
# Launcher script (notebook or CI/CD)
from sagemaker.sklearn.estimator import SKLearn

sklearn_estimator = SKLearn(
    entry_point='train.py',
    framework_version='1.2-1',
    instance_type='ml.m5.xlarge',
    role=role,
    hyperparameters={
        'n-estimators': 200,
        'max-depth': 10
    }
)

sklearn_estimator.fit({
    'train': 's3://my-bucket/data/train/',
    'validation': 's3://my-bucket/data/val/'
})
```

---

## 4. Custom Training Containers

Use custom containers when your framework is not natively supported.

### Container Requirements

Your Docker container must:
1. Read hyperparameters from `/opt/ml/input/config/hyperparameters.json`
2. Read training data from `/opt/ml/input/data/`
3. Write model artifacts to `/opt/ml/model/`
4. Write failure messages to `/opt/ml/output/failure`
5. Exit with code 0 on success, non-zero on failure

### Build & Push to ECR

```bash
# Build Docker image
docker build -t my-ml-training .

# Authenticate to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  123456789.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag my-ml-training:latest \
  123456789.dkr.ecr.us-east-1.amazonaws.com/my-ml-training:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/my-ml-training:latest
```

### Use Custom Container in Estimator

```python
estimator = Estimator(
    image_uri='123456789.dkr.ecr.us-east-1.amazonaws.com/my-ml-training:latest',
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    output_path='s3://my-bucket/output/'
)
```

---

## 5. Hyperparameter Tuning (HPO)

### Tuning Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Bayesian Optimization** | Uses past results to choose next values (Gaussian Process) | Most effective; few evaluations needed |
| **Random Search** | Randomly samples from search space | Independent jobs; easy to parallelize |
| **Grid Search** | Exhaustive search of all combinations | Small search spaces only |
| **Hyperband** | Adaptive resource allocation (early stopping of bad runs) | Faster convergence; many epochs |
| **Auto** | SageMaker chooses (usually Bayesian) | Default choice |

> **Exam Tip:** Bayesian is the default strategy and most efficient for most use cases. Use Hyperband for neural networks with epochs.

### Hyperparameter Types

| Type | Class | Example |
|------|-------|---------|
| Continuous | `ContinuousParameter(min, max)` | `learning_rate`, `dropout` |
| Integer | `IntegerParameter(min, max)` | `num_layers`, `batch_size` |
| Categorical | `CategoricalParameter([values])` | `optimizer`, `activation` |

### HPO Configuration

```python
from sagemaker.tuner import (
    HyperparameterTuner,
    ContinuousParameter,
    IntegerParameter,
    CategoricalParameter
)

hyperparameter_ranges = {
    'eta': ContinuousParameter(0.01, 0.3, scaling_type='Logarithmic'),
    'max_depth': IntegerParameter(3, 10),
    'num_round': IntegerParameter(50, 500),
    'subsample': ContinuousParameter(0.5, 1.0),
    'gamma': ContinuousParameter(0, 5),
}

tuner = HyperparameterTuner(
    estimator=xgb,
    objective_metric_name='validation:auc',
    hyperparameter_ranges=hyperparameter_ranges,
    objective_type='Maximize',
    max_jobs=20,              # total training jobs
    max_parallel_jobs=5,      # concurrent jobs
    strategy='Bayesian',      # Bayesian, Random, Hyperband, Auto
    early_stopping_type='Auto'  # stop poor-performing jobs early
)

tuner.fit({
    'train': train_input,
    'validation': validation_input
})

# Get best training job
best_job = tuner.best_training_job()
best_estimator = sagemaker.estimator.Estimator.attach(best_job)
```

### Warm Starting HPO

Warm start allows HPO to reuse previous tuning results — saves time and cost.

```python
from sagemaker.tuner import WarmStartConfig, WarmStartTypes

warm_start_config = WarmStartConfig(
    warm_start_type=WarmStartTypes.TRANSFER_LEARNING,  # or IDENTICAL_DATA_AND_ALGORITHM
    parents={"previous-tuning-job-name"}
)

tuner_warm = HyperparameterTuner(
    ...,
    warm_start_config=warm_start_config
)
```

| Warm Start Type | Use Case |
|-----------------|---------|
| `IDENTICAL_DATA_AND_ALGORITHM` | Same data, same algorithm — continue previous search |
| `TRANSFER_LEARNING` | Different data or different hyperparameter ranges |

---

## 6. SageMaker Experiments

Track, organize, and compare ML training runs across experiments.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Experiment** | Logical grouping (e.g., "churn-model-v2") |
| **Run** | Single training execution within an experiment |
| **Trial Component** | Atomic unit (processing step, training step, evaluation step) |
| **Tags** | Key-value labels for filtering |

### Using Experiments

```python
from sagemaker.experiments.run import Run

with Run(
    experiment_name="churn-prediction",
    run_name="xgboost-v1",
    sagemaker_session=sess
) as run:
    run.log_parameter("learning_rate", 0.1)
    run.log_parameter("max_depth", 6)
    run.log_metric("train:auc", 0.91)
    run.log_metric("validation:auc", 0.87)
    run.log_file("confusion_matrix.png", is_output=True)

# Compare runs in SageMaker Studio UI or:
from sagemaker.experiments import Experiment
experiment = Experiment.load("churn-prediction")
df = experiment.list_runs().to_dataframe()
```

---

## 7. SageMaker Debugger

Debugger provides **real-time insights** into training jobs — detect vanishing gradients, overfitting, and system bottlenecks.

### Debugger Rule Categories

| Category | Example Rules | What They Detect |
|----------|--------------|-----------------|
| **Training Issues** | `VanishingGradient`, `ExplodingGradient` | Gradient problems in deep learning |
| **Overfitting** | `Overfit`, `LossNotDecreasing` | Model convergence issues |
| **Profile Analysis** | `ProfilerRule` | CPU/GPU utilization bottlenecks |
| **Custom Rules** | User-defined | Business-specific conditions |

### Enabling Debugger

```python
from sagemaker.debugger import Rule, DebuggerHookConfig, rules_spec

rules = [
    Rule.sagemaker(rules_spec.vanishing_gradient()),
    Rule.sagemaker(rules_spec.overfit()),
    Rule.sagemaker(rules_spec.loss_not_decreasing()),
    Rule.sagemaker(rules_spec.ProfilerReport()),
]

hook_config = DebuggerHookConfig(
    hook_parameters={
        "save_interval": "100"  # save tensors every 100 steps
    }
)

estimator = Estimator(
    ...,
    rules=rules,
    debugger_hook_config=hook_config
)
```

---

## 8. SageMaker JumpStart

JumpStart provides **pre-trained foundation models and ML solutions** — fine-tune with minimal code.

### JumpStart Capabilities

| Category | Examples |
|----------|---------|
| **Foundation Models** | Llama 3, Mistral, Falcon, Stable Diffusion, Titan |
| **Pre-trained Models** | ResNet, BERT, RoBERTa, ViT, DeiT |
| **ML Solutions** | Demand Forecasting, Fraud Detection, Churn Prediction templates |

### Fine-Tuning with JumpStart

```python
from sagemaker.jumpstart.estimator import JumpStartEstimator

# Fine-tune a pre-trained model
estimator = JumpStartEstimator(
    model_id="huggingface-tc-bert-base-uncased",  # text classification
    model_version="*",
    instance_type="ml.p3.2xlarge",
    role=role,
    hyperparameters={
        "epochs": 5,
        "learning_rate": 2e-5,
        "batch_size": 16
    }
)

estimator.fit({"training": "s3://my-bucket/fine-tune-data/"})

# Deploy fine-tuned model
predictor = estimator.deploy(instance_type="ml.m5.xlarge")
```

### JumpStart Model IDs (Common)

| Model | Task | ID Pattern |
|-------|------|-----------|
| BERT | Text Classification | `huggingface-tc-bert-base-uncased` |
| ResNet | Image Classification | `pytorch-ic-resnet50` |
| Llama 3 | Text Generation | `meta-textgeneration-llama-3-*` |
| Stable Diffusion | Image Generation | `model-imagegeneration-stabilityai-*` |

---

## 9. Distributed Training

### Types of Parallelism

| Type | Description | When to Use |
|------|-------------|-------------|
| **Data Parallelism** | Split data across GPUs; same model on each GPU | Large datasets, model fits on 1 GPU |
| **Model Parallelism** | Split model layers across GPUs | Model too large for 1 GPU (LLMs) |
| **Pipeline Parallelism** | Split model into sequential stages | Variants of model parallelism |

### SageMaker Distributed Training Libraries

```python
# Data Parallelism with SageMaker Distributed Data Parallel (SMDDP)
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    instance_type='ml.p3dn.24xlarge',  # high-bandwidth NVLink required
    instance_count=2,                   # 2 instances = 16 GPUs total
    framework_version='2.0.0',
    py_version='py310',
    distribution={
        'smdistributed': {
            'dataparallel': {
                'enabled': True,
                'custom_mpi_options': '-verbose -x NCCL_DEBUG=VERSION'
            }
        }
    }
)
```

```python
# Model Parallelism
estimator = PyTorch(
    entry_point='train_mp.py',
    instance_type='ml.p4d.24xlarge',
    instance_count=1,
    distribution={
        'smdistributed': {
            'modelparallel': {
                'enabled': True,
                'parameters': {
                    'microbatches': 4,
                    'placement_strategy': 'spread',
                    'pipeline': 'interleaved',
                    'optimize': 'speed',
                    'partitions': 4       # split model into 4 partitions
                }
            }
        }
    }
)
```

### Distributed Training Frameworks Comparison

| Framework | AWS Library | Best For |
|-----------|-------------|---------|
| PyTorch DDP | SMDDP | Standard data parallel PyTorch |
| PyTorch FSDP | SageMaker + native | Large models, memory efficiency |
| Horovod | SageMaker MPI | TensorFlow/PyTorch, established framework |
| MPI | Built-in | Custom multi-node workloads |

---

## 10. Model Evaluation & Selection

### Classification Metrics

| Metric | Formula | When to Use |
|--------|---------|-------------|
| **Accuracy** | $\frac{TP + TN}{N}$ | Balanced classes |
| **Precision** | $\frac{TP}{TP + FP}$ | Minimize false positives (spam filter) |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | Minimize false negatives (cancer detection) |
| **F1-Score** | $\frac{2 \times P \times R}{P + R}$ | Imbalanced classes, balance P & R |
| **AUC-ROC** | Area under ROC curve | Ranking quality, threshold-independent |
| **PR-AUC** | Area under Precision-Recall | Severely imbalanced datasets |
| **Log Loss** | $-\frac{1}{N}\sum y \log \hat{y}$ | Probabilistic predictions |

### Regression Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| **MAE** | $\frac{1}{N}\sum|y - \hat{y}|$ | Robust to outliers |
| **MSE** | $\frac{1}{N}\sum(y - \hat{y})^2$ | Penalizes large errors more |
| **RMSE** | $\sqrt{MSE}$ | Same units as target |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Proportion of variance explained |
| **MAPE** | $\frac{100\%}{N}\sum\frac{|y-\hat{y}|}{y}$ | Percentage error (time series) |

### Bias-Variance Tradeoff

```
High Bias (Underfitting):
  → Training error is HIGH
  → Validation error ≈ Training error (both high)
  → Solution: More complex model, more features, reduce regularization

High Variance (Overfitting):
  → Training error is LOW
  → Validation error >> Training error (large gap)
  → Solution: More data, regularization (L1/L2), dropout, cross-validation

Ideal:
  → Low training error AND low validation error
  → Small gap between train and validation
```

### Cross-Validation

```python
# K-Fold Cross-Validation
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print(f"F1: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Stratified K-Fold for imbalanced data
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### Time-Series Validation

```
NEVER use random splits for time series — always use temporal ordering.

Training Window:          |--train--|
Validation Window:                  |val|
Test Window:                              |test|

Walk-forward validation:
  Step 1: |----train 1----|val1|
  Step 2: |------train 2------|val2|
  Step 3: |--------train 3--------|val3|
```

---

## 11. Key Facts & Exam Tips

### Algorithm Quick Reference

| Scenario | Algorithm |
|----------|-----------|
| Tabular binary/multi-class classification | **XGBoost** |
| When you need multiple models trained simultaneously | **Linear Learner** |
| Anomaly detection in unlabeled data | **Random Cut Forest** |
| Time-series forecasting (multiple related series) | **DeepAR** |
| Text classification + Word2Vec | **BlazingText** |
| Topic discovery from documents | **LDA** or **NTM** |
| Clustering unknown groups | **K-Means** |
| Dimensionality reduction of numerical features | **PCA** |
| Image classification | **Image Classification (ResNet)** |
| Finding similar items / recommendations | **Object2Vec** |

### Important Training Concepts

- **Pipe Mode** → stream data from S3 during training (RecordIO only → legacy)
- **FastFile Mode** → stream data from S3 with POSIX interface (any format → preferred)
- **Managed Spot Training** → save 60-90%; requires checkpointing for resume
- **Warm Pool** → keep instances warm between jobs; reduces startup time (not free)
- **Local Mode** → run training locally in SageMaker notebook for fast debugging
- **Debugger Profiler** → automatically detects GPU/CPU bottlenecks

### Exam Scenario Traps

| Scenario | Correct Answer | Wrong Answer |
|---------|---------------|-------------|
| Reduce training cost for long jobs | Managed Spot + Checkpointing | Reserved Instances |
| Train model with GPU but dataset doesn't fit in memory | Distributed Model Parallel | Single GPU instance |
| Track and compare 50 training runs | SageMaker Experiments | CloudWatch Logs |
| Optimize hyperparameters efficiently with limited budget | Bayesian HPO | Grid Search |
| Handle class imbalance in XGBoost | `scale_pos_weight` | Simply upsample manually |
| Compare algorithm performance over multiple datasets | SageMaker Experiments | Manual CSV tracking |
