# Domain 4: ML Solution Monitoring, Maintenance, and Security
## MLA-C01 Study Guide — 24% of Exam

---

## Table of Contents
1. [SageMaker Model Monitor](#1-sagemaker-model-monitor)
2. [SageMaker Clarify](#2-sagemaker-clarify)
3. [SageMaker Debugger & Profiler](#3-sagemaker-debugger--profiler)
4. [CloudWatch for ML Workloads](#4-cloudwatch-for-ml-workloads)
5. [IAM Security for ML](#5-iam-security-for-ml)
6. [Network Security (VPC)](#6-network-security-vpc)
7. [Data Encryption](#7-data-encryption)
8. [Model Governance & Compliance](#8-model-governance--compliance)
9. [Cost Optimization](#9-cost-optimization)
10. [SageMaker ML Lineage Tracking](#10-sagemaker-ml-lineage-tracking)
11. [Key Facts & Exam Tips](#11-key-facts--exam-tips)

---

## 1. SageMaker Model Monitor

Model Monitor continuously analyzes **deployed endpoint data** to detect issues automatically.

### Four Types of Model Monitor

| Monitor Type | What It Detects | Baseline From |
|-------------|----------------|---------------|
| **Data Quality Monitor** | Feature distribution drift, missing values, schema changes | Training data statistics |
| **Model Quality Monitor** | Prediction accuracy/quality degradation (AUC, F1 drop) | Ground truth labels vs predictions |
| **Bias Drift Monitor** | Changes in model fairness metrics over time | Clarify baseline |
| **Feature Attribution Drift** | Changes in which features the model relies on (SHAP values) | Clarify explainability baseline |

### Data Quality Monitor Setup

```python
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.model_monitor import CronExpressionGenerator

monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
)

# Step 1: Create baseline from training data
monitor.suggest_baseline(
    baseline_dataset='s3://my-bucket/training-data/train.csv',
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri='s3://my-bucket/baseline/',
    wait=True
)

# Step 2: Schedule monitor on endpoint
monitor.create_monitoring_schedule(
    monitor_schedule_name="data-quality-schedule",
    endpoint_input=predictor.endpoint_name,
    output_s3_uri='s3://my-bucket/monitor-reports/',
    statistics=monitor.baseline_statistics(),   # from suggest_baseline output
    constraints=monitor.suggested_constraints(),
    schedule_cron_expression=CronExpressionGenerator.hourly(),   # or daily, custom
    enable_cloudwatch_metrics=True
)
```

### Model Quality Monitor Setup

```python
from sagemaker.model_monitor import ModelQualityMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

model_quality_monitor = ModelQualityMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
)

# Baseline from validation data (needs ground truth labels)
baseline_job = model_quality_monitor.suggest_baseline(
    baseline_dataset='s3://my-bucket/validation.csv',
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri='s3://my-bucket/model-quality-baseline/',
    problem_type='BinaryClassification',       # or 'Regression', 'MulticlassClassification'
    inference_attribute='prediction',
    ground_truth_attribute='label',
    probability_attribute='probability',
    probability_threshold_attribute=0.5,
)

# Schedule continuous monitoring
model_quality_monitor.create_monitoring_schedule(
    monitor_schedule_name="model-quality-schedule",
    endpoint_input=EndpointInput(
        endpoint_name=predictor.endpoint_name,
        inference_attribute='prediction',
        ground_truth_s3_input='s3://my-bucket/ground-truth/'  # labeled as it arrives
    ),
    output_s3_uri='s3://my-bucket/model-quality-reports/',
    schedule_cron_expression=CronExpressionGenerator.daily()
)
```

### Model Monitor Violation Types

| Violation | Description |
|-----------|-------------|
| `data_type_check` | Column has wrong data type |
| `completeness_check` | Column has more nulls than baseline |
| `baseline_drift_check` | Distribution has drifted beyond threshold |
| `domain_content_check` | Value outside expected domain |

### Drift Detection Methods Used by Model Monitor

| Method | Description | Use Case |
|--------|-------------|---------|
| **KL Divergence** | Measures distribution difference | Continuous features |
| **L-infinity Norm** | Max difference between CDFs | Continuous features |
| **Chi-squared test** | Compares categorical distributions | Categorical features |
| **TVD (Total Variation Distance)** | Sum of absolute differences in PMF | Categorical features |

### Ground Truth Collection for Model Quality

```python
# Capture endpoint I/O for model quality monitoring
from sagemaker.model_monitor import DataCaptureConfig

data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=20,           # capture 20% of traffic
    destination_s3_uri='s3://my-bucket/data-capture/',
    capture_options=[
        {"CaptureMode": "Input"},
        {"CaptureMode": "Output"}
    ],
    csv_content_types=['text/csv'],
    json_content_types=['application/json']
)

# Apply data capture when deploying endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',
    data_capture_config=data_capture_config
)
```

---

## 2. SageMaker Clarify

Clarify detects **bias in data/models** and provides **feature explainability** (SHAP values).

### Bias Concepts

| Term | Definition | Example |
|------|-----------|---------|
| **Sensitive Attribute** | Protected characteristic | Gender, age, race |
| **Facet** | Sub-group defined by sensitive attribute | `gender = female` |
| **Label** | Target variable | Loan approved (1/0) |
| **CI (Class Imbalance)** | Imbalance in representation of facets | 80% males in dataset |
| **DPL (Difference in Positive Proportions in Labels)** | Different positive label rates across facets | Women approved 40% vs men 70% |
| **DPPL (Difference in Positive Proportions in Predicted Labels)** | Post-training bias metric | Model predicts women approved 35% vs men 72% |

### Pre-Training Bias Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **CI** (Class Imbalance) | Imbalance in facet representation | [-1, +1]; 0 = balanced |
| **DPL** (Diff in Positive Proportions in Labels) | Label rate difference across facets | [-1, +1]; 0 = no bias |
| **KL** (KL Divergence) | Distribution difference | [0, ∞); 0 = identical |
| **JS** (Jensen-Shannon) | Symmetric KL divergence | [0, 1]; 0 = identical |
| **KS** (Kolmogorov-Smirnov) | Max CDF difference | [0, 1]; 0 = identical |

### Post-Training Bias Metrics

| Metric | Description |
|--------|-------------|
| **DPPL** (Diff in Positive Proportions in Predicted Labels) | Difference in model's positive prediction rates |
| **DI** (Disparate Impact) | Ratio of positive prediction rates; < 0.8 often indicates bias |
| **DCO** (Difference in Conditional Outcomes) | Accuracy difference across facets |
| **RD** (Recall Difference) | Recall gap between facets |
| **DAR** (Difference in Acceptance Rates) | Positive prediction rate gap |
| **FT** (Flip Test) | Counterfactual fairness — would outcome change if attribute flipped? |

### Running Clarify Bias Analysis

```python
from sagemaker import clarify

clarify_processor = clarify.SageMakerClarifyProcessor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    sagemaker_session=sess
)

bias_config = clarify.BiasConfig(
    label_values_or_threshold=[1],        # positive outcome = 1
    facet_name='gender',                  # sensitive attribute column
    facet_values_or_threshold=[0],        # facet to analyze (0 = female)
    group_name='age_group'                # subgroup for intersectional bias
)

data_config = clarify.DataConfig(
    s3_data_input_path='s3://bucket/data.csv',
    s3_output_path='s3://bucket/clarify-output/',
    label='loan_approved',
    headers=['customer_id', 'gender', 'income', 'age_group', 'loan_approved'],
    dataset_type='text/csv'
)

model_config = clarify.ModelConfig(
    model_name='loan-model',
    instance_type='ml.m5.xlarge',
    instance_count=1,
    content_type='text/csv'
)

# Run pre + post training bias analysis
clarify_processor.run_bias(
    data_config=data_config,
    bias_config=bias_config,
    model_config=model_config,
    pre_training_methods='all',
    post_training_methods='all'
)
```

### SHAP Explainability

SHAP (SHapley Additive exPlanations) quantifies each feature's contribution to a prediction.

```python
shap_config = clarify.SHAPConfig(
    baseline=[                        # baseline/reference data point
        [0, 50000, 35, 0]            # neutral values per feature
    ],
    num_samples=100,                  # number of SHAP perturbations
    agg_method='mean_abs',           # aggregation method for global SHAP
    save_local_shap_values=True      # save per-sample SHAP values
)

clarify_processor.run_explainability(
    data_config=data_config,
    model_config=model_config,
    explainability_config=shap_config
)
```

**Reading SHAP Values:**
- Positive SHAP → feature increased prediction
- Negative SHAP → feature decreased prediction
- |SHAP| magnitude → feature importance

---

## 3. SageMaker Debugger & Profiler

### Debugger Use Cases

| Problem | Debugger Rule |
|---------|-------------|
| Weights not updating | `VanishingGradient` |
| Loss exploding | `ExplodingGradient` |
| Model memorizing training data | `Overfit` |
| Training loss not decreasing | `LossNotDecreasing` |
| Poor class prediction on imbalanced data | `ClassImbalance` |
| Training stalled | `Stalled` |

### System Profiling

```python
from sagemaker.debugger import ProfilerConfig, FrameworkProfile

profiler_config = ProfilerConfig(
    system_monitor_interval_millis=500,    # sample every 500ms
    framework_profile_params=FrameworkProfile(
        num_steps=10,                      # profile 10 steps
        start_step=5                       # start profiling at step 5
    )
)

estimator = PyTorch(
    ...,
    profiler_config=profiler_config
)
```

**What Profiler Reports:**
- GPU/CPU utilization over time
- Memory usage
- I/O bandwidth
- Network bandwidth
- Bottleneck identification (data loading, forward/backward pass, etc.)

---

## 4. CloudWatch for ML Workloads

### Key SageMaker CloudWatch Metrics

| Namespace | Metric | Description |
|-----------|--------|-------------|
| `AWS/SageMaker` | `TrainingJobsTotal` | Count of training jobs |
| `AWS/SageMaker` | `Invocations` | Endpoint invocation count |
| `AWS/SageMaker` | `InvocationsPerInstance` | Per-instance invocations |
| `AWS/SageMaker` | `ModelLatency` | Model processing time (μs) |
| `AWS/SageMaker` | `OverheadLatency` | SageMaker overhead (μs) |
| `AWS/SageMaker` | `Invocation4XXErrors` | Client errors (bad input) |
| `AWS/SageMaker` | `Invocation5XXErrors` | Model errors |
| `AWS/SageMaker` | `MemoryUtilization` | Container memory % |
| `AWS/SageMaker` | `CPUUtilization` | Container CPU % |
| `AWS/SageMaker` | `GPUUtilization` | GPU % |
| `AWS/SageMaker` | `DiskUtilization` | Disk % |

### Setting Up Monitoring Alarms

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Alarm for high latency
cloudwatch.put_metric_alarm(
    AlarmName='high-model-latency',
    Namespace='AWS/SageMaker',
    MetricName='ModelLatency',
    Dimensions=[
        {'Name': 'EndpointName', 'Value': 'my-endpoint'},
        {'Name': 'VariantName', 'Value': 'AllTraffic'}
    ],
    Statistic='Average',
    Period=300,          # 5 minutes
    EvaluationPeriods=3,
    Threshold=500000,    # 500ms in microseconds
    ComparisonOperator='GreaterThanThreshold',
    AlarmActions=['arn:aws:sns:us-east-1:123456789:ml-ops-alerts'],
    TreatMissingData='notBreaching'
)

# Alarm for 5XX errors (model failures)
cloudwatch.put_metric_alarm(
    AlarmName='endpoint-5xx-errors',
    Namespace='AWS/SageMaker',
    MetricName='Invocation5XXErrors',
    Dimensions=[{'Name': 'EndpointName', 'Value': 'my-endpoint'}],
    Statistic='Sum',
    Period=300,
    EvaluationPeriods=1,
    Threshold=1,
    ComparisonOperator='GreaterThanOrEqualToThreshold',
    AlarmActions=['arn:aws:sns:us-east-1:123456789:ml-ops-alerts']
)
```

### Custom Metrics from Training Scripts

```python
# Emit custom metrics from training script
import re

# SageMaker parses regex-matched metrics from STDOUT/STDERR
# Training script output:
print("Train AUC: 0.91")
print("Validation AUC: 0.87")
print("F1 Score: 0.83")

# In estimator definition, define the regex patterns:
from sagemaker.estimator import Estimator

estimator = Estimator(
    ...,
    metric_definitions=[
        {'Name': 'train:auc', 'Regex': 'Train AUC: ([0-9\\.]+)'},
        {'Name': 'validation:auc', 'Regex': 'Validation AUC: ([0-9\\.]+)'},
        {'Name': 'f1', 'Regex': 'F1 Score: ([0-9\\.]+)'}
    ]
)
```

### CloudWatch Logs Insights for ML

```
# Find training jobs that failed
fields @timestamp, @message
| filter @logStream like /training-job/
| filter @message like /ERROR|Exception|failed/
| sort @timestamp desc
| limit 50

# Track endpoint invocation errors
fields @timestamp, @message
| filter @logStream like /endpoint/
| stats count() as errors by bin(5m)
| sort @timestamp desc
```

---

## 5. IAM Security for ML

### Principle of Least Privilege for ML

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:DescribeTrainingJob",
        "sagemaker:StopTrainingJob",
        "sagemaker:CreateModel",
        "sagemaker:CreateEndpoint",
        "sagemaker:InvokeEndpoint"
      ],
      "Resource": "arn:aws:sagemaker:us-east-1:123456789:*"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::my-ml-bucket/project-A/*"
    },
    {
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::123456789:role/SageMakerExecutionRole",
      "Condition": {
        "StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}
      }
    }
  ]
}
```

### SageMaker IAM Roles

| Role | Purpose |
|------|---------|
| **SageMaker Execution Role** | Role assumed by SageMaker to access S3, ECR, CloudWatch, etc. |
| **SageMaker Studio User Role** | Per-user/profile role for Studio access |
| **SageMaker Pipeline Role** | Role used by Pipelines to execute steps |

### Resource-Based Policies

```python
# S3 bucket policy — restrict to specific SageMaker role only
{
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {
            "AWS": "arn:aws:iam::123456789:role/SageMakerExecutionRole"
        },
        "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
        "Resource": [
            "arn:aws:s3:::my-ml-bucket",
            "arn:aws:s3:::my-ml-bucket/*"
        ]
    }]
}
```

### Condition Keys for SageMaker

| Condition Key | Example Use |
|---------------|-------------|
| `sagemaker:RootAccess` | Prevent root access in SageMaker notebooks |
| `sagemaker:DirectInternetAccess` | Block internet access in notebooks |
| `sagemaker:VpcSecurityGroupIds` | Require VPC for training/hosting |
| `sagemaker:VpcSubnets` | Require specific VPC subnets |
| `sagemaker:InstanceTypes` | Limit to approved instance types |
| `sagemaker:InterContainerTrafficEncryption` | Require encrypted inter-node traffic |
| `sagemaker:VolumeKmsKey` | Require KMS encryption for storage volumes |

---

## 6. Network Security (VPC)

### SageMaker in a VPC

```python
# Training job inside VPC
estimator = Estimator(
    ...,
    subnets=['subnet-12345abc', 'subnet-67890def'],       # private subnets
    security_group_ids=['sg-abcdef12'],
    encrypt_inter_container_traffic=True,                  # encrypt multi-node traffic
    enable_network_isolation=True                          # no internet access during training
)

# Endpoint inside VPC
model.deploy(
    ...,
    vpc_config={
        'Subnets': ['subnet-12345abc'],
        'SecurityGroupIds': ['sg-abcdef12']
    }
)
```

### VPC Considerations for ML Workloads

| Component | VPC Requirement |
|-----------|----------------|
| **Training Jobs** | Can run with or without VPC; VPC required for compliance |
| **Endpoints** | Can run with or without VPC |
| **SageMaker Studio** | Recommended in VPC for enterprise |
| **Processing Jobs** | Can run in VPC |
| **S3 access from VPC** | Use **VPC Gateway Endpoint** (free) |
| **Other AWS services from VPC** | Use **VPC Interface Endpoints (PrivateLink)** |

### PrivateLink for SageMaker API

```
Without PrivateLink:  Notebook → Internet → SageMaker API ← training instances
With PrivateLink:     Notebook → VPC Endpoint → SageMaker API (no internet traversal)

Required VPC Interface Endpoints for fully private SageMaker:
  - com.amazonaws.us-east-1.sagemaker.api
  - com.amazonaws.us-east-1.sagemaker.runtime
  - com.amazonaws.us-east-1.s3 (gateway endpoint)
  - com.amazonaws.us-east-1.ecr.api
  - com.amazonaws.us-east-1.ecr.dkr
  - com.amazonaws.us-east-1.logs (CloudWatch)
```

### Network Isolation for Training

```python
estimator = Estimator(
    ...,
    enable_network_isolation=True   # container cannot make outbound network calls
)
```

> **Exam Tip:** `enable_network_isolation=True` prevents the container from downloading packages during training. All libraries must be baked into the container image or provided as input data.

---

## 7. Data Encryption

### Encryption at Rest

```python
import boto3

# S3 default encryption
s3 = boto3.client('s3')
s3.put_bucket_encryption(
    Bucket='my-ml-bucket',
    ServerSideEncryptionConfiguration={
        'Rules': [{
            'ApplyServerSideEncryptionByDefault': {
                'SSEAlgorithm': 'aws:kms',
                'KMSMasterKeyID': 'arn:aws:kms:us-east-1:123456789:key/...'
            }
        }]
    }
)

# Training job with KMS encryption
estimator = Estimator(
    ...,
    volume_kms_key='arn:aws:kms:us-east-1:123456789:key/...',  # EBS volume
    output_kms_key='arn:aws:kms:us-east-1:123456789:key/...',  # S3 output
)
```

### Encryption in Transit

```python
# Inter-container traffic encryption (multi-node training)
estimator = Estimator(
    ...,
    encrypt_inter_container_traffic=True  # TLS between training nodes
)
```

### KMS Key Policy for SageMaker

```json
{
    "Statement": [{
        "Effect": "Allow",
        "Principal": {
            "AWS": "arn:aws:iam::123456789:role/SageMakerExecutionRole"
        },
        "Action": [
            "kms:Decrypt",
            "kms:GenerateDataKey",
            "kms:CreateGrant",
            "kms:DescribeKey"
        ],
        "Resource": "*",
        "Condition": {
            "StringEquals": {
                "kms:ViaService": "sagemaker.us-east-1.amazonaws.com"
            }
        }
    }]
}
```

### Encryption Summary Table

| Data | Encryption Option |
|------|-----------------|
| S3 training data | SSE-S3 or SSE-KMS |
| S3 model artifacts | SSE-S3 or SSE-KMS |
| EBS (training instance storage) | KMS via `volume_kms_key` |
| Multi-node inter-container traffic | TLS via `encrypt_inter_container_traffic` |
| SageMaker Studio EFS | KMS |
| Feature Store offline (S3) | SSE-KMS |
| Feature Store online (DynamoDB) | KMS |

---

## 8. Model Governance & Compliance

### SageMaker Model Cards

Model Cards document a model's intended use, training details, evaluation results, and ethical considerations.

```python
from sagemaker.model_card import (
    ModelCard,
    ModelOverview,
    TrainingDetails,
    EvaluationJob,
    IntendedUses,
    BusinessDetails
)

model_card = ModelCard(
    name="churn-prediction-model-card",
    status=ModelCardStatusEnum.DRAFT,
    model_overview=ModelOverview(
        model_description="XGBoost model for predicting customer churn",
        problem_type="Binary Classification",
        algorithm_type="XGBoost"
    ),
    intended_uses=IntendedUses(
        purpose_of_model="Identify at-risk customers for retention campaigns",
        intended_users="Marketing and CRM teams",
        factors_affecting_model_efficiency="Model assumes customer data is at least 90 days old"
    ),
    business_details=BusinessDetails(
        business_problem="Reduce monthly customer churn by 20%"
    )
)

model_card.create()
```

### AWS Audit Manager for ML

- Define control frameworks for ML compliance
- Continuously collect evidence from SageMaker, S3, CloudTrail
- Generate compliance reports (SOC 2, ISO 27001, GDPR)

### AWS CloudTrail for ML Audit

CloudTrail logs **every API call** to SageMaker, enabling full audit:

```
Events logged:
  - CreateTrainingJob
  - DeleteEndpoint
  - InvokeEndpoint (via API Gateway, not directly from endpoint)
  - UpdateModelPackage (approval changes)
  - CreatePipeline / StartPipelineExecution
  
Not logged by CloudTrail:
  - Actual model invocations (use Data Capture for this)
```

### SageMaker Role Manager

Simplifies IAM role creation for ML personas:

| Persona | Description |
|---------|-------------|
| **Data Scientist** | Can run training jobs, experiments, notebooks; no delete permissions |
| **MLOps Engineer** | Can deploy endpoints, manage pipelines; no access to raw data |
| **Data Engineer** | Can access S3, Glue, Athena; limited SageMaker permissions |

---

## 9. Cost Optimization

### Training Cost Strategies

| Strategy | Cost Savings | Trade-off |
|----------|-------------|-----------|
| **Managed Spot Training** | 60-90% | Interruption risk; requires checkpointing |
| **Amazon SageMaker Savings Plans** | 30-64% | 1 or 3 year commitment |
| **Right-size instances** | 20-50% | Performance testing needed |
| **Warm Pools** | Reduces startup overhead | Per-second cost while warm |
| **Distributed training efficiency** | Reduces total training time | Setup complexity |

### Inference Cost Strategies

| Strategy | Cost Savings | Notes |
|----------|-------------|-------|
| **Serverless Inference** | 100% when idle | Cold start penalty |
| **Async scale-to-zero** | 100% when idle | For non-real-time workloads |
| **AWS Graviton (c7g)** | 30-40% vs c5 | ARM architecture |
| **AWS Inferentia (inf1/inf2)** | 70% vs GPU | Requires Neuron SDK |
| **Multi-Model Endpoints** | 1 endpoint, many models | Same framework constraint |
| **Reserved Instances** | 30-40% vs on-demand | 1 or 3 year commitment |

### SageMaker Inference Recommender

Automatically tests your model against multiple instance types and provides:
- Optimal instance type for cost vs latency
- Benchmark results for each instance
- Real-time vs serverless comparison

### Cost Monitoring with AWS Cost Explorer

```python
import boto3

ce = boto3.client('ce')

# Get SageMaker cost breakdown
response = ce.get_cost_and_usage(
    TimePeriod={'Start': '2024-01-01', 'End': '2024-01-31'},
    Granularity='MONTHLY',
    Filter={
        'Dimensions': {
            'Key': 'SERVICE',
            'Values': ['Amazon SageMaker']
        }
    },
    GroupBy=[
        {'Type': 'DIMENSION', 'Key': 'USAGE_TYPE'}
    ],
    Metrics=['UnblendedCost']
)
```

### Cost Tags for ML Projects

```python
# Apply tags to all SageMaker resources for cost allocation
estimator = Estimator(
    ...,
    tags=[
        {'Key': 'Project', 'Value': 'churn-prediction'},
        {'Key': 'Team', 'Value': 'ml-platform'},
        {'Key': 'Environment', 'Value': 'production'},
        {'Key': 'CostCenter', 'Value': 'engineering-001'}
    ]
)
```

---

## 10. SageMaker ML Lineage Tracking

Lineage tracking automatically records all ML workflow artifacts and their relationships.

### Lineage Entities

| Entity | Description | Examples |
|--------|-------------|---------|
| **Artifact** | Input/output data or model | S3 datasets, model.tar.gz |
| **Action** | An operation that transforms artifacts | Training step, deployment |
| **Context** | Grouping of related entities | Experiment, pipeline run |
| **Association** | Relationship between entities | "Model was-trained-on Dataset" |

### Querying Lineage

```python
from sagemaker.lineage.context import Context
from sagemaker.lineage.artifact import Artifact
from sagemaker.lineage.association import Association
from sagemaker.lineage.query import LineageQuery, LineageQueryDirectionEnum

# Find all entities upstream of a deployed endpoint
lg = LineageQuery(sess)

# Upstream query — trace endpoint back to its training data
results = lg.query(
    start_arns=['arn:aws:sagemaker:us-east-1:123456789:endpoint/my-endpoint'],
    query_filter=LineageQueryFilter(
        created_before=datetime.datetime.now(),
        entity_types=['Artifact', 'Action']
    ),
    direction=LineageQueryDirectionEnum.ASCENDANTS,
    max_depth=10
)

# What data was used to train this model?
for artifact in results.artifacts():
    print(f"Artifact: {artifact.artifact_name} — {artifact.source.source_uri}")
```

---

## 11. Key Facts & Exam Tips

### Model Monitor Quick Reference

| Symptom | Monitor Type | Action |
|---------|-------------|--------|
| Input data looks different from training | Data Quality Monitor → baseline_drift_check | Retrain with new data |
| Model F1 dropped from 0.91 to 0.72 | Model Quality Monitor → degraded performance | Investigate + retrain |
| Model is now unfair to a demographic | Bias Drift Monitor → facet bias metric | Audit training data + retrain |
| Model using different features than at launch | Feature Attribution Drift → SHAP drift | Investigate concept drift |

### Security Best Practices Checklist

- [ ] S3 buckets encrypted with SSE-KMS
- [ ] SageMaker training in private VPC subnets
- [ ] `enable_network_isolation=True` for training containers
- [ ] `encrypt_inter_container_traffic=True` for multi-node jobs
- [ ] VPC Gateway Endpoint for S3 (avoids internet for S3 traffic)
- [ ] VPC Interface Endpoints for SageMaker API, ECR, CloudWatch
- [ ] Least-privilege IAM for all SageMaker roles
- [ ] Block public access on all ML S3 buckets
- [ ] Data capture enabled on production endpoints
- [ ] CloudTrail enabled and S3 logs encrypted
- [ ] `RootAccess: Disabled` on SageMaker notebooks
- [ ] `DirectInternetAccess: Disabled` on SageMaker notebooks

### Common Exam Scenarios

| Scenario | Best Solution |
|---------|--------------|
| Detect when endpoint input data drifts | SageMaker Data Quality Monitor |
| Track model fairness over time | SageMaker Bias Drift Monitor (Clarify) |
| Explain individual predictions to regulators | SageMaker Clarify SHAP explainability |
| Provide audit log of all model changes | AWS CloudTrail + Model Registry |
| Ensure training containers cannot call internet | `enable_network_isolation=True` |
| Prevent SageMaker from using internet for S3 | VPC + S3 Gateway Endpoint |
| Meet compliance requiring no data in transit unencrypted | `encrypt_inter_container_traffic` + SSE-KMS |
| Reduce inference costs for bursty workload | Serverless Inference |
| Model retraining should trigger automatically when drift detected | Model Monitor → CloudWatch Alarm → EventBridge → SageMaker Pipeline |
| Demonstrate explainability of each feature | Clarify SHAP analysis |

### Drift Detection Flow

```
Production Endpoint
      │ (Data Capture — 20% sampling)
      ▼
Amazon S3 (captured requests/responses)
      │
      ▼
Model Monitor (hourly/daily schedule compares to baseline)
      │
      ├── VIOLATION DETECTED
      │         │
      │         ▼
      │   CloudWatch Metric → Alarm → SNS → Lambda
      │                                        │
      │                               Start SageMaker Pipeline
      │                               (retrain with new data)
      │
      └── NO VIOLATION → continue monitoring
```

### Key Numbers to Remember

| Limit | Value |
|-------|-------|
| Real-time endpoint max payload | 6 MB |
| Serverless endpoint max payload | 4 MB |
| Async endpoint max payload | 1 GB |
| Async endpoint max processing time | 15 minutes |
| Batch Transform instances max | 100 |
| HPO max parallel jobs | Depends on account quota |
| Lambda max timeout (for endpoint trigger) | 15 minutes |
| Data Capture sampling max | 100% |
| Serverless max memory | 6 GB |
| Serverless max concurrency | 200 |
