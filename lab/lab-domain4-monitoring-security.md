# Lab Practice — Domain 4: Monitoring, Security, and Maintenance of ML Solutions
## MLA-C01 | Hands-On Simulations | 6 Labs

> **Format:** Each lab provides a realistic scenario, step-by-step tasks, and verification checklists. Labs are designed for SageMaker Studio or notebook environments.

---

## Lab 1: SageMaker Model Monitor — Data Quality Monitoring

### Scenario
A deployed credit-scoring model may experience data drift over time as customer profiles change. Set up Data Quality Monitoring to automatically detect when input feature distributions shift from the training baseline.

### Tasks

**Step 1 — Enable Data Capture on the endpoint**
```python
import sagemaker, boto3
from sagemaker.model_monitor import DataCaptureConfig

session       = sagemaker.Session()
bucket        = session.default_bucket()
role          = sagemaker.get_execution_role()
endpoint_name = "lab4-credit-scoring-endpoint"

data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,             # capture ALL requests during lab (use 10-20% in prod)
    destination_s3_uri=f"s3://{bucket}/lab4/data-capture/",
    capture_options=["REQUEST", "RESPONSE"],   # capture both input and output
)

# Deploy endpoint with data capture enabled
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=endpoint_name,
    data_capture_config=data_capture_config,
)
print("Endpoint with data capture deployed")
```

**Step 2 — Establish the baseline**
```python
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
)

monitor.suggest_baseline(
    baseline_dataset=f"s3://{bucket}/lab4/baseline/train.csv",
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri=f"s3://{bucket}/lab4/baseline-results/",
    wait=True,
    logs=True,
)

print("Baseline computed at:", monitor.latest_baselining_job.outputs[0].destination)
```

**Step 3 — Inspect baseline statistics and constraints**
```python
import json

baseline_config = monitor.suggested_baseline_statistics()
print("Baseline statistics JSON:")
print(json.dumps(baseline_config.body_dict, indent=2))
# Shows: per-feature mean, std, min, max, quantiles, completeness, type

baseline_constraints = monitor.suggested_baseline_constraints()
print("\nBaseline constraints JSON:")
print(json.dumps(baseline_constraints.body_dict, indent=2))
# Shows: allowed range, completeness threshold per feature
```

**Step 4 — Create a monitoring schedule**
```python
from sagemaker.model_monitor import CronExpressionGenerator

monitor.create_monitoring_schedule(
    monitor_schedule_name="lab4-data-quality-schedule",
    endpoint_input=endpoint_name,
    statistics=monitor.baseline_statistics(),
    constraints=monitor.suggested_baseline_constraints(),
    schedule_cron_expression=CronExpressionGenerator.hourly(),
    output_s3_uri=f"s3://{bucket}/lab4/monitoring-reports/",
    enable_cloudwatch_metrics=True,
)
print("Monitoring schedule created — runs hourly")
```

**Step 5 — Generate drifted traffic to trigger violations**
```python
import numpy as np
import pandas as pd

# Normal traffic (matches baseline distribution)
normal_data = np.random.randn(100, 20)

# Drifted traffic: shift features 3 standard deviations
drifted_data = np.random.randn(100, 20) + 3.0   # dramatic mean shift

runtime = boto3.client("sagemaker-runtime")

def send_batch(data, label="normal"):
    for row in data:
        payload = ",".join(str(v) for v in row)
        runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="text/csv",
            Body=payload
        )
    print(f"Sent {len(data)} {label} requests")

send_batch(normal_data, "normal")
send_batch(drifted_data, "drifted")
```

**Step 6 — Review violation reports**
```python
# After the next scheduled monitoring run:
executions = monitor.list_executions()
latest     = executions[-1]

print("Execution status:", latest.describe()["MonitoringExecutionStatus"])

# Read the violation report
report = latest.output.destination
s3 = boto3.client("s3")

# Parse S3 path
s3_parts = report.replace("s3://", "").split("/", 1)
violation_report_key = s3_parts[1] + "/constraint_violations.json"

obj = s3.get_object(Bucket=s3_parts[0], Key=violation_report_key)
violations = json.loads(obj["Body"].read())

for v in violations.get("violations", []):
    print(f"Feature: {v['feature_name']} | Type: {v['constraint_check_type']} | "
          f"Description: {v['description']}")
# Expected: dist_param violations for drifted features
```

### Verification Checklist
- [ ] Endpoint data capture files appear in S3 within 1 minute of first invocation
- [ ] Baseline statistics JSON contains per-feature statistics
- [ ] Monitoring schedule shows `Scheduled` status
- [ ] After sending drifted data, violation report contains `dist_param_check` violations
- [ ] CloudWatch metric `feature_baseline_drift` > 0 for drifted features

---

## Lab 2: SageMaker Clarify — Bias Detection and Explainability

### Scenario
Your loan approval model may treat applicants differently based on gender. Run SageMaker Clarify to detect both pre-training dataset bias and post-training model bias, then explain individual predictions.

### Tasks

**Step 1 — Run pre-training bias analysis on the dataset**
```python
from sagemaker import clarify

clarify_processor = clarify.SageMakerClarifyProcessor(
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    sagemaker_session=session,
)

bias_config = clarify.BiasConfig(
    label_values_or_threshold=[1],           # positive outcome = loan approved
    facet_name="gender",                     # protected attribute
    facet_values_or_threshold=[0],           # 0 = female (check if disadvantaged)
    group_name="age_group",                  # optional subgroup
)

data_config = clarify.DataConfig(
    s3_data_input_path=f"s3://{bucket}/lab4/train.csv",
    s3_output_path=f"s3://{bucket}/lab4/clarify-output/pre-training/",
    label="loan_approved",
    headers=["gender", "age", "income", "credit_score", "loan_approved"],
    dataset_type="text/csv",
)

clarify_processor.run_pre_training_bias(
    data_config=data_config,
    bias_config=bias_config,
    methods=["CI", "DPL", "KL", "JS", "LP", "TVD", "KS"],
    wait=True,
    logs=True,
)
print("Pre-training bias analysis complete")
```

**Step 2 — Interpret pre-training bias metrics**
```python
# Read the pre-training bias report
pre_bias_report = clarify_processor.latest_job.outputs[0].destination
s3_parts = pre_bias_report.replace("s3://", "").split("/", 1)
obj = s3.get_object(Bucket=s3_parts[0], Key=s3_parts[1] + "/analysis.json")
report = json.loads(obj["Body"].read())

pre_bias_metrics = report["pre_training_bias_metrics"]["facets"]["gender"][0]["metrics"]
for metric in pre_bias_metrics:
    print(f"{metric['name']:8s}: {metric['value']:+.4f}  — {metric['description']}")

# Key metrics to understand:
# CI (Class Imbalance): >0 means more privileged group samples
#   Ideal: close to 0
# DPL (Diff in Positive Proportion in Labels): >0 means privileged group has more positive labels
#   Ideal: close to 0 (|DPL| < 0.1 is generally acceptable)
```

**Step 3 — Run post-training bias analysis**
```python
model_config = clarify.ModelConfig(
    model_name="lab4-loan-model",
    instance_type="ml.m5.xlarge",
    instance_count=1,
    content_type="text/csv",
    accept_type="application/json",
)

predictions_config = clarify.ModelPredictedLabelConfig(
    probability_threshold=0.5               # threshold for binary classification
)

clarify_processor.run_post_training_bias(
    data_config=data_config,
    data_bias_config=bias_config,
    model_config=model_config,
    model_predicted_label_config=predictions_config,
    methods=["DPPL", "DI", "DCO", "RD", "DAR", "DRR", "AD", "CDDPL", "TE"],
    wait=True,
)

# Key post-training metrics:
# DPPL (Diff in Positive Proportion of Predicted Labels): did model amplify bias?
# DI   (Disparate Impact): ratio of positive prediction rates — <0.8 indicates problem
# DCO  (Diff in Conditional Outcomes): error rate difference between groups
print("Post-training bias analysis complete")
```

**Step 4 — Run SHAP explainability**
```python
shap_config = clarify.SHAPConfig(
    baseline=[
        [0, 35, 50000, 680]              # baseline: average applicant (no gender, median values)
    ],
    num_samples=100,                     # number of permutations per record
    agg_method="mean_abs",               # aggregate SHAP values by mean absolute
    save_local_shap_values=True,         # save per-record SHAP values
)

explainability_config = clarify.ExplainabilityConfig(
    shap_config=shap_config,
)

clarify_processor.run_explainability(
    data_config=data_config,
    model_config=model_config,
    explainability_config=explainability_config,
    wait=True,
)
```

**Step 5 — Read SHAP global feature importance**
```python
shap_output = clarify_processor.latest_job.outputs[0].destination
s3_parts = shap_output.replace("s3://", "").split("/", 1)
obj = s3.get_object(Bucket=s3_parts[0], Key=s3_parts[1] + "/explanations_shap/global_shap_values.csv")

import pandas as pd, io
shap_df = pd.read_csv(io.StringIO(obj["Body"].read().decode()))

print("\nGlobal Feature Importance (SHAP):")
feature_importance = shap_df.abs().mean().sort_values(ascending=False)
for feat, importance in feature_importance.items():
    bar = "█" * int(importance * 100)
    print(f"  {feat:15s}: {importance:.4f} {bar}")

# Interpret: higher = more influential on predictions
# If 'gender' is near top, model is using it heavily
```

### Verification Checklist
- [ ] Pre-training CI metric reveals class imbalance by group
- [ ] Pre-training DPL > 0 indicates different positive label rates by gender
- [ ] Post-training DI < 0.8 flags disparate impact (if present)
- [ ] SHAP global importance shows `credit_score` and `income` as top features
- [ ] If `gender` SHAP importance is high → model is using protected attribute → investigate

---

## Lab 3: Model Quality Monitor — Tracking Prediction Accuracy Over Time

### Scenario
Set up continuous model accuracy monitoring. The system collects live predictions and merges them with delayed ground truth labels to compute ongoing accuracy metrics.

### Tasks

**Step 1 — Create Model Quality Baseline**
```python
from sagemaker.model_monitor import ModelQualityMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

model_quality_monitor = ModelQualityMonitor(
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    volume_size_in_gb=20,
    max_runtime_in_seconds=1800,
    sagemaker_session=session,
)

# Baseline dataset: predictions on validation set with known ground truth
# Format: {"prediction": 0.9, "label": 1, "probability": 0.9}
model_quality_monitor.suggest_baseline(
    baseline_dataset=f"s3://{bucket}/lab4/model-quality-baseline.jsonl",
    dataset_format=DatasetFormat.json(lines=True),
    output_s3_uri=f"s3://{bucket}/lab4/model-quality-baseline-results/",
    problem_type="BinaryClassification",
    inference_attribute="prediction",       # field name of predicted label
    probability_attribute="probability",    # field name of predicted probability
    ground_truth_attribute="label",
    wait=True,
)
```

**Step 2 — Create the monitoring schedule**
```python
from sagemaker.model_monitor import EndpointInput

endpoint_input = EndpointInput(
    endpoint_name=endpoint_name,
    destination="/opt/ml/processing/input_data",
    inference_attribute="0",               # index of prediction in response
    probability_attribute="1",             # index of probability score in response
)

model_quality_monitor.create_monitoring_schedule(
    monitor_schedule_name="lab4-model-quality-schedule",
    endpoint_input=endpoint_input,
    ground_truth_input=f"s3://{bucket}/lab4/ground-truth/",   # labels uploaded here later
    problem_type="BinaryClassification",
    output_s3_uri=f"s3://{bucket}/lab4/model-quality-reports/",
    schedule_cron_expression=CronExpressionGenerator.hourly(),
    constraints=model_quality_monitor.suggested_baseline_constraints(),
    enable_cloudwatch_metrics=True,
)
print("Model quality monitoring schedule created")
```

**Step 3 — Simulate ground truth label delivery**
```python
import json, time, uuid

# In production, this would be your application posting actual outcomes
# Ground truth file format: one record per line, matching InferenceId
def upload_ground_truth(predictions_with_labels):
    """
    predictions_with_labels: list of {"inferenceId": "...", "groundTruthLabel": 0 or 1}
    """
    # Create a JSONL ground truth file with matching inference IDs
    gt_records = []
    for record in predictions_with_labels:
        gt_records.append(json.dumps({
            "groundTruthData": {
                "data":     str(record["groundTruthLabel"]),
                "encoding": "CSV"
            },
            "eventMetadata": {
                "eventId": record["inferenceId"]          # matches captured inference ID
            },
            "eventVersion": "0"
        }))

    # Upload to the ground truth S3 path
    gt_content = "\n".join(gt_records)
    current_hour = time.strftime("%Y/%m/%d/%H")
    key = f"lab4/ground-truth/{current_hour}/labels.jsonl"

    s3.put_object(Bucket=bucket, Key=key, Body=gt_content.encode())
    print(f"Uploaded {len(predictions_with_labels)} ground truth labels to s3://{bucket}/{key}")

# Simulate sending ground truth for some captured predictions
upload_ground_truth([
    {"inferenceId": f"id-{uuid.uuid4()}", "groundTruthLabel": 1},
    {"inferenceId": f"id-{uuid.uuid4()}", "groundTruthLabel": 0},
])
```

**Step 4 — Read model quality report metrics**
```python
# After monitor runs:
executions = model_quality_monitor.list_executions()
if executions:
    latest = executions[-1]
    print("Status:", latest.describe()["MonitoringExecutionStatus"])

    # Parse model quality report
    output_path = latest.output.destination
    s3_parts = output_path.replace("s3://", "").split("/", 1)

    try:
        obj = s3.get_object(
            Bucket=s3_parts[0],
            Key=s3_parts[1] + "/constraint_violations.json"
        )
        violations = json.loads(obj["Body"].read())
        print("Model quality violations:")
        for v in violations.get("violations", []):
            print(f"  {v['metric_name']}: {v['description']}")
    except s3.exceptions.NoSuchKey:
        print("No violations found (model quality within baseline)")
```

### Verification Checklist
- [ ] Model quality baseline produces constraints with AUC, F1, accuracy thresholds
- [ ] Monitoring schedule configured with `ground_truth_input` S3 path
- [ ] Ground truth JSONL file uploaded correctly with matching inference IDs
- [ ] Monitor execution appears in `list_executions()` after scheduled run
- [ ] CloudWatch metric `binary_classification_accuracy` exists for the endpoint

---

## Lab 4: IAM, VPC, and KMS Security for SageMaker

### Scenario
A company's ML platform requires strict security: SageMaker training jobs must run in a private VPC, model artifacts must be KMS-encrypted, and IAM roles must follow least privilege.

### Tasks

**Step 1 — Create a least-privilege IAM role for training**
```json
// training-role-policy.json — minimum permissions for a SageMaker training job
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3TrainingDataAccess",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::my-training-data-bucket",
        "arn:aws:s3:::my-training-data-bucket/*"
      ]
    },
    {
      "Sid": "S3ModelOutputAccess",
      "Effect": "Allow",
      "Action": ["s3:PutObject"],
      "Resource": "arn:aws:s3:::my-model-artifacts-bucket/training-output/*"
    },
    {
      "Sid": "KMSForEncryption",
      "Effect": "Allow",
      "Action": ["kms:GenerateDataKey", "kms:Decrypt"],
      "Resource": "arn:aws:kms:us-east-1:ACCOUNT_ID:key/KEY_ID"
    },
    {
      "Sid": "ECRImagePull",
      "Effect": "Allow",
      "Action": [
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:GetAuthorizationToken"
      ],
      "Resource": "*"
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
      "Resource": "arn:aws:logs:*:*:log-group:/aws/sagemaker/TrainingJobs:*"
    }
  ]
}
```

**Step 2 — Create KMS key for model artifact encryption**
```python
kms = boto3.client("kms")

key_response = kms.create_key(
    Description="SageMaker ML training artifact encryption key",
    KeyUsage="ENCRYPT_DECRYPT",
    Policy=json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "AllowSageMakerTrainingRole",
                "Effect": "Allow",
                "Principal": {"AWS": f"arn:aws:iam::ACCOUNT_ID:role/SageMakerTrainingRole"},
                "Action": ["kms:GenerateDataKey", "kms:Decrypt", "kms:ReEncrypt*"],
                "Resource": "*"
            },
            {
                "Sid": "AllowRootAccount",
                "Effect": "Allow",
                "Principal": {"AWS": f"arn:aws:iam::ACCOUNT_ID:root"},
                "Action": "kms:*",
                "Resource": "*"
            }
        ]
    })
)
kms_key_arn = key_response["KeyMetadata"]["Arn"]
print("KMS Key ARN:", kms_key_arn)

# Create alias for easy reference
kms.create_alias(
    AliasName="alias/sagemaker-ml-key",
    TargetKeyId=kms_key_arn
)
```

**Step 3 — Configure training job with VPC + KMS**
```python
from sagemaker.estimator import Estimator

secure_estimator = Estimator(
    image_uri=xgboost_image,
    role="arn:aws:iam::ACCOUNT_ID:role/SageMakerTrainingRole",   # least-privilege role
    instance_count=1,
    instance_type="ml.m5.xlarge",

    # VPC configuration — run in private subnet, no internet access
    subnets=["subnet-0abc123private"],          # private subnet ID (no internet gateway)
    security_group_ids=["sg-0xyz456ml"],        # security group allowing inter-container traffic

    # KMS encryption for training volume and model artifacts
    volume_kms_key=kms_key_arn,                 # encrypt instance EBS volume
    output_kms_key=kms_key_arn,                 # encrypt model artifacts in S3

    # Prevent container from making internet calls
    enable_network_isolation=False,             # False = can access VPC endpoints (S3, ECR)
                                                # True  = fully isolated, cannot access anything

    # Encrypt traffic between training nodes (multi-node jobs)
    encrypt_inter_container_traffic=True,

    output_path=f"s3://{bucket}/secure-training/output/",
)

secure_estimator.set_hyperparameters(
    objective="binary:logistic",
    num_round=100, eta=0.1, max_depth=6,
)
```

**Step 4 — Create S3 Gateway Endpoint (required for VPC-isolated training)**
```bash
# Create S3 Gateway VPC Endpoint — allows private subnet access to S3 without internet
aws ec2 create-vpc-endpoint \
  --vpc-id vpc-0abc123yourvpc \
  --service-name com.amazonaws.us-east-1.s3 \
  --route-table-ids rtb-0abc123private \
  --query "VpcEndpoint.VpcEndpointId"

# Create ECR Interface Endpoints (required to pull container images in VPC)
aws ec2 create-vpc-endpoint \
  --vpc-id vpc-0abc123yourvpc \
  --vpc-endpoint-type Interface \
  --service-name com.amazonaws.us-east-1.ecr.api \
  --subnet-ids subnet-0abc123private \
  --security-group-ids sg-0xyz456ml

aws ec2 create-vpc-endpoint \
  --vpc-id vpc-0abc123yourvpc \
  --vpc-endpoint-type Interface \
  --service-name com.amazonaws.us-east-1.ecr.dkr \
  --subnet-ids subnet-0abc123private \
  --security-group-ids sg-0xyz456ml

echo "VPC endpoints created — training jobs can now access S3 and ECR without internet"
```

**Step 5 — Verify security controls**
```python
# Verify training job uses VPC config
sm_client = boto3.client("sagemaker")
job_name  = secure_estimator.latest_training_job.name
response  = sm_client.describe_training_job(TrainingJobName=job_name)

vpc_config = response.get("VpcConfig", {})
print("VPC Subnets:", vpc_config.get("Subnets", []))
print("Security Groups:", vpc_config.get("SecurityGroupIds", []))
print("Volume KMS:", response.get("ResourceConfig", {}).get("VolumeKmsKeyId", "None"))
print("Output KMS:", response.get("OutputDataConfig", {}).get("KmsKeyId", "None"))
print("Network Isolation:", response.get("EnableNetworkIsolation"))
print("Inter-container TLS:", response.get("EnableInterContainerTrafficEncryption"))
```

### Verification Checklist
- [ ] IAM role has only the minimum required permissions (no `sagemaker:*` or `s3:*`)
- [ ] KMS key created and accessible by SageMaker training role
- [ ] Training job description shows `VpcConfig` with private subnet IDs
- [ ] `VolumeKmsKeyId` and `OutputDataConfig.KmsKeyId` are set to the CMK ARN
- [ ] Without S3 Gateway endpoint, training job fails with timeout on data access
- [ ] With S3 Gateway endpoint, training job accesses data successfully

---

## Lab 5: Model Monitor — Bias Drift and Feature Attribution Drift

### Scenario
Your loan model has been running for 6 months. Set up Clarify-based monitoring to detect if model bias or feature attribution patterns have changed over time.

### Tasks

**Step 1 — Compute bias drift baseline with Clarify**
```python
from sagemaker.model_monitor import ClarifyModelMonitor

bias_monitor = ClarifyModelMonitor(
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    volume_size_in_gb=20,
    max_runtime_in_seconds=1800,
    sagemaker_session=session,
)

model_config = clarify.ModelConfig(
    model_name="lab4-loan-model",
    instance_type="ml.m5.xlarge",
    instance_count=1,
    content_type="text/csv",
    accept_type="application/json",
)

bias_config = clarify.BiasConfig(
    label_values_or_threshold=[1],
    facet_name="gender",
    facet_values_or_threshold=[0],
)

# Generate baseline bias statistics
bias_monitor.suggest_baseline(
    data_config=clarify.DataConfig(
        s3_data_input_path=f"s3://{bucket}/lab4/baseline-with-predictions.csv",
        s3_output_path=f"s3://{bucket}/lab4/bias-baseline/",
        label="prediction",
        headers=["gender", "age", "income", "credit_score", "prediction"],
        dataset_type="text/csv",
    ),
    bias_config=bias_config,
    model_config=model_config,
    model_predicted_label_config=clarify.ModelPredictedLabelConfig(0.5),
    methods={"pre_training_bias": {"methods": ["CI", "DPL"]},
             "post_training_bias": {"methods": ["DPPL", "DI"]}},
    wait=True,
)
```

**Step 2 — Schedule bias drift monitoring**
```python
bias_monitor.create_monitoring_schedule(
    monitor_schedule_name="lab4-bias-drift-schedule",
    endpoint_input=clarify.EndpointInput(
        endpoint_name=endpoint_name,
        destination="/opt/ml/processing/input",
        features_attribute="features",
        inference_attribute="prediction",
    ),
    output_s3_uri=f"s3://{bucket}/lab4/bias-drift-reports/",
    schedule_cron_expression=CronExpressionGenerator.daily(),    # run daily
    enable_cloudwatch_metrics=True,
    baseline_config=bias_monitor.baseline_config_uri(),
)
```

**Step 3 — Schedule feature attribution drift monitoring**
```python
explainability_monitor = ClarifyModelMonitor(
    role=role, instance_count=1, instance_type="ml.m5.xlarge",
    volume_size_in_gb=20, max_runtime_in_seconds=1800,
    sagemaker_session=session,
)

# Baseline: SHAP values computed from Lab 2 Clarify run
explainability_monitor.create_monitoring_schedule(
    monitor_schedule_name="lab4-feature-attribution-schedule",
    endpoint_input=clarify.EndpointInput(
        endpoint_name=endpoint_name,
        destination="/opt/ml/processing/input",
    ),
    output_s3_uri=f"s3://{bucket}/lab4/attribution-drift-reports/",
    schedule_cron_expression=CronExpressionGenerator.daily(),
    enable_cloudwatch_metrics=True,
    explainability_config=clarify.ExplainabilityConfig(
        shap_config=clarify.SHAPConfig(
            baseline=[[0, 35, 50000, 680]],
            num_samples=50,
            agg_method="mean_abs",
        )
    ),
)
print("Bias drift + feature attribution monitors scheduled")
```

**Step 4 — Build event-driven retraining pipeline**
```python
import boto3

# CloudWatch alarm: fires when DI metric drops below fairness threshold
cw = boto3.client("cloudwatch")
cw.put_metric_alarm(
    AlarmName="lab4-bias-drift-alarm",
    ComparisonOperator="LessThanThreshold",
    EvaluationPeriods=1,
    MetricName="bias_metric:DPPL",           # published by Bias Drift Monitor
    Namespace="/aws/sagemaker/Endpoints/bias-metrics",
    Period=86400,                            # daily evaluation
    Statistic="Average",
    Threshold=0.1,                           # flag if DPPL > 0.1 (|DPPL| < 0.1 is OK)
    Dimensions=[{"Name": "MonitoringSchedule", "Value": "lab4-bias-drift-schedule"}],
    AlarmActions=["arn:aws:lambda:us-east-1:ACCOUNT_ID:function:trigger-retraining"],
)

# Lambda function (retraining trigger):
retraining_lambda_code = """
import boto3

def lambda_handler(event, context):
    sm = boto3.client("sagemaker")
    sm.start_pipeline_execution(
        PipelineName="loan-approval-pipeline",
        PipelineExecutionDisplayName="bias-drift-triggered-retraining",
        PipelineParameters=[
            {"Name": "AUCThreshold",    "Value": "0.78"},
            {"Name": "InputData",       "Value": "s3://bucket/latest-data/"},
        ]
    )
    return {"statusCode": 200}
"""
print("Retraining Lambda logic defined")
```

### Verification Checklist
- [ ] Bias drift baseline exists in S3 with bias statistics JSON
- [ ] Both monitoring schedules show `Scheduled` status
- [ ] CloudWatch publishes bias metrics (check in CloudWatch console)
- [ ] Alarm triggers Lambda when DPPL exceeds threshold
- [ ] Lambda starts the SageMaker Pipeline execution

---

## Lab 6: CloudTrail Audit + Model Cards

### Scenario
Your compliance team needs an audit trail of all ML operations and formal model documentation for regulatory review. Enable CloudTrail for SageMaker and create a Model Card.

### Tasks

**Step 1 — Verify CloudTrail captures SageMaker API calls**
```python
import boto3
from datetime import datetime, timedelta

cloudtrail = boto3.client("cloudtrail")

# Look up recent SageMaker API calls
response = cloudtrail.lookup_events(
    LookupAttributes=[
        {"AttributeKey": "EventSource", "AttributeValue": "sagemaker.amazonaws.com"}
    ],
    StartTime=datetime.utcnow() - timedelta(hours=1),
    MaxResults=20,
)

print("Recent SageMaker API calls:")
for event in response["Events"]:
    evt = json.loads(event["CloudTrailEvent"])
    print(f"  {event['EventTime'].strftime('%H:%M:%S')} | "
          f"{event['EventName']:40s} | "
          f"User: {evt.get('userIdentity', {}).get('userName', 'role-session')}")
```

**Step 2 — Audit model approval history**
```python
# Find who approved the production model
response = cloudtrail.lookup_events(
    LookupAttributes=[
        {"AttributeKey": "EventName", "AttributeValue": "UpdateModelPackage"}
    ],
    StartTime=datetime.utcnow() - timedelta(days=30),
    MaxResults=10,
)

print("\nModel approval history:")
for event in response["Events"]:
    evt    = json.loads(event["CloudTrailEvent"])
    user   = evt.get("userIdentity", {}).get("userName", "unknown")
    params = evt.get("requestParameters", {})
    print(f"  {event['EventTime'].strftime('%Y-%m-%d %H:%M')} | "
          f"Approved by: {user} | "
          f"Status: {params.get('modelApprovalStatus', 'N/A')}")
```

**Step 3 — Create a Model Card**
```python
sm_client = boto3.client("sagemaker")

response = sm_client.create_model_card(
    ModelCardName="loan-approval-model-card",
    Content=json.dumps({
        "model_overview": {
            "model_description": "XGBoost binary classifier for personal loan approval decisions",
            "model_owner":       "ML Platform Team",
            "model_artifact":    [f"s3://{bucket}/secure-training/output/model.tar.gz"],
            "algorithm_type":    "XGBoost",
            "model_id":          "loan-approval-v3",
        },
        "intended_uses": {
            "purpose_of_model":          "Automated loan approval pre-screening",
            "intended_uses":             "Score applicants before human underwriter review",
            "factors_affecting_performance": "Economic conditions, regional lending policies",
            "risk_rating":               "Medium",
            "explanations_for_risk_rating": "Decisions affect individuals financially; human review required",
        },
        "training_details": {
            "objective_function":        "Binary cross-entropy (binary:logistic)",
            "training_observations":     50000,
            "training_job_name":         "lab4-xgboost-2024-01-15",
        },
        "evaluation_details": [
            {
                "name": "Validation AUC",
                "evaluation_job_arn": "arn:aws:sagemaker:us-east-1:ACCOUNT_ID:processing-job/eval-job",
                "datasets":          [f"s3://{bucket}/lab4/test.csv"],
                "metric_groups": [
                    {
                        "name": "Binary Classification Metrics",
                        "metric_data": [
                            {"name": "AUC",       "type": "number", "value": 0.88},
                            {"name": "F1",        "type": "number", "value": 0.83},
                            {"name": "Accuracy",  "type": "number", "value": 0.87},
                        ]
                    }
                ]
            }
        ],
        "additional_information": {
            "ethical_considerations": "Pre-training DPL=0.04 indicates slight label imbalance by gender; post-training DI=0.86 above 0.8 threshold.",
            "caveats_and_recommendations": "Model requires quarterly bias audits. Decisions above $50K require human underwriter sign-off.",
        }
    }),
    ModelCardStatus="Draft",               # Draft → PendingReview → Approved → Archived
)

print("Model Card created:", response["ModelCardArn"])
```

**Step 4 — Export Model Card for auditors**
```python
# Export to PDF-compatible format for compliance team
export_response = sm_client.create_model_card_export_job(
    ModelCardName="loan-approval-model-card",
    ModelCardVersion=1,
    ModelCardExportJobName="loan-model-card-export-v1",
    OutputConfig={
        "S3OutputPath": f"s3://{bucket}/model-cards/exports/"
    }
)
print("Export job ARN:", export_response["ModelCardExportJobArn"])

# Share Model Card with stakeholders (transition to Approved)
sm_client.update_model_card(
    ModelCardName="loan-approval-model-card",
    ModelCardStatus="PendingReview"          # triggers review workflow
)
print("Model Card submitted for review")
```

### Verification Checklist
- [ ] CloudTrail shows all SageMaker API calls with user identity
- [ ] `UpdateModelPackage` events reveal who approved the model and when
- [ ] Model Card created with all required sections (overview, intended use, evaluation)
- [ ] Model Card status transitions: `Draft` → `PendingReview` → `Approved`
- [ ] Export job produces an artifact in S3 for auditor distribution
- [ ] No `AccessDenied` errors in CloudTrail (IAM roles are correctly configured)

---

## Summary — Domain 4 Lab Skills Matrix

| Lab | Service / Concept | Skills Practiced |
|-----|------------------|-----------------|
| 1 | Data Quality Monitor | Baseline, data capture, monitoring schedule, violation reports |
| 2 | SageMaker Clarify | Pre/post training bias metrics, SHAP explainability, global importance |
| 3 | Model Quality Monitor | Ground truth ingestion, inference IDs, accuracy tracking |
| 4 | IAM + VPC + KMS | Least-privilege roles, private VPC training, KMS CMK, VPC endpoints |
| 5 | Bias Drift + Attribution Drift | ClarifyModelMonitor, event-driven retraining, CloudWatch alarms |
| 6 | CloudTrail + Model Cards | API audit trail, Model Card lifecycle, compliance export |

### Common Mistakes to Avoid
- **Data Capture**: Setting `sampling_percentage=100` in production captures every request — can generate huge S3 costs; use 10–20%
- **Model Quality Monitor**: Forgetting to set `ground_truth_input` path → monitor runs but cannot compute accuracy metrics
- **Baseline**: Running baseline with production data instead of training data → baseline reflects real-world drift, not the intended training distribution
- **KMS**: Setting `volume_kms_key` but forgetting `output_kms_key` → model artifact in S3 is unencrypted even though the training volume was encrypted
- **VPC**: Adding a NAT Gateway when IAM/compliance forbids internet access — instead use VPC Interface Endpoints for ECR and S3 Gateway Endpoint
- **Model Cards**: Model Card must be in `Draft` status to edit — `PendingReview` or `Approved` cards are read-only until moved back
- **Clarify SHAP baseline**: Using all-zeros as baseline distorts SHAP values for features with non-zero means; use actual mean or median feature values
