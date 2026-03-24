# Lab Practice — Domain 3: Deployment and Orchestration of ML Workflows
## MLA-C01 | Hands-On Simulations | 6 Labs

> **Format:** Each lab provides a realistic scenario, step-by-step tasks, and verification checklists. Labs are designed for SageMaker Studio or notebook environments.

---

## Lab 1: Real-Time Endpoint — Deploy, Invoke, and Auto Scale

### Scenario
Deploy a trained XGBoost model to a SageMaker real-time endpoint. Configure auto scaling to handle variable traffic, then validate it scales correctly.

### Tasks

**Step 1 — Deploy the model to an endpoint**
```python
import sagemaker
import boto3

session  = sagemaker.Session()
bucket   = session.default_bucket()
role     = sagemaker.get_execution_role()
region   = session.boto_region_name

# Assume estimator from Domain 2 Lab 1 is trained
# estimator.deploy creates an endpoint directly from the training job
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="lab3-xgboost-endpoint",
    serializer=sagemaker.serializers.CSVSerializer(),
    deserializer=sagemaker.deserializers.JSONDeserializer(),
)
print("Endpoint deployed:", predictor.endpoint_name)
```

**Step 2 — Invoke the endpoint**
```python
import numpy as np

# Generate a test payload (20 features, no label column)
test_sample = np.random.randn(1, 20)
payload_str  = ",".join(str(v) for v in test_sample[0])

# Single prediction
result = predictor.predict(payload_str)
print("Prediction result:", result)

# Batch invocation (multiple rows in one request — up to 6 MB payload)
batch_samples = np.random.randn(100, 20)
batch_payload = "\n".join(",".join(str(v) for v in row) for row in batch_samples)
batch_result  = predictor.predict(batch_payload)
print(f"Batch predictions (first 5): {batch_result[:5]}")

# Direct boto3 invocation (what applications use in production)
runtime = boto3.client("sagemaker-runtime")
response = runtime.invoke_endpoint(
    EndpointName="lab3-xgboost-endpoint",
    ContentType="text/csv",
    Body=payload_str
)
print("Direct invoke result:", response["Body"].read().decode("utf-8"))
```

**Step 3 — Configure Application Auto Scaling**
```python
autoscaling = boto3.client("application-autoscaling")
sm_client   = boto3.client("sagemaker")

endpoint_name   = "lab3-xgboost-endpoint"
variant_name    = "AllTraffic"    # default variant name
resource_id     = f"endpoint/{endpoint_name}/variant/{variant_name}"

# Register the endpoint as a scalable target
autoscaling.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId=resource_id,
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=1,
    MaxCapacity=10,
)

# Target tracking: scale to keep ~70 invocations per instance per minute
autoscaling.put_scaling_policy(
    PolicyName="lab3-target-tracking",
    ServiceNamespace="sagemaker",
    ResourceId=resource_id,
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    PolicyType="TargetTrackingScaling",
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": 70.0,
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
        },
        "ScaleInCooldown":  300,   # 5min cooldown before scaling in
        "ScaleOutCooldown": 60,    # 1min cooldown before scaling out
    }
)
print("Auto scaling configured")
```

**Step 4 — Load test the endpoint**
```python
import concurrent.futures, time

def invoke_endpoint(i):
    sample = np.random.randn(1, 20)
    payload = ",".join(str(v) for v in sample[0])
    try:
        runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="text/csv",
            Body=payload
        )
        return True
    except Exception as e:
        return False

# Send 500 requests concurrently to trigger scale-out
print("Load testing — sending 500 concurrent requests...")
start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    results = list(executor.map(invoke_endpoint, range(500)))
elapsed = time.time() - start
print(f"Completed {sum(results)}/500 successful in {elapsed:.1f}s")

# Check if auto scaling added instances
response = sm_client.describe_endpoint(EndpointName=endpoint_name)
for variant in response["ProductionVariants"]:
    print(f"Current instance count: {variant['CurrentInstanceCount']}")
```

### Verification Checklist
- [ ] Endpoint shows `InService` in SageMaker console
- [ ] Single prediction returns a probability score between 0 and 1
- [ ] Auto scaling policy applied (visible in EC2 Auto Scaling console → SageMaker)
- [ ] After load test, instance count increases from 1 to >1
- [ ] `InvocationsPerInstance` CloudWatch metric rises above threshold during load test

---

## Lab 2: Asynchronous Inference & Serverless Inference

### Scenario
Test two specialized inference types: Async for large payloads with longer processing, and Serverless for traffic that drops to zero overnight.

### Tasks

**Step 1 — Deploy an Async Inference endpoint**
```python
from sagemaker.async_inference import AsyncInferenceConfig

async_config = AsyncInferenceConfig(
    output_path=f"s3://{bucket}/lab3-async/output/",   # results written here
    max_concurrent_invocations_per_instance=4,
    notification_config={
        "SuccessTopic": "arn:aws:sns:us-east-1:ACCOUNT_ID:async-success",
        "ErrorTopic":   "arn:aws:sns:us-east-1:ACCOUNT_ID:async-error",
    }
)

async_predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    endpoint_name="lab3-async-endpoint",
    async_inference_config=async_config,
)
print("Async endpoint deployed")
```

**Step 2 — Invoke async endpoint**
```python
from sagemaker.async_inference.waiter_config import WaiterConfig

# Upload large input to S3 (async accepts up to 1 GB payload)
large_batch = np.random.randn(10000, 20)
batch_df = pd.DataFrame(large_batch, columns=[f"f{i}" for i in range(20)])
batch_df.to_csv("large_batch.csv", index=False, header=False)
input_s3 = session.upload_data("large_batch.csv", bucket, "lab3-async/input")

# Invoke asynchronously — returns immediately, does NOT block
response = runtime.invoke_endpoint_async(
    EndpointName="lab3-async-endpoint",
    ContentType="text/csv",
    InputLocation=input_s3,
)
output_location = response["OutputLocation"]
print(f"Job submitted. Output will be at: {output_location}")

# Poll for result (or wait via SDK)
async_predictor.predict_async(
    input_path=input_s3,
    initial_args={"ContentType": "text/csv"}
).get_result(WaiterConfig(max_attempts=30, delay=10))  # poll every 10s
```

**Step 3 — Configure auto scale-to-zero for async**
```python
# Async endpoints support scale to 0 — no cost when idle
autoscaling.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId=f"endpoint/lab3-async-endpoint/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=0,            # scale to ZERO when no jobs
    MaxCapacity=5,
)

autoscaling.put_scaling_policy(
    PolicyName="async-scale-to-zero",
    ServiceNamespace="sagemaker",
    ResourceId=f"endpoint/lab3-async-endpoint/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    PolicyType="TargetTrackingScaling",
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": 5.0,
        "CustomizedMetricSpecification": {
            "MetricName": "ApproximateBacklogSizePerInstance",
            "Namespace":  "AWS/SageMaker",
            "Dimensions": [{"Name": "EndpointName", "Value": "lab3-async-endpoint"}],
            "Statistic":  "Average"
        }
    }
)
print("Scale-to-zero configured — endpoint scales down when queue is empty")
```

**Step 4 — Deploy a Serverless Inference endpoint**
```python
from sagemaker.serverless import ServerlessInferenceConfig

serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=2048,            # 1024, 2048, 3072, 4096, 5120, or 6144
    max_concurrency=10,                # max simultaneous requests
)

serverless_predictor = estimator.deploy(
    serverless_inference_config=serverless_config,
    endpoint_name="lab3-serverless-endpoint",
    serializer=sagemaker.serializers.CSVSerializer(),
)
print("Serverless endpoint deployed — zero cost when idle")

# Invoke (same API as real-time)
result = serverless_predictor.predict(payload_str)
print("Serverless result:", result)
# Note: first request after idle period has cold start (~1-2s extra latency)
```

### Verification Checklist
- [ ] Async endpoint: invoking returns `OutputLocation` immediately (non-blocking)
- [ ] Result file appears in the S3 output path after processing completes
- [ ] `MinCapacity=0` allows async endpoint to scale to zero instances
- [ ] Serverless endpoint has no instance count (it's serverless)
- [ ] Cold start latency is observable on first request after idle period
- [ ] Async endpoint supports payloads >>6 MB (test with 50 MB input)

---

## Lab 3: Multi-Model Endpoint (MME) — Host 1,000 Models on One Endpoint

### Scenario
An e-commerce company has 500 per-store sales models, all using the same XGBoost container. Deploy all models to a single MME endpoint to minimize costs.

### Tasks

**Step 1 — Prepare multiple model artifacts in S3**
```python
import tarfile, os

# Simulate 10 store models (use same XGBoost model artifact, different names)
base_model_path = "model.tar.gz"   # from earlier training job

s3 = boto3.client("s3")
model_prefix = "lab3-mme/models"

for store_id in range(1, 11):
    model_key = f"{model_prefix}/store_{store_id:03d}/model.tar.gz"
    s3.copy_object(
        CopySource={"Bucket": bucket, "Key": "lab2-xgb/output/model.tar.gz"},
        Bucket=bucket,
        Key=model_key
    )
    print(f"Uploaded store_{store_id:03d} model → s3://{bucket}/{model_key}")
```

**Step 2 — Create Mid and deploy MME**
```python
from sagemaker.multidatamodel import MultiDataModel
from sagemaker import image_uris

xgboost_image = image_uris.retrieve("xgboost", region, "1.7-1")

# MultiDataModel points to the S3 prefix containing all model artifacts
mme = MultiDataModel(
    name="lab3-store-models",
    model_data_prefix=f"s3://{bucket}/{model_prefix}/",
    image_uri=xgboost_image,
    role=role,
)

mme_predictor = mme.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    endpoint_name="lab3-mme-endpoint",
    serializer=sagemaker.serializers.CSVSerializer(),
    deserializer=sagemaker.deserializers.JSONDeserializer(),
)
print("MME endpoint deployed")
```

**Step 3 — Invoke with specific target model**
```python
# Call different models using TargetModel header
for store_id in [1, 5, 10]:
    model_name = f"store_{store_id:03d}/model.tar.gz"

    response = runtime.invoke_endpoint(
        EndpointName="lab3-mme-endpoint",
        ContentType="text/csv",
        TargetModel=model_name,       # key that selects which model to load
        Body=payload_str,
    )
    result = response["Body"].read().decode("utf-8")
    print(f"Store {store_id:03d}: prediction = {result}")
```

**Step 4 — Observe model loading behavior**
```python
# First invocation of a model = cold start (model loads from S3 → memory)
# Subsequent invocations = fast (model cached in memory)

import time

# Cold start timing
start = time.time()
runtime.invoke_endpoint(
    EndpointName="lab3-mme-endpoint",
    ContentType="text/csv",
    TargetModel="store_008/model.tar.gz",   # not invoked before = cold
    Body=payload_str
)
cold_latency = time.time() - start
print(f"Cold start latency: {cold_latency*1000:.0f}ms")

# Warm invocation
start = time.time()
runtime.invoke_endpoint(
    EndpointName="lab3-mme-endpoint",
    ContentType="text/csv",
    TargetModel="store_008/model.tar.gz",   # cached now = warm
    Body=payload_str
)
warm_latency = time.time() - start
print(f"Warm invocation latency: {warm_latency*1000:.0f}ms")
# Expected: cold ~500ms+, warm ~10ms
```

### Verification Checklist
- [ ] All 10 model artifacts exist in the S3 prefix
- [ ] MME endpoint shows `InService`
- [ ] `TargetModel` header routes correctly to different model artifacts
- [ ] Cold start is noticeably slower than warm invocation
- [ ] CloudWatch `ModelLoadingWaitTime` metric is non-zero on cold loads

---

## Lab 4: SageMaker Pipelines — End-to-End MLOps Workflow

### Scenario
Build a complete SageMaker Pipeline: preprocess → train → evaluate → conditional register. The pipeline should be parameterizable so the same pipeline can be run with different datasets.

### Tasks

**Step 1 — Define pipeline parameters**
```python
from sagemaker.workflow.parameters import (
    ParameterInteger, ParameterString, ParameterFloat
)

processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
training_instance_type    = ParameterString(name="TrainingInstanceType", default_value="ml.m5.xlarge")
model_approval_status     = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
auc_threshold             = ParameterFloat(name="AUCThreshold", default_value=0.80)
input_data                = ParameterString(name="InputData", default_value=f"s3://{bucket}/lab4/raw/")
```

**Step 2 — Processing Step**
```python
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

processor = SKLearnProcessor(
    framework_version="1.0-1", role=role,
    instance_type="ml.m5.xlarge",
    instance_count=processing_instance_count,
)

processing_step = ProcessingStep(
    name="PreprocessData",
    processor=processor,
    code="preprocessing.py",
    inputs=[
        ProcessingInput(source=input_data, destination="/opt/ml/processing/input")
    ],
    outputs=[
        ProcessingOutput(output_name="train", source="/opt/ml/processing/output/train",
                         destination=f"s3://{bucket}/pipeline-output/train"),
        ProcessingOutput(output_name="test",  source="/opt/ml/processing/output/test",
                         destination=f"s3://{bucket}/pipeline-output/test"),
    ]
)
```

**Step 3 — Training Step**
```python
from sagemaker.workflow.steps import TrainingStep
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

xgb_estimator = Estimator(
    image_uri=xgboost_image, role=role,
    instance_count=1, instance_type=training_instance_type,
    output_path=f"s3://{bucket}/pipeline-output/model/",
)
xgb_estimator.set_hyperparameters(
    objective="binary:logistic", num_round=100,
    eta=0.1, max_depth=6, eval_metric="auc"
)

training_step = TrainingStep(
    name="TrainModel",
    estimator=xgb_estimator,
    inputs={
        "train": TrainingInput(
            # Reference ProcessingStep output at runtime
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            content_type="text/csv"
        )
    }
)
```

**Step 4 — Evaluation Step + PropertyFile**
```python
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.properties import PropertyFile

evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"           # script writes this file
)

eval_processor = SKLearnProcessor(
    framework_version="1.0-1", role=role,
    instance_type="ml.m5.xlarge", instance_count=1,
)

evaluation_step = ProcessingStep(
    name="EvaluateModel",
    processor=eval_processor,
    code="evaluate.py",
    inputs=[
        ProcessingInput(
            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            source=processing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            destination="/opt/ml/processing/test"
        ),
    ],
    outputs=[
        ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation"
        )
    ],
    property_files=[evaluation_report]
)
```

**Step 5 — ConditionStep: register only if AUC > threshold**
```python
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.steps import CreateModelStep, RegisterModel
from sagemaker.workflow.fail_step import FailStep

# Read AUC from PropertyFile
model_auc = JsonGet(
    step_name=evaluation_step.name,
    property_file=evaluation_report,
    json_path="metrics.auc.value"       # path within evaluation.json
)

# Create Model (needed before RegisterModel)
from sagemaker.model import Model
model = Model(
    image_uri=xgboost_image,
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    sagemaker_session=session, role=role,
)
create_model_step = CreateModelStep(name="CreateModel", model=model)

# Register Model
register_step = RegisterModel(
    name="RegisterModel",
    estimator=xgb_estimator,
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["application/json"],
    inference_instances=["ml.m5.large", "ml.m5.xlarge"],
    transform_instances=["ml.m5.xlarge"],
    model_package_group_name="loan-approval-models",
    approval_status=model_approval_status,
)

# Fail if AUC is below threshold
fail_step = FailStep(
    name="FailLowAUC",
    error_message=f"Model AUC below threshold {auc_threshold}"
)

condition_step = ConditionStep(
    name="CheckAUCThreshold",
    conditions=[
        ConditionGreaterThanOrEqualTo(left=model_auc, right=auc_threshold)
    ],
    if_steps=[create_model_step, register_step],
    else_steps=[fail_step],
)
```

**Step 6 — Assemble and run the pipeline**
```python
from sagemaker.workflow.pipeline import Pipeline

pipeline = Pipeline(
    name="loan-approval-pipeline",
    parameters=[
        processing_instance_count,
        training_instance_type,
        model_approval_status,
        auc_threshold,
        input_data,
    ],
    steps=[
        processing_step,
        training_step,
        evaluation_step,
        condition_step,
    ],
    sagemaker_session=session,
)

pipeline.upsert(role_arn=role)
print("Pipeline upserted")

execution = pipeline.start(
    parameters={
        "AUCThreshold": 0.75,                     # override default
        "TrainingInstanceType": "ml.m5.xlarge",
    }
)
execution.wait()
print("Pipeline execution status:", execution.describe()["PipelineExecutionStatus"])
```

### Verification Checklist
- [ ] Pipeline graph shows all 5 steps in SageMaker Studio
- [ ] TrainingStep reads its input from ProcessingStep's output (runtime property reference)
- [ ] EvaluationStep produces `evaluation.json` in the output S3 path
- [ ] ConditionStep routes to RegisterModel when AUC > threshold
- [ ] Pipeline execution status = `Succeeded`
- [ ] Model appears in SageMaker Model Registry with `PendingManualApproval` status

---

## Lab 5: Blue/Green Deployment with Auto Rollback

### Scenario
Deploy a new model version to production with zero downtime using blue/green deployment. Configure auto rollback if error rate exceeds 5%.

### Tasks

**Step 1 — Get the existing endpoint configuration**
```python
sm_client   = boto3.client("sagemaker")
endpoint_name = "lab3-xgboost-endpoint"

# Current endpoint config (blue)
response = sm_client.describe_endpoint(EndpointName=endpoint_name)
current_config = response["EndpointConfigName"]
print("Current config (blue):", current_config)
```

**Step 2 — Create a new endpoint configuration (green)**
```python
# New model artifact (e.g., from a newer training job)
new_model_arn = "arn:aws:sagemaker:us-east-1:ACCOUNT_ID:model/lab3-xgboost-v2"

sm_client.create_endpoint_config(
    EndpointConfigName="lab3-xgboost-v2-config",
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "ModelName":   "lab3-xgboost-v2",
        "InitialInstanceCount": 1,
        "InstanceType": "ml.m5.large",
    }]
)
```

**Step 3 — Update endpoint with blue/green policy**
```python
sm_client.update_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName="lab3-xgboost-v2-config",
    DeploymentConfig={
        "BlueGreenUpdatePolicy": {
            "TrafficRoutingConfiguration": {
                "Type": "LINEAR",
                "LinearStepSize": {
                    "Type": "CAPACITY_PERCENT",
                    "Value": 10               # shift 10% every bake period
                },
                "WaitIntervalInSeconds": 300  # 5 min between shifts
            },
            "TerminationWaitInSeconds": 300,  # wait before terminating blue fleet
            "MaximumExecutionTimeoutInSeconds": 3600
        },
        "AutoRollbackConfiguration": {
            "Alarms": [
                {
                    "AlarmName": "lab3-high-error-rate"   # pre-created CloudWatch alarm
                }
            ]
        }
    }
)
print("Blue/green update initiated — traffic shifting 10% every 5 min")
```

**Step 4 — Create the CloudWatch alarm for auto rollback**
```python
import boto3

cw = boto3.client("cloudwatch")
cw.put_metric_alarm(
    AlarmName="lab3-high-error-rate",
    ComparisonOperator="GreaterThanThreshold",
    EvaluationPeriods=2,
    MetricName="ModelError",
    Namespace="AWS/SageMaker",
    Period=60,
    Statistic="Sum",
    Threshold=5.0,          # alarm if >5 errors in 2 consecutive 1-min periods
    AlarmDescription="Triggers auto rollback during blue/green deployment",
    Dimensions=[
        {"Name": "EndpointName", "Value": endpoint_name},
        {"Name": "VariantName",  "Value": "AllTraffic"},
    ],
    TreatMissingData="notBreaching",
)
print("Rollback alarm created")
```

**Step 5 — Monitor the deployment**
```python
import time

while True:
    response = sm_client.describe_endpoint(EndpointName=endpoint_name)
    status = response["EndpointStatus"]
    print(f"Endpoint status: {status}")
    
    if status in ("InService", "Failed", "RollingBack"):
        break
    time.sleep(30)

if status == "InService":
    print("Deployment completed successfully")
elif status == "RollingBack":
    print("Auto rollback triggered — alarm fired, reverting to blue")
```

### Verification Checklist
- [ ] Update endpoint call succeeds with `Updating` status
- [ ] Endpoint status shows traffic shifting (visible in console under Variants)
- [ ] If error rate exceeds alarm threshold, endpoint status transitions to `RollingBack`
- [ ] After successful deployment, all traffic routes to new model config
- [ ] Old (blue) endpoint config is terminated after `TerminationWaitInSeconds`

---

## Lab 6: EventBridge → Lambda → Automated Model Deployment

### Scenario
Build an automated deployment trigger: when a model is approved in the SageMaker Model Registry, EventBridge fires an event that Lambda catches and deploys the model to the production endpoint automatically.

### Tasks

**Step 1 — Write the Lambda function**

```python
# lambda_deploy.py — deploy approved model from Registry to endpoint
import boto3
import json

sm_client = boto3.client("sagemaker")

ENDPOINT_NAME    = "production-loan-approval"
ENDPOINT_CONFIG_PREFIX = "auto-deploy"

def lambda_handler(event, context):
    print("Event received:", json.dumps(event))

    # Extract model package ARN from EventBridge event
    model_package_arn    = event["detail"]["ModelPackageArn"]
    approval_status      = event["detail"]["ModelApprovalStatus"]

    if approval_status != "Approved":
        print(f"Status is {approval_status}, skipping deployment")
        return {"statusCode": 200, "body": "No deployment needed"}

    print(f"Deploying approved model: {model_package_arn}")

    # Create a new model from the approved package
    model_name = f"{ENDPOINT_CONFIG_PREFIX}-{context.aws_request_id[:8]}"
    sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={"ModelPackageName": model_package_arn},
        ExecutionRoleArn=context.invoked_function_arn.replace(":function:*", ":role/SageMakerRole"),
    )

    # Create endpoint config
    endpoint_config_name = f"{ENDPOINT_CONFIG_PREFIX}-{context.aws_request_id[:8]}"
    sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            "VariantName":    "AllTraffic",
            "ModelName":      model_name,
            "InitialInstanceCount": 1,
            "InstanceType":   "ml.m5.large",
        }]
    )

    # Update or create endpoint
    try:
        sm_client.update_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=endpoint_config_name
        )
        print(f"Endpoint {ENDPOINT_NAME} updated")
    except sm_client.exceptions.ResourceNotFound:
        sm_client.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=endpoint_config_name
        )
        print(f"Endpoint {ENDPOINT_NAME} created")

    return {"statusCode": 200, "body": f"Deploying {model_name}"}
```

**Step 2 — Create EventBridge rule**
```bash
# Create rule that matches SageMaker Model Package state changes
aws events put-rule \
  --name "ModelApprovedTrigger" \
  --event-pattern '{
    "source":      ["aws.sagemaker"],
    "detail-type": ["SageMaker Model Package State Change"],
    "detail": {
      "ModelApprovalStatus": ["Approved"]
    }
  }' \
  --state ENABLED

# Add Lambda as target
aws events put-targets \
  --rule ModelApprovedTrigger \
  --targets '[{
    "Id": "1",
    "Arn": "arn:aws:lambda:us-east-1:ACCOUNT_ID:function:auto-deploy-model"
  }]'

# Grant EventBridge permission to invoke the Lambda
aws lambda add-permission \
  --function-name auto-deploy-model \
  --statement-id EventBridgeInvoke \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn arn:aws:events:us-east-1:ACCOUNT_ID:rule/ModelApprovedTrigger
```

**Step 3 — Test the automation end-to-end**
```python
# Approve a model in the registry — this will trigger the EventBridge → Lambda flow
sm_client.update_model_package(
    ModelPackageArn="arn:aws:sagemaker:us-east-1:ACCOUNT_ID:model-package/loan-approval-models/1",
    ModelApprovalStatus="Approved"                # this fires the EventBridge event
)
print("Model approved — EventBridge event will fire in <1 second")

# Watch Lambda logs (wait ~30s)
import time; time.sleep(30)

# Check endpoint status
response = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
print("Endpoint status:", response["EndpointStatus"])
# Expected: Updating or InService
```

**Step 4 — Verify in CloudTrail**
```python
cloudtrail = boto3.client("cloudtrail")
response = cloudtrail.lookup_events(
    LookupAttributes=[{"AttributeKey": "EventName", "AttributeValue": "UpdateEndpoint"}],
    MaxResults=5
)
for event in response["Events"]:
    print(event["EventName"], event["EventTime"], event["Username"])
# Confirms Lambda (not a human) triggered the deployment
```

### Verification Checklist
- [ ] EventBridge rule matches `SageMaker Model Package State Change` with `Approved` status
- [ ] Lambda has permission to be invoked by EventBridge and to call SageMaker APIs
- [ ] Approving a model package triggers Lambda within <5 seconds
- [ ] Lambda creates/updates the production endpoint correctly
- [ ] CloudTrail shows the endpoint update was made by the Lambda execution role (not a human)

---

## Summary — Domain 3 Lab Skills Matrix

| Lab | Service / Concept | Skills Practiced |
|-----|------------------|-----------------|
| 1 | SageMaker Real-Time Endpoint | Deploy, invoke, Application Auto Scaling, target tracking |
| 2 | Async + Serverless Inference | Scale-to-zero, large payloads, cold start, output S3 location |
| 3 | Multi-Model Endpoint | Multiple model artifacts, TargetModel header, cold/warm load |
| 4 | SageMaker Pipelines | Steps, PropertyFile, JsonGet, ConditionStep, FailStep, parameters |
| 5 | Blue/Green Deployment | LINEAR traffic shift, AutoRollbackConfiguration, CloudWatch alarm |
| 6 | EventBridge + Lambda | Model Registry approval event, automated deployment, CloudTrail audit |

### Common Mistakes to Avoid
- **Real-Time endpoint**: Cannot set `MinCapacity=0` — it always needs at least 1 instance (use Serverless or Async for scale-to-zero)
- **MME**: All models must use the **same framework container** — cannot mix PyTorch and XGBoost in one MME
- **MME TargetModel**: The value is the **relative S3 path** within the prefix, not the full S3 URI
- **Pipelines PropertyFile**: `path` in `PropertyFile` refers to the filename inside the output directory, not the full S3 path
- **Pipelines ConditionStep**: `if_steps` and `else_steps` receive a list — single step must still be wrapped in `[...]`
- **Blue/Green**: The rollback CloudWatch alarm must exist **before** calling `update_endpoint`, not after
