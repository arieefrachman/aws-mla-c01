# Domain 3: Deployment and Orchestration of ML Workflows
## MLA-C01 Study Guide — 22% of Exam

---

## Table of Contents
1. [SageMaker Inference Types](#1-sagemaker-inference-types)
2. [Real-Time Inference Endpoints](#2-real-time-inference-endpoints)
3. [Serverless Inference](#3-serverless-inference)
4. [Asynchronous Inference](#4-asynchronous-inference)
5. [Batch Transform](#5-batch-transform)
6. [Advanced Endpoint Configurations](#6-advanced-endpoint-configurations)
7. [SageMaker Model Registry](#7-sagemaker-model-registry)
8. [SageMaker Pipelines (MLOps)](#8-sagemaker-pipelines-mlops)
9. [Inference Optimization](#9-inference-optimization)
10. [Orchestration with Other AWS Services](#10-orchestration-with-other-aws-services)
11. [Key Facts & Exam Tips](#11-key-facts--exam-tips)

---

## 1. SageMaker Inference Types

### Choosing the Right Inference Mode

| Mode | Latency | Payload Size | Instance | Best For |
|------|---------|-------------|----------|---------|
| **Real-Time** | Milliseconds | Up to 6 MB | Persistent (always on) | User-facing APIs, low latency required |
| **Serverless** | Seconds (cold start) | Up to 4 MB | Auto-scaled (0 → N) | Intermittent traffic, unpredictable load |
| **Asynchronous** | Minutes | Up to 1 GB | Persistent (scales to 0) | Large payloads, long processing, non-urgent |
| **Batch Transform** | Minutes to Hours | Unlimited | Ephemeral (only during job) | Offline scoring of large datasets |

### Decision Framework

```
Is the request user-facing (real-time UI/API)?
   YES → Real-Time Endpoint
   NO ↓
Is traffic intermittent/unpredictable OR you want to scale to zero?
   YES → Serverless Inference
   NO ↓
Is the payload large (> 6 MB) OR does processing take a long time?
   YES → Asynchronous Inference
   NO ↓
Are you scoring a large dataset offline (not individual requests)?
   YES → Batch Transform
```

---

## 2. Real-Time Inference Endpoints

### Creating an Endpoint

```python
import boto3
import sagemaker
from sagemaker.model import Model

# From a trained estimator
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.c5.xlarge',
    endpoint_name='my-realtime-endpoint'
)

# From a model artifact (manual)
model = Model(
    image_uri=image_uri,
    model_data='s3://my-bucket/model/model.tar.gz',
    role=role,
    env={'SAGEMAKER_CONTAINER_LOG_LEVEL': '20'}
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.c5.xlarge'
)
```

### Invoking the Endpoint

```python
import json
import boto3

sagemaker_runtime = boto3.client('sagemaker-runtime')

# Invoke with CSV
response = sagemaker_runtime.invoke_endpoint(
    EndpointName='my-realtime-endpoint',
    ContentType='text/csv',
    Accept='application/json',
    Body='1.2,3.4,5.6,7.8'
)
result = json.loads(response['Body'].read())

# Invoke with JSON
payload = {"instances": [{"features": [1.2, 3.4, 5.6]}]}
response = sagemaker_runtime.invoke_endpoint(
    EndpointName='my-realtime-endpoint',
    ContentType='application/json',
    Body=json.dumps(payload)
)
```

### Auto Scaling Endpoints

```python
import boto3

autoscaling = boto3.client('application-autoscaling')

# Register scalable target
autoscaling.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId='endpoint/my-endpoint/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=1,
    MaxCapacity=10
)

# Define scaling policy (Target Tracking)
autoscaling.put_scaling_policy(
    PolicyName='my-scaling-policy',
    ServiceNamespace='sagemaker',
    ResourceId='endpoint/my-endpoint/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 70.0,  # target 70% invocations per instance
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
        },
        'ScaleOutCooldown': 60,
        'ScaleInCooldown': 300
    }
)
```

### Endpoint Update (Blue/Green Deployment)

```python
import boto3
sm_client = boto3.client('sagemaker')

# Update endpoint with new model (gradual traffic shift)
sm_client.update_endpoint(
    EndpointName='my-endpoint',
    EndpointConfigName='new-endpoint-config',
    DeploymentConfig={
        'BlueGreenUpdatePolicy': {
            'TrafficRoutingConfiguration': {
                'Type': 'LINEAR',             # LINEAR, CANARY, ALL_AT_ONCE
                'LinearStepSize': {
                    'Type': 'CAPACITY_PERCENT',
                    'Value': 10               # shift 10% at a time
                },
                'WaitIntervalInSeconds': 300  # wait 5 min between shifts
            },
            'MaximumExecutionTimeoutInSeconds': 1800
        },
        'AutoRollbackConfiguration': {
            'Alarms': [
                {'AlarmName': 'my-error-alarm'}  # auto-rollback if alarm fires
            ]
        }
    }
)
```

### Traffic Splitting (A/B Testing)

```python
# Endpoint config with two variants
sm_client.create_endpoint_config(
    EndpointConfigName='ab-test-config',
    ProductionVariants=[
        {
            'VariantName': 'ModelA',
            'ModelName': 'model-version-1',
            'InstanceType': 'ml.c5.xlarge',
            'InitialInstanceCount': 1,
            'InitialVariantWeight': 0.9    # 90% traffic
        },
        {
            'VariantName': 'ModelB',
            'ModelName': 'model-version-2',
            'InstanceType': 'ml.c5.xlarge',
            'InitialInstanceCount': 1,
            'InitialVariantWeight': 0.1    # 10% traffic (canary)
        }
    ]
)
```

---

## 3. Serverless Inference

Serverless inference automatically provisions compute — no instance management needed.

### Configuration

```python
from sagemaker.serverless import ServerlessInferenceConfig

serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=2048,    # 1024, 2048, 3072, 4096, 5120, 6144
    max_concurrency=10,         # concurrent requests (1-200)
    provisioned_concurrency=2   # pre-warm instances (reduces cold start)
)

predictor = model.deploy(
    serverless_inference_config=serverless_config
)
```

### Serverless vs Real-Time Comparison

| Factor | Serverless | Real-Time |
|--------|-----------|----------|
| Instance management | None (automatic) | You choose instance type/count |
| Cold start | Yes (first request) | No |
| Scale to zero | Yes → $0 when idle | No → charge even at 0 TPS |
| Max memory | 6 GB | Depends on instance |
| Max payload | 4 MB | 6 MB |
| Ideal traffic | < 1 TPS or bursty | Consistent, high-throughput |

---

## 4. Asynchronous Inference

For workloads where requests may take up to **15 minutes** and payloads can be up to **1 GB**.

### Architecture

```
Client sends request   →  S3 (input payload)
                          ↓
                     Async Endpoint (SNS notification: success/failure)
                          ↓
                       S3 (output result)
                          ↓
                     Client polls S3 or receives SNS notification
```

### Configuration

```python
from sagemaker.async_inference import AsyncInferenceConfig

async_config = AsyncInferenceConfig(
    output_path='s3://my-bucket/async-output/',
    max_concurrent_invocations_per_instance=4,
    notification_config={
        'SuccessTopic': 'arn:aws:sns:us-east-1:123456789:ml-success',
        'ErrorTopic': 'arn:aws:sns:us-east-1:123456789:ml-error'
    }
)

predictor = model.deploy(
    instance_type='ml.m5.xlarge',
    initial_instance_count=1,
    async_inference_config=async_config
)

# Invoke async endpoint
response = sagemaker_runtime.invoke_endpoint_async(
    EndpointName='my-async-endpoint',
    InputLocation='s3://my-bucket/inputs/request.json',
    ContentType='application/json'
)

output_location = response['OutputLocation']  # poll this S3 path
```

### Async Endpoint Auto Scaling to Zero

```python
# Scale to 0 when no requests (unique to async endpoints)
autoscaling.put_scaling_policy(
    PolicyName='async-scale-to-zero-policy',
    ...
    TargetTrackingScalingPolicyConfiguration={
        'CustomizedMetricSpecification': {
            'MetricName': 'ApproximateBacklogSizePerInstance',
            'Namespace': 'AWS/SageMaker',
            'Dimensions': [{'Name': 'EndpointName', 'Value': 'my-async-endpoint'}],
            'Statistic': 'Average'
        },
        'TargetValue': 5.0
    }
)
```

> **Exam Tip:** Only Asynchronous Inference endpoints can scale to **zero instances** (no active requests → 0 cost).

---

## 5. Batch Transform

Batch Transform runs **offline inference** on large datasets.

### When to Use

- Score millions of records offline (daily/weekly batch jobs)
- No need for a persistent endpoint
- Large payloads (> 6 MB per record or entire dataset)
- Apply the same pre/post-processing to all records

### Running Batch Transform

```python
from sagemaker.transformer import Transformer

transformer = Transformer(
    model_name='my-model',
    instance_count=1,
    instance_type='ml.m5.2xlarge',
    output_path='s3://my-bucket/batch-output/',
    assemble_with='Line',           # how to join split predictions
    accept='text/csv',             # output MIME type
    max_payload=6,                 # MB per mini-batch
    max_concurrent_transforms=4
)

transformer.transform(
    data='s3://my-bucket/batch-input/',
    data_type='S3Prefix',
    content_type='text/csv',
    split_type='Line',             # split input by line
    join_source='Input'            # join input with output in result
)
```

### Batch Transform with SageMaker Pipelines

```python
from sagemaker.workflow.steps import TransformStep
from sagemaker.inputs import TransformInput

step_transform = TransformStep(
    name="BatchScoreStep",
    step_args=transformer.transform(
        data=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
        content_type="text/csv",
        split_type="Line"
    )
)
```

---

## 6. Advanced Endpoint Configurations

### Multi-Model Endpoints (MME)

Host **hundreds of models** on a single endpoint — models loaded on demand.

```python
from sagemaker.multidatamodel import MultiDataModel

# All models must use the SAME container/framework
mme = MultiDataModel(
    name='multi-model-endpoint',
    model_data_prefix='s3://my-bucket/models/',  # all model.tar.gz go here
    model=base_model,
    sagemaker_session=sess
)

predictor = mme.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.2xlarge'
)

# Invoke specific model
predictor.predict(
    data='1.2,3.4,5.6',
    target_model='customer-A-model.tar.gz'   # specify which model to use
)
```

**MME Use Case:** Serve thousands of tenant-specific models (e.g., one model per customer) without per-model endpoint costs.

### Multi-Container Endpoints (MCE)

Run **multiple different containers** on a single endpoint — serial or ensemble inference.

```python
# Serial inference pipeline: preprocessing → model → postprocessing
sm_client.create_model(
    ModelName='inference-pipeline-model',
    Containers=[
        {
            'Image': preprocessing_image,
            'ModelDataUrl': 's3://bucket/preprocessing.tar.gz'
        },
        {
            'Image': model_image,
            'ModelDataUrl': 's3://bucket/model.tar.gz'
        },
        {
            'Image': postprocessing_image,
            'ModelDataUrl': 's3://bucket/postprocessing.tar.gz'
        }
    ],
    ExecutionRoleArn=role
)
```

**Pipeline Mode (Serial):** Data flows sequentially through each container.
**Direct Access Mode:** Each container can be invoked independently.

### Elastic Inference

Attach fractional GPU acceleration to CPU instances for cost-effective inference.

```python
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',
    accelerator_type='ml.eia2.medium'  # attach fractional GPU
)
```

> **Exam Tip:** Elastic Inference is deprecated in new regions. AWS Inferentia instances (inf1, inf2) are the preferred cost-effective GPU inference solution.

### Inference Recommender

Automatically benchmarks instance types and recommends the optimal one for your model.

```python
# Run default recommendation job (tests multiple instances)
sm_client.create_inference_recommendations_job(
    JobName='my-recommendation-job',
    JobType='Default',   # 'Default' (quick) or 'Advanced' (comprehensive)
    RoleArn=role,
    InputConfig={
        'ModelPackageVersionArn': model_package_arn,
        'Job': {
            'ContentType': 'text/csv',
            'PayloadConfig': {
                'SamplePayloadUrl': 's3://bucket/sample-payload.csv',
                'SupportedContentTypes': ['text/csv']
            }
        }
    }
)
```

---

## 7. SageMaker Model Registry

Model Registry provides **central model versioning, metadata, and approval workflows**.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Model Package Group** | Container for model versions (e.g., `churn-prediction-models`) |
| **Model Package** | A specific model version with metadata, metrics, and artifacts |
| **Approval Status** | `Approved`, `Rejected`, `PendingManualApproval` |
| **Model Lineage** | Tracks which data + training job produced the model |
| **Model Card** | Governance document with intended use, metrics, limitations |

### Registering a Model

```python
from sagemaker.model import Model

model = Model(
    image_uri=image_uri,
    model_data=estimator.model_data,
    role=role
)

# Register to Model Registry
model_package = model.register(
    model_package_group_name='churn-prediction-models',
    inference_instances=['ml.m5.xlarge', 'ml.c5.xlarge'],
    transform_instances=['ml.m5.xlarge'],
    content_types=['text/csv'],
    response_types=['application/json'],
    approval_status='PendingManualApproval',  # requires human review
    description='XGBoost churn model v3.2 — AUC 0.91',
    model_metrics={
        'ModelQuality': {
            'Statistics': {
                'ContentType': 'application/json',
                'S3Uri': 's3://bucket/metrics/evaluation.json'
            }
        }
    }
)

print(f"Model Package ARN: {model_package.model_package_arn}")
```

### Approving a Model

```python
# Manual approval via SDK
sm_client.update_model_package(
    ModelPackageArn=model_package_arn,
    ModelApprovalStatus='Approved'
)

# Or via EventBridge → Lambda for automated approval based on metrics
# EventBridge rule: ModelPackage.PendingManualApproval → Lambda → approve if metrics pass
```

### Event-Driven Deployment from Registry

```python
# Lambda triggered by EventBridge when model is Approved
def lambda_handler(event, context):
    model_package_arn = event['detail']['ModelPackageArn']
    
    sagemaker_client = boto3.client('sagemaker')
    
    # Create model from approved package
    sagemaker_client.create_model(
        ModelName=f"approved-model-{int(time.time())}",
        Containers=[{'ModelPackageName': model_package_arn}],
        ExecutionRoleArn=role
    )
    
    # Then create endpoint config and update endpoint...
```

---

## 8. SageMaker Pipelines (MLOps)

Pipelines provides **CI/CD for ML** — define end-to-end workflows as DAGs.

### Pipeline Step Types

| Step | Description |
|------|-------------|
| `ProcessingStep` | Run SageMaker Processing Job |
| `TrainingStep` | Run SageMaker Training Job |
| `TuningStep` | Run HPO job (HyperparameterTuner) |
| `TransformStep` | Run Batch Transform |
| `ModelStep` | Register/create SageMaker Model |
| `ConditionStep` | Conditional branching (if/else) |
| `FailStep` | Explicitly fail pipeline |
| `LambdaStep` | Invoke Lambda function |
| `NotificationStep` | Send SNS notification |
| `QualityCheckStep` | Run data/model quality checks |
| `ClarifyCheckStep` | Run bias/explainability check |
| `EMRStep` | Run job on Amazon EMR |

### Full Pipeline Example

```python
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, TransformStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.condition_step import ConditionStep, ConditionGreaterThan
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import JsonGet

# --- Pipeline Parameters ---
input_data = ParameterString(name="InputData", default_value="s3://my-bucket/data/")
model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="Approved")
accuracy_threshold = ParameterFloat(name="AccuracyThreshold", default_value=0.85)

# --- Step 1: Preprocessing ---
step_process = ProcessingStep(
    name="PreprocessData",
    step_args=sklearn_processor.run(
        code="preprocessing.py",
        inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/train", output_name="train"),
            ProcessingOutput(source="/opt/ml/processing/test", output_name="test")
        ]
    )
)

# --- Step 2: Training ---
step_train = TrainingStep(
    name="TrainModel",
    step_args=xgb.fit({
        "train": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            content_type="text/csv"
        )
    }),
    depends_on=[step_process]
)

# --- Step 3: Evaluation ---
evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)

step_eval = ProcessingStep(
    name="EvaluateModel",
    step_args=sklearn_processor.run(
        code="evaluate.py",
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model"
            )
        ],
        outputs=[ProcessingOutput(source="/opt/ml/processing/evaluation", output_name="evaluation")]
    ),
    property_files=[evaluation_report],
    depends_on=[step_train]
)

# --- Step 4: Register Model (conditional on AUC threshold) ---
step_register = ModelStep(
    name="RegisterModel",
    step_args=model.register(
        model_package_group_name="churn-models",
        approval_status=model_approval_status,
        content_types=["text/csv"],
        response_types=["application/json"]
    )
)

step_fail = FailStep(
    name="FailPipeline",
    error_message=Join(on=" ", values=["AUC below threshold:", accuracy_threshold])
)

# --- Step 5: Condition (check AUC) ---
step_condition = ConditionStep(
    name="CheckAUCThreshold",
    conditions=[
        ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name=step_eval.name,
                property_file=evaluation_report,
                json_path="binary_classification_metrics.auc.value"
            ),
            right=accuracy_threshold
        )
    ],
    if_steps=[step_register],
    else_steps=[step_fail]
)

# --- Define Pipeline ---
pipeline = Pipeline(
    name="churn-prediction-pipeline",
    parameters=[input_data, model_approval_status, accuracy_threshold],
    steps=[step_process, step_train, step_eval, step_condition],
    sagemaker_session=sess
)

pipeline.upsert(role_arn=role)
pipeline.start()
```

### Pipeline Triggers

```
Manual:        pipeline.start()
Scheduled:     EventBridge Scheduler → pipeline.start_pipeline_execution()
Data-driven:   S3 Event → EventBridge → Lambda → pipeline.start_pipeline_execution()
CI/CD:         CodePipeline → CodeBuild → CLI command
```

### SageMaker Projects

Projects are pre-built **MLOps templates** powered by AWS Service Catalog.

| Template | Description |
|----------|-------------|
| **MLOps template for model building, training, and deployment** | End-to-end pipeline from S3 → endpoint |
| **MLOps template with third-party Git** | GitHub/GitLab-triggered pipelines |
| **MLOps template with Amazon CodePipeline** | Full CI/CD with code repo |

---

## 9. Inference Optimization

### SageMaker Neo (Model Compilation)

Compile and optimize models for specific hardware — up to **25x faster inference**.

```python
# Compile model for target hardware
compiled_model = estimator.compile_model(
    target_instance_family='ml_c5',  # target instance family
    input_shape={'data': [1, 3, 224, 224]},  # example input shape
    output_path='s3://my-bucket/compiled-model/',
    framework='pytorch',
    framework_version='1.13',
    compile_max_run=300
)

# Deploy compiled model
predictor = compiled_model.deploy(
    initial_instance_count=1,
    instance_type='ml.c5.xlarge'
)
```

### AWS Inferentia & Neuron SDK

Inferentia chips (`inf1`, `inf2`) provide high-throughput, low-cost inference.

```python
# Compile for Inferentia using Neuron SDK
from sagemaker.pytorch import PyTorchModel

model = PyTorchModel(
    model_data='s3://my-bucket/model.tar.gz',
    role=role,
    framework_version='1.13.1',
    py_version='py39',
    entry_point='inference.py'
)

predictor = model.deploy(
    instance_type='ml.inf1.xlarge',   # Inferentia instance
    initial_instance_count=1
)
```

### Inference Container Best Practices

| Optimization | Technique |
|-------------|-----------|
| **Model quantization** | FP32 → INT8 or FP16 → 4x smaller, faster |
| **Model pruning** | Remove unimportant weights |
| **Batch requests** | Process multiple requests together |
| **ONNX runtime** | Framework-agnostic optimized inference |
| **TensorRT** | NVIDIA-optimized GPU inference |
| **Triton Inference Server** | Multi-framework serving on GPU |

---

## 10. Orchestration with Other AWS Services

### AWS Step Functions for ML

Use Step Functions when orchestrating **non-SageMaker steps** or integrating with other AWS services.

```json
{
  "Comment": "ML Pipeline with Step Functions",
  "StartAt": "Start Glue ETL",
  "States": {
    "Start Glue ETL": {
      "Type": "Task",
      "Resource": "arn:aws:states:::glue:startJobRun.sync",
      "Parameters": {
        "JobName": "my-etl-job"
      },
      "Next": "Start SageMaker Training"
    },
    "Start SageMaker Training": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:createTrainingJob.sync",
      "Parameters": {
        "TrainingJobName.$": "$.trainingJobName",
        "AlgorithmSpecification": {
          "TrainingImage": "...",
          "TrainingInputMode": "File"
        },
        "RoleArn": "...",
        "OutputDataConfig": {"S3OutputPath": "s3://bucket/output/"},
        "ResourceConfig": {"InstanceType": "ml.m5.xlarge", "InstanceCount": 1}
      },
      "End": true
    }
  }
}
```

### Amazon MWAA (Managed Airflow)

Use MWAA when you have complex, multi-step DAGs with custom dependencies or need Airflow's ecosystem.

```python
# SageMaker operator for Airflow
from airflow.providers.amazon.aws.operators.sagemaker import SageMakerTrainingOperator

train_task = SageMakerTrainingOperator(
    task_id='train_model',
    config={
        'TrainingJobName': 'my-training-job',
        'AlgorithmSpecification': {'TrainingImage': '...', 'TrainingInputMode': 'File'},
        'RoleArn': '{{var.value.sagemaker_role}}',
        ...
    },
    wait_for_completion=True
)
```

### Comparison: SageMaker Pipelines vs Step Functions vs MWAA

| Criteria | SageMaker Pipelines | Step Functions | MWAA (Airflow) |
|---------|---------------------|---------------|----------------|
| **Best for** | ML-only workflows | Mixed AWS service workflows | Complex data engineering DAGs |
| **SageMaker integration** | Native, deep | API-based | Provider library |
| **Infrastructure** | Fully managed | Fully managed | Semi-managed (workers) |
| **Lineage tracking** | Yes (ML Lineage) | No | Limited |
| **Trigger types** | Manual/scheduled/event | Manual/event | DAG schedule/sensor |
| **Cost model** | Per pipeline step | Per state transition | Per environment + workers |

### AWS Lambda for ML Triggers

```python
# Lambda for event-driven inference (e.g., new S3 file → run model)
import boto3
import json

def lambda_handler(event, context):
    s3_key = event['Records'][0]['s3']['object']['key']
    bucket = event['Records'][0]['s3']['bucket']['name']
    
    sagemaker_runtime = boto3.client('sagemaker-runtime')
    
    # Read the uploaded file
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=s3_key)
    payload = obj['Body'].read().decode('utf-8')
    
    # Invoke endpoint
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName='my-production-endpoint',
        ContentType='text/csv',
        Body=payload
    )
    
    prediction = json.loads(response['Body'].read())
    
    # Store result back to S3
    s3.put_object(
        Bucket=bucket,
        Key=s3_key.replace('input/', 'predictions/'),
        Body=json.dumps(prediction)
    )
    
    return {'statusCode': 200, 'body': 'Processed'}
```

---

## 11. Key Facts & Exam Tips

### Inference Mode Quick Reference

| Scenario | Answer |
|---------|--------|
| User-facing API with consistent traffic | Real-Time Endpoint |
| Need to scale to zero cost when no traffic | Serverless Inference OR Async with scale-to-zero |
| Large payload (PDF, audio, 500 MB) | Asynchronous Inference |
| Score 10 million records nightly | Batch Transform |
| 1000 customer-specific models, cost-effective | Multi-Model Endpoint |
| Chain preprocessing → model → postprocessing | Inference Pipeline (Multi-Container) |
| Canary/blue-green model update | Endpoint Update with `BlueGreenUpdatePolicy` |
| A/B test two model versions | Traffic Splitting (variant weights) |

### Deployment Cost Optimization

| Strategy | Savings |
|----------|---------|
| Serverless (intermittent traffic) | 0 cost when idle |
| Async endpoint scaling to zero | 0 cost when backlog is empty |
| Graviton instances (ml.c7g) | 30-40% cheaper than x86 |
| AWS Inferentia (inf1/inf2) | 70% cheaper than GPU per inference |
| SageMaker Neo compilations | Smaller model → smaller instance needed |
| Reserved Instance commitments | 30-40% cheaper than on-demand |

### MLOps CI/CD Pattern

```
Code Commit → CodePipeline
                    │
                    ├── Unit tests (CodeBuild)
                    ├── Trigger SageMaker Pipeline (data + train + eval + register)
                    ├── Manual approval gate (human review of model metrics)
                    └── Deploy to staging endpoint → integration tests
                                └── Deploy to production (blue/green)
```

### Common Exam Pitfalls

- **Batch Transform** ≠ Batch processing of requests — it's for offline scoring
- **Multi-Model Endpoint** = same framework, many models. **Multi-Container Endpoint** = different frameworks, few models
- **Serverless** has cold start but scales to zero; **Real-Time** has no cold start but is always on
- **Only Async endpoints** can scale instance count to zero
- **SageMaker Pipelines** uses S3 URIs for inter-step data — steps don't directly pass data in memory
- **Model Registry approval** does NOT automatically deploy — you must create an endpoint separately or use EventBridge to trigger deployment
