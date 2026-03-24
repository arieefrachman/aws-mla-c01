# Mock Exam — Domain 3: Deployment and Orchestration of ML Workflows
## MLA-C01 | 20 Questions | ~52 minutes

> **Instructions:** Choose the single best answer unless the question says "Choose TWO."
> Answers and explanations are at the bottom — don't peek!

---

### Question 1
A company deploys a real-time SageMaker inference endpoint. During peak hours, the endpoint receives 500 requests per second, and during off-peak hours it receives fewer than 5. The team wants the endpoint to scale automatically without managing instance counts manually. Which configuration should they apply?

- A) Deploy using Serverless Inference with high `max_concurrency`
- B) Configure Application Auto Scaling with `SageMakerVariantInvocationsPerInstance` target tracking
- C) Deploy multiple endpoints and use Route 53 to round-robin traffic
- D) Use Batch Transform with a 15-minute interval trigger

---

### Question 2
An ML team deploys a model that processes medical image files. Each image is approximately 300 MB. Cardiologists upload images and expect results within 2–3 minutes, but do not need sub-second responses. Which inference type is BEST suited for this use case?

- A) Real-Time Inference endpoint
- B) Serverless Inference
- C) Asynchronous Inference
- D) Batch Transform

---

### Question 3
A company maintains 8,000 customer-specific regression models, all built using the same XGBoost container. They cannot afford to run 8,000 separate endpoints. What deployment solution should they use?

- A) Deploy one endpoint with all models hard-coded in the container
- B) SageMaker Multi-Model Endpoint (MME) — all model artifacts in a shared S3 prefix
- C) SageMaker Multi-Container Endpoint (MCE) with 8,000 containers
- D) Deploy in batches of 100 endpoints, use load balancing

---

### Question 4
A data science team wants to automate the full ML lifecycle: ingest data → preprocess → train → evaluate → register in Model Registry → deploy if performance threshold is met. The entire workflow involves ONLY SageMaker services. Which orchestration tool is the NATIVE best choice?

- A) AWS Step Functions with SageMaker SDK calls
- B) Amazon MWAA (Managed Airflow)
- C) SageMaker Pipelines
- D) AWS CodePipeline + CodeBuild

---

### Question 5
After a new model version is registered in the SageMaker Model Registry with status `PendingManualApproval`, a team lead approves it via the SageMaker Studio UI. The team wants this approval to **automatically trigger** deployment to a production endpoint without manual steps. Which AWS service should they use to implement this automation?

- A) SageMaker Pipelines scheduled with a cron expression
- B) Amazon EventBridge rule matching the `ModelPackage` state change → AWS Lambda
- C) AWS CodeBuild polling the Model Registry every 5 minutes
- D) Amazon SQS trigger when Model Registry is updated

---

### Question 6
A company is running an inference pipeline in SageMaker where raw text input must be tokenized (container 1), passed to a BERT model (container 2), then postprocessed into a readable category label (container 3). All three containers must be invoked in sequence for every request. Which endpoint configuration supports this?

- A) SageMaker Multi-Model Endpoint with 3 models
- B) Three separate endpoints chained via Lambda
- C) SageMaker Multi-Container Endpoint in Serial (Pipeline) mode
- D) SageMaker Real-Time endpoint with inline preprocessing in the model

---

### Question 7
A company deploys a new version of a recommendation model to production. They want to route 5% of live traffic to the new model version and keep 95% on the existing model, comparing business metrics for 2 weeks before a full cutover. What is this deployment strategy called, and how is it configured in SageMaker?

- A) Blue/Green deployment — use `update_endpoint` with `BlueGreenUpdatePolicy`
- B) Canary testing — use production variants with `InitialVariantWeight=0.05` and `0.95`
- C) Shadow deployment — deploy the new model but ignore its outputs
- D) Rolling update — replace instances one at a time

---

### Question 8
A SageMaker Pipelines workflow has a `ConditionStep` that checks if model AUC > 0.88. If true, it should register the model. If false, the pipeline should explicitly fail and alert the team. Which step should be placed in the `else_steps` of the `ConditionStep`?

- A) Another `ConditionStep` with a lower threshold
- B) `FailStep` with a descriptive error message
- C) `ProcessingStep` to retrain the model
- D) `LambdaStep` to send an SNS notification (and continue)

---

### Question 9
An ML engineer uses SageMaker Batch Transform to score 50 million records stored as CSV in S3. They want each output record to include the original input row alongside the prediction. Which Batch Transform parameter achieves this?

- A) `assemble_with='Line'`
- B) `split_type='Line'`
- C) `join_source='Input'`
- D) `content_type='text/csv'`

---

### Question 10
A company has an endpoint serving predictions. They are deploying a new model and want zero downtime — the old model should keep serving traffic while the new model is provisioned. As the new model passes health checks, traffic shifts in 10% increments every 5 minutes. If error rate exceeds a CloudWatch alarm threshold, the deployment should automatically revert. Which deployment approach handles all of these requirements?

- A) Delete the old endpoint and create a new one
- B) Blue/green deployment with `LINEAR` traffic routing and `AutoRollbackConfiguration`
- C) Update endpoint with immediate `ALL_AT_ONCE` traffic shift
- D) Use production variants and manually shift weights via SDK

---

### Question 11
An ML team wants to compile their PyTorch model to run optimally on `ml.c5` CPU instances, achieving lower latency with no code changes to the inference script. Which SageMaker feature should they use?

- A) SageMaker JumpStart model optimization
- B) SageMaker Neo model compilation targeting `ml_c5`
- C) AWS Inferentia with the Neuron SDK
- D) SageMaker Debugger profiler recommendations

---

### Question 12
A company has sporadic ML inference traffic — the endpoint is idle for 12 hours per day but has short bursts. They want **zero cost** during idle periods and accept a startup delay of 1–2 seconds for the first request after an idle period. Which endpoint type meets these requirements?

- A) Real-Time endpoint with auto scaling min=0
- B) Asynchronous endpoint with scale-to-zero policy
- C) Serverless Inference
- D) Batch Transform triggered by SQS messages

---

### Question 13
A SageMaker Pipeline includes the following steps in order: `ProcessingStep → TrainingStep → EvaluationStep → ConditionStep`. The `EvaluationStep` must access the model artifact produced by the `TrainingStep`. How does SageMaker Pipelines pass this data between steps?

- A) Steps share an in-memory object store managed by SageMaker
- B) The model artifact S3 URI is passed via `step_train.properties.ModelArtifacts.S3ModelArtifacts`
- C) The team must hardcode the S3 path in each step definition
- D) SageMaker Pipelines automatically mounts shared EFS between all steps

---

### Question 14
An ML engineer wants SageMaker Batch Transform to split a large CSV input file into mini-batches of at most 5 MB each, send each batch to the model, and reassemble the results. Which parameters control this behavior? (Choose TWO)

- A) `split_type='Line'` — split input by line boundaries
- B) `max_payload=5` — set max 5 MB per mini-batch
- C) `instance_count=5` — use 5 instances for parallelism
- D) `assemble_with='None'` — don't reassemble output
- E) `content_type='application/x-recordio'`

---

### Question 15
A team builds an ML workflow that: (1) runs an AWS Glue ETL job, (2) trains a SageMaker model, (3) writes results to DynamoDB, and (4) sends an SNS notification. The workflow must support retries, error handling, and conditional branching. Which orchestration tool is MOST appropriate?

- A) SageMaker Pipelines — it supports all AWS service integrations
- B) AWS Step Functions — designed for orchestrating diverse AWS services
- C) Amazon MWAA — required for any multi-service workflow
- D) AWS Lambda chained via SQS

---

### Question 16
During a SageMaker Pipeline execution, the `TrainingStep` fails because the training instance ran out of disk space. The pipeline does not retry and moves the execution to FAILED status. How should the pipeline be configured to retry this step up to 3 times before failing?

- A) Set `retry_policies` on the `TrainingStep` with `StepExceptionTypeList=['SageMaker.JOB_INTERNAL_ERROR']` and `MaxAttempts=3`
- B) Wrap the TrainingStep in a Lambda with a try-catch
- C) Set `max_retry_attempts=3` on the Estimator object
- D) Configure auto-restart in the SageMaker Training Job definition

---

### Question 17
A company needs to serve a large language model (LLM) with 70 billion parameters for real-time inference. The model is too large for a single GPU but the team wants fast token generation. Which instance type and configuration is MOST appropriate?

- A) `ml.g4dn.xlarge` — single T4 GPU, cost-effective
- B) `ml.c5.4xlarge` — CPU inference for LLMs
- C) `ml.p4d.24xlarge` with model parallelism across 8 A100 GPUs
- D) `ml.inf2.48xlarge` — Inferentia2 for LLMs (AWS Neuron)

---

### Question 18
A model is deployed to a SageMaker endpoint. The MLOps team builds a pipeline in SageMaker Pipelines. They want to pass the AUC metric from an evaluation step's output JSON file into a `ConditionStep`. Which SageMaker Pipelines construct enables reading a value from a JSON file produced by a step?

- A) `ParameterFloat` with a default value
- B) `PropertyFile` combined with `JsonGet` function
- C) Direct S3 URI reference in the condition
- D) `ExecutionVariable` pointing to the processing output

---

### Question 19
A team registers a model in the SageMaker Model Registry. The model package includes evaluation metrics showing AUC of 0.92. Three months later, the same model package is still deployed but the team has lost track of which training job and dataset produced it. Which SageMaker feature helps them trace the full lineage?

- A) SageMaker Experiments — search by metric value
- B) SageMaker ML Lineage Tracking — query upstream entities from the endpoint
- C) CloudTrail logs — search for the CreateTrainingJob API call
- D) SageMaker Debugger output artifacts

---

### Question 20
A company has a SageMaker real-time endpoint serving a classification model. They deploy a new model and want to test it on production traffic without affecting the user experience — the new model receives traffic in parallel, but its outputs are discarded and only the existing model's outputs are returned. What deployment pattern describes this?

- A) A/B Testing with variant weights 50/50
- B) Canary deployment with 1% traffic
- C) Shadow deployment (shadow variant)
- D) Blue/green deployment with 0% traffic to green

---

## ✅ Answers & Explanations

---

**Q1 — Answer: B (Application Auto Scaling with target tracking)**
SageMaker endpoints support Application Auto Scaling. `SageMakerVariantInvocationsPerInstance` is the standard metric for target tracking policies — SageMaker adds/removes instances to keep invocations-per-instance near the target. Serverless (A) is also valid for variable traffic but has cold start issues at 500 RPS scale. Route 53 (C) doesn't scale instances.

---

**Q2 — Answer: C (Asynchronous Inference)**
Asynchronous Inference supports payloads up to 1 GB and processing times up to 15 minutes — perfect for 300 MB medical images with 2–3 minute processing. Real-Time (A) has a 6 MB payload limit. Serverless (B) has a 4 MB limit. Batch Transform (D) is for offline dataset scoring, not request/response APIs used by cardiologists.

---

**Q3 — Answer: B (Multi-Model Endpoint)**
SageMaker MME is purpose-built for hosting many models on shared infrastructure. Models are stored as `model.tar.gz` files in an S3 prefix and loaded on demand when invoked. The constraint is all models must use the same container — which is satisfied here (all XGBoost). MCE (C) is for different frameworks (max ~15 containers, not 8,000).

---

**Q4 — Answer: C (SageMaker Pipelines)**
SageMaker Pipelines is the native MLOps workflow engine for SageMaker. It has first-class steps for every SageMaker operation (Processing, Training, RegisterModel, ConditionStep, etc.), built-in parameter management, caching, and ML Lineage. Step Functions (A) and CodePipeline (D) require external wiring to SageMaker APIs. MWAA (B) is better for complex multi-system DAGs.

---

**Q5 — Answer: B (EventBridge rule → Lambda)**
SageMaker Model Registry emits an EventBridge event when a model package approval status changes. An EventBridge rule matching this event triggers a Lambda function that creates or updates the SageMaker endpoint. This is the standard event-driven CI/CD pattern for model deployment. Scheduled Pipelines (A) poll, not event-driven. SQS (D) doesn't integrate directly with Model Registry events.

---

**Q6 — Answer: C (Multi-Container Endpoint in Serial mode)**
SageMaker MCE with Serial (Pipeline) mode chains multiple containers: request flows through container 1 → 2 → 3 sequentially. This is an Inference Pipeline — the right pattern for multi-step preprocessing → model → postprocessing. MME (A) hosts many models of the SAME type, doesn't chain them. Three separate endpoints + Lambda (B) adds latency and complexity.

---

**Q7 — Answer: B (Canary testing with production variant weights)**
This is canary testing — production variants allow weight-based traffic splitting. Setting `InitialVariantWeight=0.05` on the new model and `0.95` on the old routes 5%/95% of live traffic. Blue/green (A) with `BlueGreenUpdatePolicy` is for automated deployment shifts, not extended side-by-side comparison. Shadow (C) discards the new model's output.

---

**Q8 — Answer: B (FailStep)**
`FailStep` is a SageMaker Pipelines step designed to explicitly terminate the pipeline execution with a custom error message when a condition is not met. This is the correct way to signal that model quality is insufficient. `LambdaStep` (D) could send a notification but wouldn't fail the pipeline.

---

**Q9 — Answer: C (`join_source='Input'`)**
`join_source='Input'` tells Batch Transform to append the original input data to each output prediction, producing combined rows of `[input_features, prediction]`. `split_type='Line'` (B) controls how input is split (important but not the "join" feature). `assemble_with` (A) controls how outputs are reassembled.

---

**Q10 — Answer: B (Blue/Green with LINEAR routing + AutoRollbackConfiguration)**
`BlueGreenUpdatePolicy` with `LINEAR` traffic type shifts traffic incrementally (e.g., 10% every 5 min) while the old (blue) stack keeps serving. `AutoRollbackConfiguration` automatically reverts if a CloudWatch alarm fires. `ALL_AT_ONCE` (C) has no zero-downtime benefit. Manual variant weights (D) don't provide automatic rollback.

---

**Q11 — Answer: B (SageMaker Neo)**
SageMaker Neo compiles and optimizes models for specific hardware targets. `target_instance_family='ml_c5'` produces an optimized binary for C5 CPUs, reducing inference latency significantly with no code changes to the inference script. Inferentia (C) requires Neuron SDK code changes. JumpStart (A) doesn't optimize existing custom models.

---

**Q12 — Answer: C (Serverless Inference)**
Serverless Inference scales to zero automatically — zero cost when idle, and provisions compute on demand. A 1–2s startup (cold start) for the first request is expected and acceptable here. Real-time endpoints (A) cannot have min=0 instances — they maintain at least 1. Async endpoints (B) can scale to zero but have a more complex request/response model than described.

---

**Q13 — Answer: B (step_train.properties.ModelArtifacts.S3ModelArtifacts)**
SageMaker Pipelines uses **runtime properties** — special references that resolve to actual S3 URIs when the pipeline executes. You reference `step_train.properties.ModelArtifacts.S3ModelArtifacts` (or `.ProcessingOutputConfig.Outputs[...].S3Output.S3Uri` for processing), and SageMaker automatically threads the correct S3 path between steps. No in-memory sharing (A) or EFS (D) is used.

---

**Q14 — Answer: A and B (`split_type='Line'` + `max_payload=5`)**
`split_type='Line'` tells SageMaker to split the CSV file at line boundaries into mini-batches. `max_payload=5` (in MB) sets the maximum size of each mini-batch sent to the model. Together they control how input is split and batched. `instance_count=5` (C) controls parallelism but not the per-batch size. `join_source` assembles results.

---

**Q15 — Answer: B (AWS Step Functions)**
Step Functions is designed for orchestrating diverse AWS services (Glue, SageMaker, DynamoDB, SNS) with built-in retry logic, error handling, and conditional branching via native state machine language. SageMaker Pipelines (A) is purpose-built for SageMaker-only workflows. MWAA (C) is heavier and requires more setup. Lambda chains (D) are fragile without state management.

---

**Q16 — Answer: A (retry_policies on the TrainingStep)**
SageMaker Pipelines' `TrainingStep` (and other step types) supports `retry_policies` with configurable exception types and max attempts. `StepExceptionTypeList` can include `SageMaker.JOB_INTERNAL_ERROR` for infrastructure failures. This is native pipeline retry logic — separate from Lambda or Training Job configuration.

---

**Q17 — Answer: C or D**
For a 70B parameter LLM: `ml.p4d.24xlarge` with 8x A100 GPUs (C) supports model parallelism for large models. `ml.inf2.48xlarge` (D) has 12 Inferentia2 chips connected with NeuronLink, which is AWS's recommended solution for large-scale LLM inference. In practice, D is the exam-preferred answer for cost-effective LLM inference. `ml.g4dn.xlarge` (A) has only 1 T4 GPU — far too small for 70B parameters. CPU inference (B) is impractically slow.

---

**Q18 — Answer: B (PropertyFile + JsonGet)**
`PropertyFile` declares that a processing step will produce a JSON file at a specific path. `JsonGet` is a pipeline function that reads a value from that declared property file at runtime, enabling the `ConditionStep` to compare actual metric values dynamically. `ParameterFloat` (A) is for static pipeline parameters, not dynamic values from steps.

---

**Q19 — Answer: B (SageMaker ML Lineage Tracking)**
SageMaker ML Lineage automatically records relationships between artifacts (datasets, models, endpoints) and actions (training jobs, deployments). Starting from a deployed endpoint ARN, you can trace upstream through model → training job → input dataset. Experiments (A) tracks metrics per run but doesn't provide full lineage graphs. CloudTrail (C) shows API calls but not ML-specific lineage relationships.

---

**Q20 — Answer: C (Shadow deployment)**
A shadow deployment routes live traffic to the shadow (new) model in parallel with the production model, but the shadow model's outputs are **not returned to the user** — they are captured for offline analysis. This allows safe evaluation of the new model on real production traffic with zero user impact. A/B Testing (A) returns BOTH models' outputs to users (different users get different responses). Canary (B) returns the canary model's output to the 1% of users it serves.
