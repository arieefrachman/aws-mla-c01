# Exam Tips & Practice Questions
## MLA-C01 Study Guide — Strategies & Sample Questions

---

## Exam Strategy

### Time Management
- **170 minutes / 65 scored questions** = ~2.6 minutes per question
- Flag questions you're unsure about and return to them
- Don't spend > 3 minutes on any single question
- Reserve 15 minutes at the end to review flagged questions

### Question Analysis Approach
1. **Read the LAST sentence first** — that's usually the actual question
2. **Identify the constraint** — "most cost-effective", "lowest latency", "without changing code"
3. **Eliminate obviously wrong answers** — usually 2 of 4 are clearly wrong
4. **Choose the most specific AWS answer** — they prefer native AWS services over generic solutions
5. **When in doubt, choose the most managed/serverless option**

### Common Question Patterns

| Pattern | Example | Key Words |
|---------|---------|-----------|
| **Best service for X** | "Which service should you use to..." | "best", "most appropriate" |
| **Most cost-effective** | "How to reduce training costs?" | "save money", "cost-effective", "cheapest" |
| **Minimize operational overhead** | "Reduce maintenance burden?" | "least overhead", "fully managed", "serverless" |
| **Security requirement** | "Ensure data never leaves VPC?" | "compliance", "never over internet", "private" |
| **Specific constraint** | "Without retraining the model?" | Read carefully for business constraints |

---

## Domain 1 Practice Questions: Data Preparation

**Q1.** A data science team needs to build ML features from raw transaction data stored in S3. They want a visual interface to join tables, handle missing values, and detect bias in the dataset — all within SageMaker Studio. Which service should they use?

- A) AWS Glue DataBrew
- B) SageMaker Data Wrangler
- C) Amazon Athena
- D) AWS Glue Studio

**Answer: B** — Data Wrangler is the visual data prep tool inside SageMaker Studio with bias detection (via Clarify). Glue DataBrew is standalone. Athena is SQL only. Glue Studio is ETL pipeline builder, not feature engineering.

---

**Q2.** A machine learning engineer is training a model on a 500 GB dataset stored in S3. The training instance has 32 GB of memory. Which S3 input mode should they use to avoid downloading the entire dataset before training begins?

- A) File Mode
- B) Pipe Mode
- C) FastFile Mode
- D) Transfer Acceleration Mode

**Answer: C** — FastFile Mode streams data from S3 with a POSIX-compatible interface, supporting any format. Pipe Mode also streams but only works with RecordIO format. File Mode downloads everything first (won't fit in memory). "Transfer Acceleration Mode" is not a SageMaker input mode.

---

**Q3.** A company has thousands of customers and wants to ensure that real-time inference uses the same features computed during training, preventing training-serving skew. What should they implement?

- A) Store features in Amazon S3 and retrieve them at inference time
- B) Use Amazon DynamoDB as a feature cache
- C) Use SageMaker Feature Store with Online Store enabled
- D) Re-run the preprocessing script at inference time

**Answer: C** — Feature Store with Online Store enabled provides consistent, low-latency feature retrieval for both training (offline store) and real-time inference (online store), eliminating training-serving skew.

---

**Q4.** A data engineer needs to run nightly ETL jobs that read from RDS, transform data with PySpark, and write Parquet files to S3 for ML training. The solution should require minimal infrastructure management. Which service is MOST appropriate?

- A) Amazon EMR with persistent cluster
- B) AWS Glue ETL job (Spark)
- C) SageMaker Processing Job with PySparkProcessor
- D) Amazon EC2 with Apache Spark

**Answer: B** — AWS Glue ETL is serverless Spark, ideal for scheduled ETL with minimal infrastructure management. EMR requires cluster management. SageMaker Processing is also valid but Glue is more purpose-built for scheduled ETL pipelines.

---

**Q5.** An analyst discovers that 35% of records in a training dataset have NULL values in a key feature. Which approach is LEAST likely to introduce bias?

- A) Drop all rows with NULL values
- B) Impute with the mean value
- C) Impute based on other correlated features using KNN
- D) Replace NULLs with 0

**Answer: C** — KNN imputation uses patterns from similar records, preserving relationships between features. Dropping rows (A) introduces selection bias. Mean imputation (B) ignores feature relationships. Replacing with 0 (D) distorts the distribution.

---

## Domain 2 Practice Questions: Model Development

**Q6.** A company wants to predict customer churn (yes/no) from tabular data with 50 features. The dataset has 500,000 rows with a 95/5 class imbalance. Which algorithm and configuration is MOST appropriate?

- A) Linear Learner with `predictor_type='binary_classifier'` and `positive_example_weight_mult='balanced'`
- B) K-Means with k=2
- C) PCA to reduce dimensions then KNN
- D) BlazingText in classification mode

**Answer: A** — Linear Learner with balanced weighting handles binary classification and class imbalance natively. K-Means is unsupervised/clustering. PCA+KNN is valid but more complex. BlazingText is for text data.

---

**Q7.** A team is running hyperparameter tuning with 50 total jobs. They want to find the best hyperparameters as efficiently as possible while minimizing the number of training jobs. Which strategy should they choose?

- A) Grid Search
- B) Random Search
- C) Bayesian Optimization
- D) Hyperband

**Answer: C** — Bayesian Optimization uses results from previous evaluations to intelligently choose the next hyperparameter values, requiring fewer jobs to converge than Grid or Random Search. Hyperband is better for deep learning with epochs.

---

**Q8.** A data scientist trains an XGBoost model for fraud detection. The model achieves 99% accuracy but only 15% recall on the fraud class. What should they do FIRST?

- A) Increase `num_round` to train longer
- B) Reduce `max_depth` to prevent overfitting
- C) Set `scale_pos_weight` to the ratio of negative to positive examples
- D) Collect more non-fraud data

**Answer: C** — The model is biased toward the majority class due to severe class imbalance. `scale_pos_weight` in XGBoost adjusts the cost of misclassifying positive (fraud) examples, improving recall. Collecting more non-fraud data (D) would worsen imbalance.

---

**Q9.** A neural network training job is failing with exploding gradients after 10 epochs. The team wants to be alerted automatically when this happens. Which SageMaker feature should they use?

- A) SageMaker Experiments
- B) SageMaker Debugger with the `ExplodingGradient` rule
- C) Amazon CloudWatch custom metrics
- D) SageMaker Clarify

**Answer: B** — SageMaker Debugger has built-in rules like `ExplodingGradient` that monitor training tensors in real time and can trigger actions (stop job, send alert) when conditions are met.

---

**Q10.** A company wants to fine-tune a pre-trained BERT model for sentiment classification without managing containers or finding the right Docker image. What is the EASIEST approach?

- A) Build a custom Docker container with HuggingFace Transformers
- B) Use SageMaker JumpStart with a pre-built BERT model
- C) Download BERT from HuggingFace manually and run a Training Job
- D) Use Amazon Comprehend for custom entity recognition

**Answer: B** — SageMaker JumpStart provides ready-to-use pre-trained models including BERT variants with simplified fine-tuning APIs. No container management required.

---

**Q11.** A training job takes 8 hours and costs $50. The team wants to reduce training costs for this job, which runs daily. Which option provides the MOST cost savings?

- A) Switch from ml.p3.2xlarge to ml.p3.8xlarge (more GPUs, faster)
- B) Use Managed Spot Training with checkpointing enabled
- C) Use Warm Pools to keep instances ready
- D) Use SageMaker Savings Plans

**Answer: B** — Managed Spot Training can save 60-90% on training costs. At 80% savings, $50 → $10/day. Savings Plans provide 30-64% savings but require commitment. A larger instance may speed up training but costs more per hour. Warm Pools reduce startup overhead, not per-job cost.

---

## Domain 3 Practice Questions: Deployment & Orchestration

**Q12.** A company has 10,000 customer-specific models, all built with XGBoost, that need to be served via API. Running one endpoint per customer is too expensive. What is the MOST cost-effective deployment approach?

- A) Deploy all models in a single Multi-Container Endpoint
- B) Use SageMaker Multi-Model Endpoint (MME)
- C) Use SageMaker Serverless Inference with one endpoint per model
- D) Store all models in S3 and load them in a Lambda function

**Answer: B** — Multi-Model Endpoint (MME) is designed exactly for this scenario — many models sharing infrastructure, loaded on demand. All models must use the same container (XGBoost in this case). Multi-Container Endpoints are for different frameworks, not many models.

---

**Q13.** An ML engineer deploys a new model version and wants to send 10% of traffic to the new model while keeping 90% on the old model to validate performance in production. What should they configure?

- A) Create two separate endpoints and use Route 53 weighted routing
- B) Configure production variants with `InitialVariantWeight: 0.9` and `0.1`
- C) Update the endpoint with a canary deployment policy at 10% traffic
- D) Use A/B testing with Amazon CloudWatch

**Answer: B or C** — Both are correct approaches. Production Variants (B) allows percentage-based traffic splitting. Blue/Green Canary policy (C) also shifts traffic incrementally. B is more explicit and directly answers the question.

---

**Q14.** A machine learning inference endpoint processes large video files (up to 800 MB) for content moderation. Users submit requests and check results later. Which inference type should be used?

- A) Real-Time Inference
- B) Serverless Inference
- C) Asynchronous Inference
- D) Batch Transform

**Answer: C** — Asynchronous Inference supports payloads up to 1 GB and processing up to 15 minutes. Real-time supports only 6 MB. Serverless supports 4 MB. Batch Transform is for offline scoring, not request/response APIs.

---

**Q15.** A company runs a SageMaker inference endpoint that has very sporadic traffic — sometimes 0 requests for hours, then sudden bursts. They want to minimize costs during idle periods. What should they configure?

- A) Use Real-Time endpoint with auto scaling to 0 instances
- B) Use Serverless Inference
- C) Use Asynchronous Inference endpoint with scale-to-zero
- D) Use Batch Transform with a trigger

**Answer: B or C** — Serverless Inference scales to zero automatically and is best for intermittent traffic. Asynchronous Inference can also scale to zero (unique among persistent endpoints). Real-time endpoints **cannot** scale to 0 — they have a minimum of 1 instance.

---

**Q16.** A data science team wants to automate their ML workflow: preprocess data → train model → evaluate → register if performance is good → deploy. When performance drops below a threshold, the pipeline should fail and alert the team. Which service provides this natively?

- A) AWS Step Functions
- B) Amazon MWAA
- C) SageMaker Pipelines with ConditionStep and FailStep
- D) AWS CodePipeline with Lambda functions

**Answer: C** — SageMaker Pipelines with `ConditionStep` (for threshold check) and `FailStep` (to explicitly fail the pipeline) is purpose-built for ML workflows with ML Lineage, parameter management, and native SageMaker integration.

---

**Q17.** After a model is registered in SageMaker Model Registry with `PendingManualApproval` status, a human reviewer approves it. How can the approval automatically trigger deployment to production?

- A) SageMaker Pipelines automatically deploys approved models
- B) Configure an EventBridge rule to detect `ModelPackage` state change → trigger Lambda → create/update endpoint
- C) Use the Model Registry's built-in auto-deploy feature
- D) Set a CloudWatch alarm on approval status changes

**Answer: B** — EventBridge detects the `UpdateModelPackage` event (status change to Approved) and routes it to a Lambda function that creates the endpoint. Model Registry has no built-in auto-deploy — it requires separate automation.

---

## Domain 4 Practice Questions: Monitoring, Security, & Maintenance

**Q18.** A production model's prediction accuracy has degraded from 91% to 74% over six months. The underlying data distribution has not changed significantly. What type of monitor should be used to detect this earlier next time?

- A) SageMaker Data Quality Monitor
- B) SageMaker Model Quality Monitor
- C) SageMaker Bias Drift Monitor
- D) Feature Attribution Drift Monitor

**Answer: B** — Model Quality Monitor tracks prediction accuracy/quality metrics (F1, AUC, RMSE) against ground truth labels. Data Quality Monitor only checks input distribution, not model performance. Since the data distribution is intact, the issue is model degradation.

---

**Q19.** A financial services company needs to ensure that their ML training data NEVER travels over the public internet. All training should occur within their corporate network connected to AWS via Direct Connect. Which configurations are required? (Choose TWO)

- A) Enable encryption at rest on S3
- B) Use a VPC with private subnets for SageMaker training
- C) Create an S3 VPC Gateway Endpoint in the VPC
- D) Enable Managed Spot Training
- E) Use AWS PrivateLink for the SageMaker API

**Answer: B and C** — Training in a VPC with private subnets ensures compute doesn't touch the internet. An S3 VPC Gateway Endpoint routes S3 traffic within the VPC (otherwise S3 traffic exits VPC over internet). E (PrivateLink) provides private API access but isn't required for data not traveling over internet. A is encryption, not routing.

---

**Q20.** A healthcare company detects that their ML model is consistently under-predicting cancer risk for patients over 65. Which SageMaker capability can quantify this bias and provide a report for regulatory compliance?

- A) SageMaker Debugger with a custom rule
- B) SageMaker Model Monitor with data quality baseline
- C) SageMaker Clarify with post-training bias analysis
- D) SageMaker Experiments with metric comparison

**Answer: C** — Clarify runs post-training bias analysis using metrics like DCO (Difference in Conditional Outcomes) and RD (Recall Difference) to quantify performance differences across demographic groups. This generates reports suitable for regulatory review.

---

**Q21.** A company needs to comply with data residency regulations requiring that ML training data processed in us-east-1 never moves to any other region. Which mechanism enforces this?

- A) S3 bucket versioning
- B) IAM policy with `aws:RequestedRegion` condition key denying other regions
- C) S3 Block Public Access
- D) AWS Config rule checking bucket policies

**Answer: B** — An IAM policy with the `aws:RequestedRegion` condition combined with an explicit deny prevents any API call (including cross-region S3 operations or SageMaker in other regions) from being made in unauthorized regions.

---

**Q22.** A security audit finds that SageMaker notebook instances are accessible over the internet and can install arbitrary packages. Which TWO configurations should be enforced via IAM Service Control Policies?

- A) `sagemaker:DirectInternetAccess = Disabled`
- B) `sagemaker:RootAccess = Disabled`
- C) `sagemaker:InstanceTypes` limited to approved types
- D) `sagemaker:VolumeKmsKey` must be specified

**Answer: A and B** — `DirectInternetAccess: Disabled` prevents notebooks from calling the internet. `RootAccess: Disabled` prevents notebook users from getting root shell access (which could bypass security controls). Both are standard compliance requirements.

---

## High-Frequency Exam Topics Checklists

### SageMaker Deployment Types — Must Know Cold

| Type | Max Payload | Latency | Scales to 0? | Best For |
|------|------------|--------|-------------|---------|
| Real-Time | 6 MB | ms | ❌ | User-facing APIs |
| Serverless | 4 MB | s (cold start) | ✅ | Intermittent traffic |
| Async | 1 GB | min | ✅ (backlog=0) | Large/long requests |
| Batch | Unlimited | hours | N/A (ephemeral) | Offline scoring |

### Built-In Algorithm Data Formats — Must Know Cold

| Format | Algorithms |
|--------|----------|
| **RecordIO-Protobuf** | KNN, K-Means, PCA, Linear Learner, RCF, NTM, LDA, Object2Vec |
| **CSV** | XGBoost (native), Linear Learner, RCF |
| **LibSVM** | XGBoost (sparse data) |
| **JSON Lines** | DeepAR, Object2Vec |
| **RecordIO (JPEG)** | Image Classification, Object Detection, Semantic Segmentation |
| **Augmented Manifest** | Image Classification, Object Detection (with labels inline) |

### XGBoost Objectives — Must Know

| Task | Objective |
|------|----------|
| Binary classification | `binary:logistic` |
| Multi-class classification | `multi:softmax` / `multi:softprob` |
| Regression | `reg:squarederror` |
| Ranking | `rank:pairwise` |

### Security Controls by Layer

```
Data (S3):         SSE-KMS + Block Public Access + VPC Gateway Endpoint
Training:          VPC subnets + network isolation + KMS volume encryption
Multi-node:        encrypt_inter_container_traffic = True
Endpoint:          VPC + HTTPS (TLS) + WAF (if via API Gateway)
Notebooks:         DirectInternetAccess=Disabled + RootAccess=Disabled
API Access:        IAM least-privilege + PrivateLink endpoints
Audit:             CloudTrail + AWS Config + Model Registry (approval workflow)
```

---

## Last-Minute Key Facts

### Numbers to Memorize

| Fact | Value |
|------|-------|
| SageMaker passing score | 720 / 1000 |
| Real-time endpoint max payload | **6 MB** |
| Serverless endpoint max payload | **4 MB** |
| Async endpoint max payload | **1 GB** |
| Async endpoint max processing time | **15 minutes** |
| Spot training max savings | **~90%** |
| KNN, K-Means, PCA input format | **RecordIO-Protobuf** |
| Feature Store online latency | **< 10ms** |
| DeepAR input format | **JSON Lines** |
| XGBoost native sparse format | **LibSVM** |

### "Always" Rules

1. **Always** use stratified splits for imbalanced datasets
2. **Always** fit scalers/encoders on training data only, then transform test data
3. **Always** use temporal ordering for time-series data splits
4. **Always** use `checkpoint_s3_uri` with Managed Spot Training
5. **Always** enable VPC + S3 Gateway Endpoint for compliance training
6. **Always** set `scale_pos_weight` for imbalanced XGBoost classification
7. **Always** use `event_time` queries in Feature Store to prevent data leakage
8. **Always** use SageMaker Clarify for bias IF you need explainability reports
9. **Always** use SageMaker Model Monitor for production drift detection
10. **Always** separate concerns: Data Wrangler (prep) → Pipelines (automation) → Registry (governance)

### "Use X Not Y" Rules

| Use | Not | Because |
|-----|-----|---------|
| FastFile Mode | Pipe Mode | FastFile supports all formats |
| SageMaker Feature Store | DynamoDB (for ML features) | Feature Store has ML-native time-travel and offline/online duality |
| Async Inference | Lambda (for large payloads) | Lambda has 15-min limit too, but 6 MB request limit |
| Model Monitor | Manual drift checking | Automated, continuous, integrated |
| SageMaker Pipelines | Manual step execution | Reproducible, auditable, parameterized |
| Clarify | Custom SHAP code | Integrated with SageMaker, no infra to manage |
| Managed Spot Training | Reserved Instances (for training) | Spot is pay-per-use, reserved is for always-on resources |
| Serverless Inference | Turning endpoint off/on | Serverless auto-scales to zero automatically |

### SageMaker Limits (Common in Exam)

- Max `max_jobs` in HPO: Check service quotas (default 500)
- Max parallel jobs in HPO: Limited by account quotas (default 10)
- Feature Store Online Store: Latest value per `record_identifier` only
- Processing Job: Runs as a separate Training Job-like container
- Pipelines: Steps connected via S3 URIs — no in-memory data passing
- Batch Transform: No persistent endpoint — job terminates after completion

---

## Final Review Checklist

Before the exam, ensure you can answer these questions without notes:

### Domain 1 — Data
- [ ] What are the 3 S3 input modes for SageMaker training?
- [ ] What is the difference between Feature Store Online and Offline stores?
- [ ] When would you use Glue DataBrew vs Data Wrangler?
- [ ] How do you prevent data leakage when using Feature Store?
- [ ] What data format do most SageMaker built-in algorithms prefer?

### Domain 2 — Development
- [ ] Which algorithm detects anomalies in unlabeled data?
- [ ] What is the best HPO strategy for most use cases?
- [ ] How do you handle class imbalance in XGBoost?
- [ ] What does Managed Spot Training require to be safe?
- [ ] What does SageMaker Debugger detect?

### Domain 3 — Deployment
- [ ] What are the 4 inference modes and their max payload sizes?
- [ ] Which endpoint type can scale to 0 instances?
- [ ] What is the difference between MME and MCE?
- [ ] How does blue/green deployment work for SageMaker endpoints?
- [ ] What triggers automatic deployment after Model Registry approval?

### Domain 4 — Monitoring & Security
- [ ] What are the 4 types of Model Monitor?
- [ ] What is SHAP and which SageMaker tool uses it?
- [ ] What configurations prevent training from accessing the internet?
- [ ] How do you detect when a model is treating demographic groups differently?
- [ ] What is the drift detection → automated retraining pipeline flow?
