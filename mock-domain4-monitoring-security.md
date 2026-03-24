# Mock Exam — Domain 4: Monitoring, Security, and Deployment of ML Solutions
## MLA-C01 | 20 Questions | ~52 minutes

> **Instructions:** Choose the single best answer unless the question says "Choose TWO."
> Answers and explanations are at the bottom — don't peek!

---

### Question 1
A company deploys a binary classification model to a SageMaker endpoint. Two months after deployment, the model's accuracy has silently degraded because the real-world class distribution has shifted. Which SageMaker Model Monitor type detects this degradation?

- A) Data Quality Monitor — detects feature distribution drift
- B) Model Quality Monitor — detects prediction accuracy drift against ground truth
- C) Bias Drift Monitor — detects changes in bias metrics over time
- D) Feature Attribution Drift Monitor — detects SHAP value changes

---

### Question 2
A Model Monitor baseline was computed on training data. A monitoring schedule runs hourly on the deployed endpoint. The violation report flags the feature `age` with a `dist_param_check` violation. What does this indicate?

- A) The `age` column contains null values in the recent window
- B) The statistical distribution of `age` in live traffic differs significantly from the baseline
- C) The `age` feature has been removed from the inference payload
- D) The model's prediction for `age`-related records exceeds a threshold

---

### Question 3
A data science team wants to check for **pre-training bias** in their dataset before model training. They are concerned that a protected attribute (gender) correlates with the label (loan approval). Which SageMaker service should they use, and which metric quantifies the imbalance in label outcomes between groups?

- A) SageMaker Clarify — Class Imbalance (CI) metric
- B) SageMaker Clarify — Difference in Positive Proportions in Labels (DPL)
- C) SageMaker Model Monitor — Bias Drift Monitor
- D) SageMaker Data Wrangler — SHAP feature importance report

---

### Question 4
A trained model is deployed and SageMaker Clarify runs a post-training bias analysis. The `Disparate Impact (DI)` metric returns a value of 0.62. What does this mean?

- A) The model's accuracy is 62% higher for the privileged group
- B) The ratio of positive prediction rates between unprivileged and privileged groups is 0.62 — below the 0.8 fairness threshold
- C) 62% of predictions for the unprivileged group are incorrect
- D) The privileged group receives positive predictions 0.62 times less often

---

### Question 5
A company wants to understand WHY its SageMaker XGBoost model gives a specific prediction for each individual record — not just global feature importance. Which SageMaker Clarify capability provides local, per-prediction explanations?

- A) Global feature importance report across training data
- B) SHAP (SHapley Additive exPlanations) values per record via `SHAPConfig`
- C) Partial Dependence Plots (PDP)
- D) Confusion matrix breakdown per feature

---

### Question 6
An ML engineer needs to ensure that the SageMaker training job can ONLY communicate within a VPC and cannot access the public internet. Which two configurations enforce this? (Choose TWO)

- A) Set `enable_network_isolation=True` on the Estimator
- B) Configure `subnets` and `security_group_ids` with a private subnet that has no internet gateway route
- C) Set `encrypt_inter_container_traffic=True`
- D) Use a NAT Gateway to route training traffic
- E) Set `volume_kms_key` for the training job

---

### Question 7
A company trains distributed multi-node SageMaker training jobs. Security policy requires that network communication between training nodes must be encrypted in transit. Which parameter enforces this for multi-node training?

- A) `enable_network_isolation=True`
- B) `encrypt_inter_container_traffic=True`
- C) Set S3 encryption with KMS on model artifacts
- D) Use a VPC endpoint for S3 access

---

### Question 8
A SageMaker notebook instance is configured with `RootAccess=Enabled`. A security audit flags this as a violation. What is the security risk, and what should be done?

- A) Users can install packages — change to `RootAccess=Disabled` and use lifecycle configurations for package installation
- B) Root access enables network isolation — disable it to allow internet access
- C) Root access allows the notebook to assume any IAM role — use SCPs to restrict
- D) There is no security risk; root access is required for SageMaker notebooks

---

### Question 9
A company stores model artifacts in S3 and SageMaker Feature Store data in S3. Security policy requires that all data at rest must be encrypted with customer-managed keys (CMK) and all key usage must be auditable. Which combination meets these requirements? (Choose TWO)

- A) S3 default SSE-S3 encryption
- B) S3 SSE-KMS with a customer-managed KMS key
- C) SageMaker Feature Store `OfflineStoreConfig` with `KmsKeyId` set to the CMK ARN
- D) SageMaker notebook EBS encryption with a managed key
- E) CloudTrail logging of S3 API calls only

---

### Question 10
A model deployed on a SageMaker endpoint calls an external third-party API during inference. A security review finds this is a compliance violation — the model must not make calls to external internet addresses. How should this be enforced?

- A) Block port 443 in the endpoint's security group outbound rules
- B) Set `enable_network_isolation=True` on the Model object
- C) Use a VPC endpoint to restrict outbound traffic
- D) Configure an IAM policy denying internet access for the SageMaker execution role

---

### Question 11
A SageMaker training job runs in a VPC private subnet. The training script must read data from S3 but the subnet has no NAT Gateway or internet gateway. S3 access is consistently timing out. What is the MOST cost-effective fix?

- A) Add a NAT Gateway to the private subnet
- B) Temporarily enable `DirectInternetAccess` on the training instance
- C) Create an S3 Gateway VPC Endpoint and add it to the route table
- D) Move training data to EFS and mount it to the training job

---

### Question 12
An ML engineer needs to configure SageMaker Model Monitor to capture inference request/response data. Data Capture is enabled with `sampling_percentage=20`. What is the effect?

- A) 20% of endpoint instances are monitored
- B) 20% of inference requests and responses are captured and stored in S3
- C) Model Monitor runs every 20 minutes
- D) 20% of features are included in each captured record

---

### Question 13
A SageMaker Model Quality Monitor requires ground truth labels to evaluate prediction accuracy. The production application receives user outcomes (labels) hours after the initial prediction. How should the team deliver these delayed ground truth labels to the Model Quality Monitor?

- A) Retrain the model with corrected labels and redeploy
- B) Merge ground truth labels into the captured data using the inference ID, stored in the ground truth S3 path configured in the monitor schedule
- C) Add a separate Data Quality Monitor to track label accuracy
- D) Ground truth must be provided within 60 seconds — otherwise the record is excluded

---

### Question 14
A company detects bias drift in a production model via the SageMaker Bias Drift Monitor. The team wants this detection to automatically trigger a model retraining pipeline. Which architecture implements this event-driven retraining?

- A) Model Monitor sends email → team manually triggers pipeline via Studio
- B) Model Monitor CloudWatch alarm → EventBridge rule → Lambda → SageMaker Pipelines `start_pipeline_execution`
- C) Model Monitor webhook → SNS → SQS → SageMaker Training Job
- D) CloudTrail event → CloudWatch Logs → Lambda → Retraining

---

### Question 15
A company must prove regulatory compliance for their credit scoring model — they must document who built the model, what data was used, what bias was measured, and its intended use case. Which SageMaker feature is BEST suited to formalize this documentation?

- A) SageMaker Experiments — logs all training runs
- B) SageMaker Model Registry — stores model version metadata
- C) SageMaker Model Cards — structured documentation of model details, intended use, bias, and evaluation results
- D) SageMaker ML Lineage Tracking — upstream artifact graph

---

### Question 16
A security team reviews IAM permissions for a SageMaker execution role. The role has `sagemaker:*` and `s3:*` on `*`. The team wants to enforce least privilege. Which approach BEST follows the principle of least privilege for SageMaker roles?

- A) Reduce to `sagemaker:*` but keep `s3:*`
- B) Create separate IAM roles per SageMaker workflow (training, processing, endpoint) with only required permissions scoped to specific S3 bucket ARNs
- C) Use an AWS Managed Policy like `AmazonSageMakerFullAccess`
- D) Use IAM permission boundaries that allow `sagemaker:*`

---

### Question 17
An ML team runs SageMaker training jobs and wants to prevent any training job from accessing the public internet while ensuring jobs can still write model artifacts to S3 and pull container images from ECR. The VPC has no internet gateway. Which endpoints are required?

- A) Only an S3 Gateway Endpoint
- B) S3 Gateway Endpoint + ECR Interface VPC Endpoint (`com.amazonaws.region.ecr.api` and `com.amazonaws.region.ecr.dkr`) + S3 Interface Endpoint for large layer pulls
- C) A single NAT Gateway for all outbound traffic
- D) VPC peering to the SageMaker service VPC

---

### Question 18
A SageMaker Feature Attribution Drift Monitor compares SHAP values from the baseline against SHAP values computed on recent inference data. It reports that the feature `transaction_amount` has drifted significantly. What does this indicate?

- A) The `transaction_amount` values in recent requests are outside the training distribution
- B) The model has begun relying more or less on `transaction_amount` compared to baseline, indicating the model's decision logic is changing
- C) The `transaction_amount` column was dropped from recent inference payloads
- D) The model's accuracy for records with high `transaction_amount` has dropped

---

### Question 19
A team wants to audit all SageMaker API calls made in their AWS account — specifically who created training jobs, when endpoints were updated, and who approved model packages in the registry. Which AWS service provides this immutable audit trail?

- A) Amazon CloudWatch Metrics — track SageMaker API call counts
- B) AWS CloudTrail — records all SageMaker management API calls with caller identity, timestamps, and parameters
- C) SageMaker ML Lineage Tracking — provides API call history
- D) VPC Flow Logs — capture all network requests to SageMaker

---

### Question 20
A company is optimizing inference costs for a SageMaker real-time endpoint running on `ml.g4dn.2xlarge`. GPU utilization averages only 12% during business hours. Which combination of changes would MOST reduce cost while maintaining real-time latency?

- A) Switch to Batch Transform and schedule it hourly
- B) Downsize to `ml.g4dn.xlarge` and compile the model with SageMaker Neo for the target hardware
- C) Enable Serverless Inference with the same model artifact
- D) Move to `ml.inf1.xlarge` (Inferentia) and install the Neuron SDK — AWS Inferentia is purpose-built for cost-effective inference

---

## ✅ Answers & Explanations

---

**Q1 — Answer: B (Model Quality Monitor)**
Model Quality Monitor compares live predictions against actual outcomes (ground truth labels) to detect accuracy, precision, recall, or F1 degradation over time. This requires capture of predictions AND later ingestion of ground truth labels. Data Quality Monitor (A) checks feature distributions — it would catch input drift but not prediction accuracy degradation. Bias Drift (C) and Feature Attribution Drift (D) track those specific model properties, not overall accuracy.

---

**Q2 — Answer: B (Statistical distribution of `age` differs from baseline)**
`dist_param_check` violations occur when the statistical parameters (mean, standard deviation, min/max, quantiles) of a feature in live traffic differ from the baseline statistics by more than a configured threshold. This is **data distribution drift** — the same feature, but its distribution has changed. Null values (A) trigger `completeness` violations. Missing columns (C) trigger `type_check` violations.

---

**Q3 — Answer: B (DPL — Difference in Positive Proportions in Labels)**
DPL measures the difference in the proportion of positive labels between the privileged and unprivileged groups in the **raw dataset** (pre-training). A non-zero DPL indicates the label itself is biased against one group — a pre-training bias metric. CI (A) measures imbalance between group sizes (how many samples per group), not label outcome distributions. Bias Drift Monitor (C) is for deployed models, not pre-training dataset analysis.

---

**Q4 — Answer: B (Ratio of positive prediction rates is 0.62 — below 0.8)**
Disparate Impact = (positive prediction rate for unprivileged group) / (positive prediction rate for privileged group). A value of 0.62 means the unprivileged group receives positive predictions 62% as often as the privileged group. The widely-used "80% rule" (four-fifths rule) considers DI < 0.8 to indicate discrimination. DI = 1.0 means equal rates; DI < 1.0 means the unprivileged group is disadvantaged.

---

**Q5 — Answer: B (SHAP values per record via `SHAPConfig`)**
SHAP (SHapley Additive exPlanations) provides local explanations — for each individual prediction, SHAP values show the contribution of each feature to that specific outcome. This differs from global feature importance which averages over all records. The `SHAPConfig` in SageMaker Clarify specifies baseline values and enables per-record SHAP computation. PDPs (C) show global average relationships, not per-record explanations.

---

**Q6 — Answer: A and B (network_isolation + private subnet without internet route)**
`enable_network_isolation=True` (A) prevents the container from making any outbound network calls (but also blocks S3 access via the container — S3 must be pre-downloaded or use input channels). Using private subnets without an internet gateway route (B) is how VPC isolation is enforced at the network level. `encrypt_inter_container_traffic=True` (C) encrypts node-to-node communication for multi-node training but doesn't restrict internet access. S3 access in a VPC-isolated, network-isolated setup requires S3 Gateway Endpoints.

---

**Q7 — Answer: B (`encrypt_inter_container_traffic=True`)**
In distributed multi-node SageMaker training, multiple instances communicate via network (e.g., parameter sharing, gradients). `encrypt_inter_container_traffic=True` encrypts this node-to-node traffic in transit using TLS. `enable_network_isolation=True` (A) restricts outbound internet access but doesn't encrypt inter-node traffic. KMS (C) encrypts data at rest, not in transit. VPC endpoints (D) route traffic privately but don't encrypt it.

---

**Q8 — Answer: A (Root access allows system-level changes; disable it)**
Root access on a notebook instance allows users to install system packages, modify kernel environments, run arbitrary commands as root — a significant security risk in multi-user or compliance-sensitive environments. Best practice is `RootAccess=Disabled`. Packages should be installed via SageMaker lifecycle configurations (run at startup before user access). IAM role assumptions are controlled by IAM policies, not root access.

---

**Q9 — Answer: B and C (S3 SSE-KMS with CMK + Feature Store KmsKeyId)**
S3 SSE-KMS with a customer-managed key (B) encrypts S3-stored model artifacts with a CMK where all key usage events (Encrypt, Decrypt, GenerateDataKey) are logged in AWS CloudTrail — providing the required auditability. Feature Store's offline store in S3 requires separately setting `KmsKeyId` (C) in `OfflineStoreConfig` to use the CMK. SSE-S3 (A) uses AWS-managed keys — not customer-managed, so not auditable via customer key policy. Notebook EBS encryption (D) protects notebook storage, not S3 data.

---

**Q10 — Answer: B (`enable_network_isolation=True`)**
`enable_network_isolation=True` on the Model object prevents the inference container from making ANY outbound network calls, including to external APIs. This is the direct, container-level enforcement. Security group rules (A) work at the VPC level but require the endpoint to be VPC-configured. IAM policies (D) don't restrict network-level TCP/HTTPS calls made from inside the container.

---

**Q11 — Answer: C (S3 Gateway VPC Endpoint)**
An S3 Gateway Endpoint routes S3 traffic through the AWS private network via the route table — no internet gateway or NAT required. It's free of charge (unlike NAT Gateways which cost per GB). This is the standard solution for S3 access from private VPC subnets. NAT Gateway (A) works but costs money per GB. `DirectInternetAccess` (B) breaks the VPC isolation requirement.

---

**Q12 — Answer: B (20% of inference requests are captured)**
Data Capture's `sampling_percentage` controls the fraction of inference requests (and their responses) that are saved to S3 for monitoring. `sampling_percentage=20` means 1 in 5 inference transactions is captured. This does NOT affect instances (A), monitoring frequency (C), or feature selection (D).

---

**Q13 — Answer: B (Merge using inference ID into the ground truth S3 path)**
SageMaker captures each inference request's response with an `InferenceId` (a unique identifier). When ground truth labels arrive later (hours or days later), the team uploads a JSONL file to the configured ground truth S3 location with matching `InferenceId` entries. The Model Quality Monitor joins predictions with ground truth using the inference ID to compute accuracy metrics. There is no strict 60-second window (D) — labels can arrive much later.

---

**Q14 — Answer: B (CloudWatch alarm → EventBridge → Lambda → start_pipeline_execution)**
Model Monitor publishes findings as CloudWatch metrics and emits violations. A CloudWatch alarm monitors the violations metric → triggers an EventBridge rule → invokes a Lambda function → calls `sagemaker.start_pipeline_execution()` on the retraining pipeline. This is the standard event-driven MLOps architecture for closed-loop retraining. Webhooks (C) are not a SageMaker-native integration. CloudTrail (D) records API calls but is not designed for metric-based alerting.

---

**Q15 — Answer: C (SageMaker Model Cards)**
Model Cards are SageMaker's structured documentation format designed for ML governance and compliance. They capture model intended use, training information, evaluation results, bias measurements, and responsible AI considerations. They are shareable, versioned, and export-ready for auditors. Model Registry (B) stores metadata and approval status but lacks the structured documentation format. Experiments (A) and Lineage (D) track technical artifacts, not compliance narratives.

---

**Q16 — Answer: B (Separate roles per workflow, scoped to specific resources)**
Least privilege means each role has only the permissions it needs for its specific task, on specific resources. A training role needs `s3:GetObject` on the training data bucket and `s3:PutObject` on the output bucket — not `s3:*` on `*`. `AmazonSageMakerFullAccess` (C) is an AWS managed policy that grants broad access — convenient but not least privilege. Permission boundaries (D) constrain the maximum permissions but don't automatically apply least privilege.

---

**Q17 — Answer: B (S3 Gateway + ECR API + ECR DKR + S3 Interface endpoints)**
For VPC-isolated training: S3 Gateway Endpoint routes S3 data access. ECR requires two interface endpoints: `ecr.api` (for image manifest calls) and `ecr.dkr` (for Docker layer pulls). Since Docker layers can be large, large layer pulls also route through the S3 service (ECR stores layers in S3), requiring either the S3 Gateway endpoint or S3 interface endpoint. NAT Gateway (C) routes all traffic via internet — violates the requirement.

---

**Q18 — Answer: B (Model's reliance on the feature has changed)**
Feature Attribution Drift tracks changes in SHAP values over time. If `transaction_amount`'s SHAP contribution has changed versus baseline, it means the model is now making decisions based on this feature differently — even if the raw values look similar. This suggests the model's behavior (not just the data) has changed, possibly indicating the need for retraining. Data distribution drift (A) is detected by Data Quality Monitor. Dropped columns (C) trigger data quality violations.

---

**Q19 — Answer: B (AWS CloudTrail)**
CloudTrail records every AWS Management API call — including all SageMaker API calls (`CreateTrainingJob`, `UpdateEndpoint`, `UpdateModelPackage` for approvals). Each entry includes the IAM principal, timestamp, source IP, and request parameters. This is the standard audit trail for compliance. CloudWatch Metrics (A) tracks performance metrics, not API call details. ML Lineage (C) tracks artifact relationships, not API caller identities. VPC Flow Logs (D) capture network-level flows, not application-layer API calls.

---

**Q20 — Answer: D (ml.inf1.xlarge with Neuron SDK — cost-effective inference)**
AWS Inferentia (`ml.inf1` and `ml.inf2`) instances are purpose-built for ML inference with high throughput and low cost per inference. They are substantially cheaper than equivalent GPU instances for inference workloads. With 12% GPU utilization on a g4dn, the model workload is light — Inferentia is ideal. The Neuron SDK is required (compile and load model using `torch_neuron` or `tensorflow_neuron`). Option B (downsize + Neo) is also valid but Inferentia offers better price/performance for inference-only workloads when the model can be compiled for Neuron.
