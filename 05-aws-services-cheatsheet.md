# AWS Services Cheatsheet for MLA-C01
## Quick Reference — All ML-Relevant AWS Services

---

## SageMaker Service Map

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AMAZON SAGEMAKER                                 │
│                                                                         │
│  STUDIO                    DATA                    TRAINING             │
│  ┌─────────────┐          ┌──────────────────┐    ┌─────────────────┐  │
│  │ Notebooks   │          │ Data Wrangler    │    │ Training Jobs   │  │
│  │ Studio IDE  │          │ Feature Store    │    │ HPO (Tuner)     │  │
│  │ Canvas      │          │ Processing Jobs  │    │ Experiments     │  │
│  │ RStudio     │          │ Ground Truth     │    │ Debugger        │  │
│  └─────────────┘          └──────────────────┘    │ JumpStart       │  │
│                                                    └─────────────────┘  │
│  DEPLOYMENT                MONITORING              MLOps                │
│  ┌─────────────────┐      ┌──────────────────┐    ┌─────────────────┐  │
│  │ Real-Time Endpt │      │ Model Monitor    │    │ Pipelines       │  │
│  │ Serverless Inf. │      │ Clarify          │    │ Model Registry  │  │
│  │ Async Inference │      │ Profiler         │    │ Projects        │  │
│  │ Batch Transform │      │ Inference Rec.   │    │ ML Lineage      │  │
│  │ Neo (compile)   │      └──────────────────┘    │ Model Cards     │  │
│  │ MME / MCE       │                              └─────────────────┘  │
│  └─────────────────┘                                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## SageMaker Sub-Services Quick Reference

| Service | One-Line Description | Key Use Case |
|---------|---------------------|-------------|
| **SageMaker Studio** | Web-based IDE for ML | Central workspace for all ML activities |
| **SageMaker Canvas** | No-code AutoML UI | Business users building ML models |
| **SageMaker Autopilot** | Automated ML (AutoML) | Auto-generate pipelines and models |
| **SageMaker Data Wrangler** | Visual data prep in Studio | Clean and transform data for training |
| **SageMaker Processing** | Managed batch compute for preprocessing | Run scripts on managed infra |
| **SageMaker Ground Truth** | Labeling service | Create labeled training datasets |
| **SageMaker Feature Store** | Managed feature repository | Share features across teams/use cases |
| **SageMaker Training Jobs** | Managed training infra | Run training scripts on any instance |
| **SageMaker Experiments** | Track and compare runs | ML experiment management |
| **SageMaker Debugger** | Training insights and profiling | Find training issues and bottlenecks |
| **SageMaker HPO (Tuner)** | Hyperparameter optimization | Find best hyperparameters automatically |
| **SageMaker JumpStart** | Pre-trained models and solutions | Fine-tune foundation models |
| **SageMaker Neo** | Model compilation | Optimize model for target hardware |
| **SageMaker Endpoints** | Real-time inference | Deploy models as REST APIs |
| **SageMaker Serverless Inference** | Auto-scaled serverless endpoints | Low/intermittent traffic |
| **SageMaker Async Inference** | Async large-payload inference | Long-running, large payloads |
| **SageMaker Batch Transform** | Offline batch scoring | Score entire dataset at once |
| **SageMaker Multi-Model Endpoint** | Multiple models on one endpoint | Many models, cost efficiency |
| **SageMaker Inference Recommender** | Benchmark & recommend instance types | Right-size inference infra |
| **SageMaker Pipelines** | ML CI/CD workflow orchestration | End-to-end ML automation |
| **SageMaker Model Registry** | Model versioning & approval | Govern model promotion to production |
| **SageMaker Model Monitor** | Production data/model drift detection | Catch degradation automatically |
| **SageMaker Clarify** | Bias detection + SHAP explainability | Responsible AI compliance |
| **SageMaker ML Lineage** | Track data-model-endpoint relationships | Audit and reproducibility |
| **SageMaker Model Cards** | Model governance documentation | Compliance, documentation |
| **SageMaker Role Manager** | IAM role templates for ML personas | Simplified IAM for ML teams |

---

## Data Services

| Service | Category | ML Use Case |
|---------|----------|-------------|
| **Amazon S3** | Object Storage | Primary data lake for ML datasets and model artifacts |
| **AWS Glue** | ETL | Serverless ETL, Data Catalog, schema management |
| **AWS Glue DataBrew** | Visual ETL | No-code data preparation (standalone, not in Studio) |
| **Amazon Athena** | Serverless SQL | Query S3 data for EDA and feature validation |
| **Amazon Redshift** | Data Warehouse | Structured analytical data for feature engineering |
| **Amazon Redshift ML** | In-database ML | CREATE MODEL SQL syntax, runs SageMaker Autopilot |
| **AWS Lake Formation** | Data Governance | Column/row-level permissions on data lake |
| **Amazon EMR** | Spark/Hadoop | Large-scale distributed data processing |
| **Amazon RDS / Aurora** | Relational DB | Transactional source data for ML features |
| **Amazon DynamoDB** | NoSQL | Feature Store online store backing; low-latency feature lookup |
| **Amazon OpenSearch** | Search/Analytics | Vector search for ML embeddings, log analysis |

---

## Streaming & Real-Time Data

| Service | Description | ML Use Case |
|---------|-------------|-------------|
| **Amazon Kinesis Data Streams** | Real-time data stream | Stream raw data for online learning / real-time feature computation |
| **Amazon Kinesis Data Firehose** | Managed delivery to S3/Redshift | Buffer streaming data → S3 for batch training |
| **Amazon Kinesis Data Analytics** | SQL/Flink on streams | Real-time feature aggregation and anomaly detection |
| **Amazon Kinesis Video Streams** | Video ingestion | Computer vision model input streams |
| **Amazon MSK (Kafka)** | Managed Kafka | High-throughput event streaming to ML pipelines |
| **AWS IoT Greengrass** | Edge ML | Run SageMaker Neo models on IoT devices |
| **Amazon EventBridge** | Event routing | Trigger ML pipelines on data arrival, model approval events |

---

## Compute & Containers

| Service | Description | ML Use Case |
|---------|-------------|-------------|
| **Amazon EC2** | Virtual Machines | Custom training/inference outside SageMaker |
| **Amazon EC2 (p3/p4d/p5)** | GPU instances | Deep learning training (V100/A100/H100) |
| **Amazon EC2 (inf1/inf2)** | Inferentia instances | Low-cost, high-throughput inference |
| **Amazon EC2 (trn1)** | Trainium | Cost-optimized deep learning training |
| **Amazon EC2 (g4dn/g5)** | NVIDIA GPU inference | Cost-effective GPU inference |
| **Amazon ECS** | Container orchestration | Run custom ML containers at scale |
| **Amazon EKS** | Kubernetes | Kubeflow, Karpenter for ML workloads |
| **Amazon ECR** | Container registry | Store custom training/inference images |
| **AWS Batch** | Batch compute | Schedule and run large ML batch jobs |
| **AWS Lambda** | Serverless functions | Event triggers, lightweight inference, pipeline glue |

---

## ML APIs (Pre-Built AI Services)

| Service | Capability | Use Case |
|---------|-----------|---------|
| **Amazon Rekognition** | Computer Vision | Object detection, face recognition, content moderation |
| **Amazon Textract** | Document Analysis | Extract text/forms/tables from PDFs and images |
| **Amazon Comprehend** | NLP | Sentiment, entity, key phrase, topic extraction |
| **Amazon Translate** | Machine Translation | Multilingual content |
| **Amazon Polly** | Text-to-Speech | Audio generation from text |
| **Amazon Transcribe** | Speech-to-Text | Convert audio to text |
| **Amazon Lex** | Conversational AI | Chatbots (powers Alexa) |
| **Amazon Forecast** | Time-Series Forecasting | Business forecasting (no ML code needed) |
| **Amazon Personalize** | Recommendation Engine | Personalized recommendations (Netflix-style) |
| **Amazon Fraud Detector** | Fraud Detection | Online payment fraud, fake account detection |
| **Amazon Kendra** | Enterprise Search | Intelligent document search with NLP |
| **Amazon Bedrock** | Generative AI APIs | LLM APIs (Claude, Titan, Llama, etc.) |
| **Amazon Q** | Generative AI Assistant | Enterprise AI assistant |

---

## Orchestration & Automation

| Service | Description | ML Use Case |
|---------|-------------|-------------|
| **SageMaker Pipelines** | Native ML workflow DAG | End-to-end ML CI/CD |
| **AWS Step Functions** | Visual workflow engine | Orchestrate Glue + SageMaker + Lambda workflows |
| **Amazon MWAA** (Managed Airflow) | Managed Apache Airflow | Complex multi-system ML DAGs |
| **AWS CodePipeline** | CI/CD pipeline | Trigger ML pipelines from code commits |
| **AWS CodeBuild** | Build/test runner | Run tests before model training |
| **AWS CodeCommit** | Git repository | Store ML code and configs |
| **Amazon EventBridge** | Event bus | Route events between ML pipeline components |
| **AWS Lambda** | Serverless compute | Custom logic in ML pipelines |

---

## Security Services

| Service | Description | ML Application |
|---------|-------------|---------------|
| **AWS IAM** | Identity & Access Management | Roles for SageMaker, execution policies |
| **AWS KMS** | Key Management Service | Encrypt S3, EBS, SageMaker volumes |
| **AWS Secrets Manager** | Secret storage | Store API keys, DB passwords in training scripts |
| **AWS Systems Manager (SSM)** | Parameter Store | Configuration params for ML pipelines |
| **Amazon VPC** | Virtual Private Cloud | Isolate training/inference infra from internet |
| **AWS PrivateLink** | VPC Interface Endpoints | Private connectivity to AWS services |
| **AWS WAF** | Web Application Firewall | Protect ML inference REST API endpoints |
| **Amazon API Gateway** | API Management | Frontend SageMaker endpoints with auth, throttling |
| **AWS CloudTrail** | API Audit Logging | Audit all SageMaker API calls |
| **AWS Config** | Resource Compliance | Track SageMaker resource configurations |
| **AWS Audit Manager** | Compliance Reporting | Generate SOC2/ISO evidence for ML systems |
| **AWS Security Hub** | Security Posture | Centralized security findings |
| **Amazon GuardDuty** | Threat Detection | Detect unusual access to ML data/models |
| **Amazon Macie** | Data Classification | Discover PII in ML training datasets |

---

## Monitoring & Observability

| Service | Description | ML Use Case |
|---------|-------------|-------------|
| **Amazon CloudWatch** | Metrics, logs, alarms | Monitor endpoint latency, error rate, resource utilization |
| **CloudWatch Logs Insights** | Log query language | Search training/inference logs |
| **CloudWatch Container Insights** | Container metrics | Monitor ECS/EKS ML workloads |
| **AWS X-Ray** | Distributed tracing | Trace inference requests through API Gateway → SageMaker |
| **SageMaker Model Monitor** | ML-specific drift detection | Data quality, model quality, bias drift |
| **SageMaker Clarify** | Explainability + fairness | SHAP values, bias metrics |
| **SageMaker Debugger** | Training insights | Vanishing gradients, overfitting, system bottlenecks |

---

## Instance Type Reference

### Training Instances

| Family | vCPU | Memory | GPU | Use Case |
|--------|------|--------|-----|---------|
| `ml.m5.xlarge` | 4 | 16 GB | None | Small tabular ML |
| `ml.m5.4xlarge` | 16 | 64 GB | None | Medium tabular ML |
| `ml.c5.4xlarge` | 16 | 32 GB | None | CPU-intensive training |
| `ml.r5.4xlarge` | 16 | 128 GB | None | Memory-intensive |
| `ml.p3.2xlarge` | 8 | 61 GB | 1x V100 | Deep learning (single GPU) |
| `ml.p3.16xlarge` | 64 | 488 GB | 8x V100 | Deep learning (multi GPU) |
| `ml.p3dn.24xlarge` | 96 | 768 GB | 8x V100 + NVLink | Distributed training |
| `ml.p4d.24xlarge` | 96 | 1152 GB | 8x A100 | Large model training |
| `ml.p5.48xlarge` | 192 | 2048 GB | 8x H100 | Foundation model training |
| `ml.trn1.2xlarge` | 8 | 32 GB | 1x Trainium | Cost-efficient DL training |
| `ml.trn1.32xlarge` | 128 | 512 GB | 16x Trainium | Large-scale training |

### Inference Instances

| Family | vCPU | Memory | GPU | Use Case |
|--------|------|--------|-----|---------|
| `ml.c5.xlarge` | 4 | 8 GB | None | CPU inference (recommended) |
| `ml.c5.4xlarge` | 16 | 32 GB | None | High-throughput CPU inference |
| `ml.g4dn.xlarge` | 4 | 16 GB | 1x T4 | Cost-effective GPU inference |
| `ml.g5.xlarge` | 4 | 16 GB | 1x A10G | GPU inference (newer) |
| `ml.inf1.xlarge` | 4 | 8 GB | 1x Inferentia | Low-cost, high-performance |
| `ml.inf2.xlarge` | 4 | 16 GB | 1x Inferentia2 | Latest Inferentia, better perf |
| `ml.c7g.xlarge` | 4 | 8 GB | None | ARM Graviton, 30% cheaper |

---

## Decision Trees

### Which Data Service?

```
Need to run SQL queries on S3 data without loading it?
  → Amazon Athena

Need to build ETL pipelines with scheduling and Data Catalog?
  → AWS Glue

Need visual no-code data prep for data scientists?
  → SageMaker Data Wrangler (inside Studio)

Need visual no-code data prep as a standalone tool?
  → AWS Glue DataBrew

Need to process TBs of data with Spark?
  → AWS Glue (PySpark jobs) or Amazon EMR

Need a centralized metadata registry for all your data?
  → AWS Glue Data Catalog (also used by Athena, EMR, Redshift Spectrum)

Need columnar data warehouse for ML features?
  → Amazon Redshift

Need to run ML directly in your data warehouse with SQL?
  → Amazon Redshift ML
```

### Which Orchestration?

```
All steps are SageMaker steps (Processing, Training, Deployment)?
  → SageMaker Pipelines

Mix of SageMaker + other AWS services (Glue, Lambda, Athena)?
  → AWS Step Functions

Complex DAGs, Airflow operators, custom operators needed?
  → Amazon MWAA (Managed Airflow)

Simple trigger: new S3 file → run pipeline?
  → EventBridge Rule → Lambda → start SageMaker Pipeline
```

### Which Pre-Built AI Service?

```
Classify or detect objects in images?    → Amazon Rekognition
Extract text from scanned documents?     → Amazon Textract
Analyze sentiment in customer reviews?   → Amazon Comprehend
Detect fraud in transactions?            → Amazon Fraud Detector
Recommend products to users?             → Amazon Personalize
Forecast future demand/sales?            → Amazon Forecast
Build a chatbot / voice assistant?       → Amazon Lex
Transcribe customer call recordings?     → Amazon Transcribe
Search enterprise documents with NLP?    → Amazon Kendra
Need a powerful LLM (Claude, Llama)?    → Amazon Bedrock
```

---

## Key Comparisons (Commonly Tested)

### Kinesis vs MSK (Kafka)

| | Kinesis | MSK |
|-|---------|-----|
| Managed | Fully | Partially (brokers managed) |
| Protocol | Proprietary | Kafka (open-source) |
| Scaling | On-demand resharding | Add brokers |
| Retention | Up to 365 days | Configurable |
| ML use | Simpler setup | Kafka ecosystem compatibility |

### Glue vs EMR

| | AWS Glue | Amazon EMR |
|-|----------|-----------|
| Type | Serverless | Managed cluster |
| Startup time | Minutes | Minutes |
| Cost model | Per second (job only) | Cluster running cost |
| Flexibility | PySpark, Python Shell, Ray | Any Spark/Hadoop ecosystem |
| ML fit | ETL pipelines, simpler jobs | Heavy data engineering |

### SageMaker Canvas vs Autopilot

| | Canvas | Autopilot |
|-|--------|-----------|
| Users | Business users (no code) | Data scientists (SDK/Studio) |
| Interface | UI-only | SDK + UI |
| Transparency | Limited | Full pipeline visibility |
| Models | Regression, classification | Regression, classification |
| NLP | No | Time-series via Autopilot |

### Data Wrangler vs Glue DataBrew

| | Data Wrangler | Glue DataBrew |
|-|---------------|---------------|
| Location | Inside SageMaker Studio | Standalone service |
| Output | Feature Store, Pipelines, S3 | S3, Glue Catalog |
| Bias detection | Yes (Clarify) | No |
| Scheduling | Via SageMaker Pipelines | Built-in Jobs scheduler |
| Users | ML engineers/scientists | Data engineers |
