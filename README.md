# AWS Certified Machine Learning Engineer – Associate (MLA-C01)
## Complete Study Guide

> **Exam Code:** MLA-C01  
> **Level:** Associate  
> **Duration:** 170 minutes  
> **Questions:** 65 questions (scored) + 15 unscored  
> **Passing Score:** 720 / 1000  
> **Cost:** $150 USD  

---

## Exam Domain Breakdown

| Domain | Topic | Weight |
|--------|-------|--------|
| 1 | Data Preparation for Machine Learning | **28%** |
| 2 | ML Model Development | **26%** |
| 3 | Deployment and Orchestration of ML Workflows | **22%** |
| 4 | ML Solution Monitoring, Maintenance, and Security | **24%** |

---

## Repository Structure

```
ml/
├── guide/    ← Study notes per domain + cheatsheet + exam tips
├── mock/     ← Mock exam questions (20 Q per domain, with answers)
└── lab/      ← Hands-on lab simulations (6 labs per domain, with code)
```

---

## 📖 Study Guides — `guide/`

| File | Domain | Key Services |
|------|--------|-------------|
| [01 - Data Preparation](./guide/01-data-preparation.md) | Domain 1 (28%) | S3, Glue, Athena, Kinesis, SageMaker Processing, Feature Store |
| [02 - Model Development](./guide/02-model-development.md) | Domain 2 (26%) | SageMaker Training, Built-in Algorithms, HPO, Experiments |
| [03 - Deployment & Orchestration](./guide/03-deployment-orchestration.md) | Domain 3 (22%) | SageMaker Endpoints, Pipelines, Step Functions, Model Registry |
| [04 - Monitoring, Maintenance & Security](./guide/04-monitoring-security.md) | Domain 4 (24%) | SageMaker Model Monitor, CloudWatch, IAM, VPC, Clarify |
| [05 - AWS Services Cheatsheet](./guide/05-aws-services-cheatsheet.md) | All Domains | Quick reference for all relevant AWS services |
| [06 - Exam Tips & Practice Questions](./guide/06-exam-tips-practice.md) | All Domains | Strategy, sample questions, and key facts |

---

## 📝 Mock Exams — `mock/`

20 scenario-based questions per domain, each with detailed explanations.

| File | Domain | Coverage |
|------|--------|---------|
| [Domain 1 Mock](./mock/mock-domain1-data-preparation.md) | Data Preparation | FastFile, Feature Store, Glue, Athena, Kinesis, encodings |
| [Domain 2 Mock](./mock/mock-domain2-model-development.md) | Model Development | XGBoost, HPO, Debugger, JumpStart, distributed training |
| [Domain 3 Mock](./mock/mock-domain3-deployment-orchestration.md) | Deployment & Orchestration | Inference types, Pipelines, blue/green, MME, EventBridge |
| [Domain 4 Mock](./mock/mock-domain4-monitoring-security.md) | Monitoring & Security | Model Monitor, Clarify, IAM, VPC, KMS, Model Cards |

---

## 🔬 Lab Practice — `lab/`

6 hands-on labs per domain with working Python/CLI code, step-by-step tasks, and verification checklists.

| File | Domain | Labs Included |
|------|--------|--------------|
| [Domain 1 Labs](./lab/lab-domain1-data-preparation.md) | Data Preparation | S3 input modes, Feature Store, Processing Jobs, Glue ETL, Kinesis streaming, Athena queries |
| [Domain 2 Labs](./lab/lab-domain2-model-development.md) | Model Development | XGBoost training, HPO Bayesian + Warm Start, Experiments, Debugger, JumpStart BERT, distributed DDP |
| [Domain 3 Labs](./lab/lab-domain3-deployment-orchestration.md) | Deployment & Orchestration | Real-time + auto scaling, Async + Serverless, MME, SageMaker Pipelines, blue/green rollback, EventBridge automation |
| [Domain 4 Labs](./lab/lab-domain4-monitoring-security.md) | Monitoring & Security | Data Quality Monitor, Clarify bias + SHAP, Model Quality Monitor, IAM + VPC + KMS, Bias Drift, CloudTrail + Model Cards |

---

## Core AWS Services for MLA-C01

### Data Layer
- **Amazon S3** — Primary data lake storage for ML datasets
- **AWS Glue** — ETL, Data Catalog, and crawlers
- **Amazon Athena** — Serverless SQL queries on S3
- **Amazon Kinesis** — Real-time data streaming (Data Streams, Firehose, Analytics)
- **AWS Lake Formation** — Data lake governance

### ML Platform
- **Amazon SageMaker** — End-to-end ML platform (the core service for this exam)
  - Studio, Training, Processing, Tuning, Pipelines, Feature Store, Model Registry, Endpoints, Model Monitor, Clarify, Debugger, Experiments

### Compute & Containers
- **Amazon EC2** — Custom compute for ML training
- **Amazon ECR** — Container registry for custom training/inference images
- **AWS Batch** — Batch ML workloads

### Orchestration
- **AWS Step Functions** — ML workflow orchestration (non-SageMaker)
- **Amazon MWAA** (Managed Airflow) — DAG-based ML orchestration
- **AWS Lambda** — Event-driven inference triggers

### Security & Governance
- **AWS IAM** — Roles, policies, and permissions
- **AWS KMS** — Encryption key management
- **Amazon VPC** — Network isolation for ML workloads
- **AWS CloudTrail** — API audit logging
- **AWS Config** — Configuration compliance

### Monitoring & Observability
- **Amazon CloudWatch** — Metrics, logs, dashboards, alarms
- **AWS X-Ray** — Distributed tracing for inference

---

## Quick Cheat: SageMaker Component Map

```
INPUT DATA
   │
   ├─► S3 Bucket (raw data)
   │      └─► AWS Glue / SageMaker Processing  →  Feature Engineering
   │                                                     │
   │                                           SageMaker Feature Store
   │                                            (Online + Offline Store)
   ▼
TRAINING
   ├─► SageMaker Training Job (Built-in / Custom / HuggingFace / etc.)
   ├─► SageMaker Hyperparameter Tuner (HPO)
   ├─► SageMaker Experiments (tracking)
   └─► SageMaker Debugger (training insights)
   │
   ▼
MODEL REGISTRY
   └─► SageMaker Model Registry (versioning, approval, lineage)
   │
   ▼
DEPLOYMENT
   ├─► Real-Time Inference Endpoint
   ├─► Serverless Inference
   ├─► Batch Transform
   ├─► Asynchronous Inference
   └─► Multi-Model / Multi-Container Endpoints
   │
   ▼
MONITORING
   ├─► SageMaker Model Monitor (data drift, model quality, bias drift)
   ├─► SageMaker Clarify (bias & explainability)
   └─► CloudWatch (metrics, alarms, dashboards)
   │
   ▼
ORCHESTRATION (MLOps)
   └─► SageMaker Pipelines (end-to-end CI/CD for ML)
```

---

## Recommended Study Order

1. **Read guide/** – Study each domain guide in order (01 → 02 → 03 → 04)
2. **Do lab/** – Run hands-on labs for each domain to reinforce concepts
3. **Review [Cheatsheet](./guide/05-aws-services-cheatsheet.md)** – Services at a glance
4. **Practice with [Exam Tips](./guide/06-exam-tips-practice.md)** – Key facts and traps
5. **Attempt mock/** – Take each domain mock exam under timed conditions
6. **Review wrong answers** – Use the guide to revisit any missed topics

---

## Key Themes to Master

- **SageMaker is the central service** — know every sub-feature
- **MLOps with SageMaker Pipelines** — CI/CD for ML is a major topic
- **Know ALL inference types** — real-time, serverless, batch, async
- **Data drift vs. model drift** — Model Monitor concepts
- **Security layers** — IAM, VPC, KMS, PrivateLink
- **Cost optimization** — Spot instances, managed spot training, Savings Plans
- **Responsible AI** — Clarify for bias detection and explainability
