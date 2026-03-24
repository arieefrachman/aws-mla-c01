# Domain 1: Data Preparation for Machine Learning
## MLA-C01 Study Guide — 28% of Exam

---

## Table of Contents
1. [Data Ingestion & Storage](#1-data-ingestion--storage)
2. [Data Transformation & ETL](#2-data-transformation--etl)
3. [Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
4. [Feature Engineering](#4-feature-engineering)
5. [SageMaker Feature Store](#5-sagemaker-feature-store)
6. [SageMaker Processing Jobs](#6-sagemaker-processing-jobs)
7. [Data Quality & Validation](#7-data-quality--validation)
8. [Streaming Data for ML](#8-streaming-data-for-ml)
9. [Key Facts & Exam Tips](#9-key-facts--exam-tips)

---

## 1. Data Ingestion & Storage

### Amazon S3 — The ML Data Lake

S3 is the **primary storage** for all ML data on AWS. Understanding S3 configurations is essential.

| Concept | Detail |
|---------|--------|
| **Storage Classes** | S3 Standard → S3 IA → S3 Glacier (choose based on access frequency) |
| **S3 Intelligent-Tiering** | Automatically moves data between tiers — ideal for ML datasets with unpredictable access |
| **Versioning** | Tracks dataset versions; critical for reproducibility |
| **Lifecycle Policies** | Automatically archive/expire old training data |
| **Event Notifications** | Trigger Lambda/SQS/SNS on new data arrival → initiate ML pipelines |
| **Transfer Acceleration** | Speeds up uploads from edge locations |
| **Requester Pays** | Consumer pays for data transfer — useful for shared datasets |

**S3 Data Formats for ML:**
```
CSV       → Simple tabular data, widely supported by SageMaker built-in algorithms
LibSVM    → Sparse data (XGBoost native format)
RecordIO  → Efficient binary format (Image Classification, Object Detection)
Parquet   → Columnar format, best for large-scale analytics with Athena/Glue
JSON/JSONL → Semi-structured / NLP datasets
TFRecord  → TensorFlow native streaming format
```

**Best Practice: S3 URI Patterns for SageMaker**
```python
# Training data path convention
s3://my-bucket/project/datasets/train/
s3://my-bucket/project/datasets/validation/
s3://my-bucket/project/datasets/test/

# Model artifacts
s3://my-bucket/project/models/
```

### Amazon S3 Access Modes for SageMaker Training

| Mode | Description | When to Use |
|------|-------------|-------------|
| **File Mode** | Downloads all data to training instance before training | Small-medium datasets |
| **Pipe Mode** | Streams data directly from S3 during training (RecordIO) | Large datasets, memory-constrained |
| **FastFile Mode** | Streams data with file system interface; no data copy | Large datasets, arbitrary formats |

> **Exam Tip:** FastFile Mode is newer and generally preferred over Pipe Mode for large datasets.

---

### AWS Glue — Serverless ETL & Data Catalog

#### Glue Components

| Component | Purpose |
|-----------|---------|
| **Glue Data Catalog** | Centralized metadata repository; integrates with Athena, EMR, Redshift |
| **Glue Crawlers** | Automatically discover schema from S3/RDS/Redshift and populate Data Catalog |
| **Glue ETL Jobs** | PySpark/Scala/Python Shell transforms; can write to S3, Redshift, etc. |
| **Glue DataBrew** | Visual, no-code data preparation; 250+ built-in transformations |
| **Glue Elastic Views** | SQL-based materialized views across data stores |
| **Glue Studio** | Visual drag-and-drop ETL pipeline builder |

#### Glue ETL Job Types

```
Spark (PySpark/Scala) → Best for large-scale batch transformations
Python Shell          → Lighter jobs using pandas/numpy
Streaming             → Continuous ETL from Kinesis/Kafka
Ray                   → Distributed Python for ML-specific workloads
```

#### Glue for ML Pipelines

```python
# Example: Glue job outputs Parquet to S3 → SageMaker reads for training
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read from Data Catalog
datasource = glueContext.create_dynamic_frame.from_catalog(
    database="ml_database",
    table_name="raw_training_data"
)

# Apply mapping / transformation
mapped = ApplyMapping.apply(frame=datasource, mappings=[
    ("customer_id", "long", "customer_id", "long"),
    ("purchase_amount", "double", "purchase_amount", "double"),
    ("label", "long", "label", "long")
])

# Write to S3 as Parquet
glueContext.write_dynamic_frame.from_options(
    frame=mapped,
    connection_type="s3",
    connection_options={"path": "s3://my-bucket/processed/"},
    format="parquet"
)
job.commit()
```

---

### Amazon Athena — Serverless SQL on S3

- Query S3 data directly with standard SQL (no data loading required)
- Integrates with **Glue Data Catalog** for schema management
- Pay per query (per TB scanned) — use **columnar formats (Parquet/ORC)** and **partitioning** to reduce costs

```sql
-- Create external table pointing to S3 partitioned data
CREATE EXTERNAL TABLE transactions (
    customer_id BIGINT,
    amount DOUBLE,
    merchant STRING,
    label INT
)
PARTITIONED BY (year STRING, month STRING)
STORED AS PARQUET
LOCATION 's3://my-bucket/transactions/';

-- Query for ML feature validation
SELECT 
    AVG(amount) AS mean_amount,
    STDDEV(amount) AS std_amount,
    COUNT(*) AS n_samples,
    SUM(CASE WHEN amount IS NULL THEN 1 ELSE 0 END) AS nulls
FROM transactions
WHERE year = '2024' AND month = '01';
```

**Athena + SageMaker Integration:**
- Use `awswrangler` library in SageMaker notebooks to query Athena → pandas DataFrame
- Use SageMaker Processing Jobs to run Athena queries at scale

---

## 2. Data Transformation & ETL

### SageMaker Data Wrangler

SageMaker Data Wrangler is the **visual data preparation tool** within SageMaker Studio.

| Feature | Description |
|---------|-------------|
| **Data Import** | Connect to S3, Athena, Redshift, EMR, SageMaker Feature Store |
| **300+ Transforms** | Handle missing values, encode categoricals, scale features, etc. |
| **Data Quality Report** | Automatically surface statistics and anomalies |
| **Bias Report** | Detect pre-training bias in datasets (integrates with Clarify) |
| **Export** | Export to S3, Feature Store, SageMaker Pipelines, or Jupyter notebook |

**Key Data Wrangler Transforms:**
```
Numeric:      Standardize, Min-Max Normalize, Log Transform, Quantile Transform
Categorical:  One-Hot Encode, Ordinal Encode, Target Encode
Text:         TF-IDF, Count Vectorizer, Embedding
Datetime:     Extract year/month/day/hour, cyclic encoding
Missing:      Impute (mean/median/mode/custom), Drop rows, Forward fill
Outliers:     Winsorize, IQR-based removal, Standard deviation clipping
```

### Common Data Transformation Patterns

#### Handling Missing Values

| Strategy | Formula | When to Use |
|----------|---------|-------------|
| Mean imputation | $\hat{x} = \bar{x}$ | Continuous, normally distributed |
| Median imputation | $\hat{x} = \text{median}(x)$ | Continuous, skewed distribution |
| Mode imputation | Most frequent value | Categorical features |
| KNN imputation | Based on similar records | When missingness has patterns |
| Drop rows | Remove records with nulls | When missing rate < 5% |

#### Feature Scaling

| Method | Formula | Best For |
|--------|---------|---------|
| **Min-Max Normalization** | $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$ | Bounded range [0,1]; Neural Networks |
| **Standardization (Z-score)** | $x' = \frac{x - \mu}{\sigma}$ | Linear models, SVM, PCA |
| **Robust Scaling** | $x' = \frac{x - Q_1}{Q_3 - Q_1}$ | Data with outliers |
| **Log Transform** | $x' = \log(x + 1)$ | Right-skewed distributions |

---

## 3. Exploratory Data Analysis (EDA)

### Statistical Measures to Compute

```python
import pandas as pd
import numpy as np

df = pd.read_csv("s3://bucket/data.csv")

# Central tendency
df.describe()          # count, mean, std, min, quartiles, max

# Distribution
df.skew()              # Skewness (>1 or <-1 is heavily skewed)
df.kurtosis()          # Kurtosis (heavy tails)

# Missing values
df.isnull().sum() / len(df)   # % missing per column

# Correlation
df.corr()             # Pearson correlation matrix
df.corr('spearman')   # Spearman (monotonic, non-linear)
```

### Data Distribution Checks

| Check | What to Look For | Action |
|-------|----------------|--------|
| **Skewness** | |skew| > 1 | Apply log/sqrt transform |
| **Outliers** | Points > 3σ from mean | Winsorize or remove |
| **Class imbalance** | Minority class < 10% | Oversample (SMOTE), undersample, or adjust weights |
| **Multicollinearity** | |correlation| > 0.9 | Remove or merge correlated features |
| **Feature cardinality** | Many unique categorical values | Hash encoding, embedding |

### Class Imbalance Techniques

```
Oversampling:
  - SMOTE (Synthetic Minority Oversampling Technique)
  - Random oversampling

Undersampling:
  - Random undersampling
  - Tomek Links
  - NearMiss

Algorithm-level:
  - Class weights (scale_pos_weight in XGBoost)
  - Focal Loss (for neural networks)
  - Threshold adjustment (change decision boundary)

Evaluation:
  - Use F1-score, AUC-ROC, Precision-Recall curve (NOT raw accuracy)
```

---

## 4. Feature Engineering

### Feature Types and Encoding

#### Categorical Encoding

| Encoding | Best For | Notes |
|----------|---------|-------|
| **One-Hot Encoding** | Low cardinality (< 15 values) | Creates sparse features |
| **Ordinal Encoding** | Ordered categories (Low/Med/High) | Preserves order |
| **Target Encoding** | High cardinality | Replaces category with target mean — risk of leakage |
| **Hash Encoding** | Very high cardinality | Fixed dimension, some collision |
| **Embedding** | Text, large vocabulary | Learned dense representation |

#### Text Feature Extraction

| Method | Description | Use Case |
|--------|-------------|---------|
| **Bag of Words** | Word frequency count matrix | Simple text classification |
| **TF-IDF** | $\text{TF}(t,d) \times \log\frac{N}{df(t)}$ | Relevance-weighted text features |
| **Word2Vec / GloVe** | Pretrained dense word vectors | Semantic similarity tasks |
| **BERT Embeddings** | Contextual sentence embeddings | Complex NLP tasks via SageMaker JumpStart |

#### Time-Series Features

```python
# Cyclical encoding for time features (avoids discontinuity at Dec→Jan)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Lag features
df['sales_lag_1'] = df['sales'].shift(1)
df['sales_lag_7'] = df['sales'].shift(7)

# Rolling window statistics
df['sales_rolling_mean_7d'] = df['sales'].rolling(7).mean()
df['sales_rolling_std_7d'] = df['sales'].rolling(7).std()
```

### Dimensionality Reduction

| Technique | Type | Use Case |
|-----------|------|---------|
| **PCA** (Principal Component Analysis) | Linear | Reduce correlated numeric features |
| **t-SNE** | Non-linear | Visualization of high-dimensional data |
| **UMAP** | Non-linear | Faster than t-SNE, preserves global structure |
| **Feature Selection (Chi-², F-test)** | Statistical filter | Remove irrelevant features |
| **L1 Regularization (Lasso)** | Embedded | Shrinks irrelevant feature weights to 0 |
| **Recursive Feature Elimination (RFE)** | Wrapper | Iteratively removes least important features |

---

## 5. SageMaker Feature Store

Feature Store solves the **training-serving skew** problem by providing a **consistent feature source for both training and real-time inference**.

### Architecture

```
Data Sources (S3, Kinesis, DB)
        │
        ▼
Feature Processor / Ingestion API
        │
        ├─────────────────────────────────────────┐
        ▼                                         ▼
  ONLINE STORE                             OFFLINE STORE
  (Low-latency, DynamoDB-backed)           (S3-backed, Parquet)
  < 10ms point lookups                     Batch training queries
  Latest feature values only               Full history, time-travel
        │                                         │
        ▼                                         ▼
  Real-time Inference                    SageMaker Training Job
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Feature Group** | Container for related features (like a table) |
| **Record Identifier** | Unique ID for each entity (customer_id, product_id) |
| **Event Time** | Timestamp for each feature record — enables point-in-time queries |
| **Online Store** | Low-latency read store for real-time inference |
| **Offline Store** | S3-backed store for training data retrieval |
| **Feature Processor** | `@feature_processor` decorator for scheduled transforms |
| **Cross-account sharing** | Share Feature Groups across AWS accounts |

### Creating a Feature Group

```python
import boto3
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup

sess = sagemaker.Session()
role = sagemaker.get_execution_role()

feature_group = FeatureGroup(
    name="customer-features",
    sagemaker_session=sess
)

feature_group.load_feature_definitions(data_frame=df)  # auto-infer schema

feature_group.create(
    s3_uri=f"s3://my-bucket/feature-store/",
    record_identifier_name="customer_id",
    event_time_feature_name="event_time",
    role_arn=role,
    enable_online_store=True,
    description="Customer features for churn prediction"
)
```

### Ingesting Features

```python
# Batch ingestion from DataFrame
feature_group.ingest(
    data_frame=df,
    max_workers=5,
    wait=True
)

# Real-time single-record ingestion (from Lambda/streaming)
featurestore_runtime = boto3.client('sagemaker-featurestore-runtime')
featurestore_runtime.put_record(
    FeatureGroupName="customer-features",
    Record=[
        {"FeatureName": "customer_id", "ValueAsString": "C123"},
        {"FeatureName": "total_purchases", "ValueAsString": "42"},
        {"FeatureName": "event_time", "ValueAsString": "2024-01-15T10:00:00Z"},
    ]
)
```

### Point-in-Time Queries (Avoiding Data Leakage)

```python
# Retrieve features as of a specific timestamp — prevents leakage
training_dataset = feature_group.as_hive_ddl()

# Use Athena to query offline store with time-travel
query = feature_group.athena_query()
query.run(
    query_string="""
    SELECT customer_id, total_purchases, avg_order_value
    FROM "customer-features"
    WHERE event_time <= '2024-01-01'
    """,
    output_location="s3://my-bucket/query-results/"
)
training_df = query.as_dataframe()
```

---

## 6. SageMaker Processing Jobs

Processing Jobs run **pre/post-processing scripts** on managed infrastructure.

### Processing Job Use Cases

```
Pre-processing:  feature engineering, data cleaning, train/val/test split
Post-processing: model evaluation, metric computation, report generation
Data validation: data quality checks before training
```

### Running a Processing Job (Scikit-Learn)

```python
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

sklearn_processor = SKLearnProcessor(
    framework_version='1.2-1',
    instance_type='ml.m5.xlarge',
    instance_count=1,
    role=role
)

sklearn_processor.run(
    code='preprocessing.py',
    inputs=[
        ProcessingInput(
            source='s3://my-bucket/raw-data/',
            destination='/opt/ml/processing/input'
        )
    ],
    outputs=[
        ProcessingOutput(
            source='/opt/ml/processing/train',
            destination='s3://my-bucket/processed/train/'
        ),
        ProcessingOutput(
            source='/opt/ml/processing/test',
            destination='s3://my-bucket/processed/test/'
        )
    ]
)
```

### Processing Script (`preprocessing.py`)

```python
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    input_dir = '/opt/ml/processing/input'
    
    df = pd.read_csv(os.path.join(input_dir, 'data.csv'))
    
    # Feature engineering
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save outputs
    os.makedirs('/opt/ml/processing/train', exist_ok=True)
    os.makedirs('/opt/ml/processing/test', exist_ok=True)
    
    pd.DataFrame(X_train).join(pd.Series(y_train, name='label')).to_csv(
        '/opt/ml/processing/train/train.csv', index=False
    )
    pd.DataFrame(X_test).join(pd.Series(y_test, name='label')).to_csv(
        '/opt/ml/processing/test/test.csv', index=False
    )
```

### Processing Framework Options

| Processor | Framework | Use Case |
|-----------|-----------|---------|
| `SKLearnProcessor` | Scikit-Learn | General ML preprocessing |
| `PySparkProcessor` | PySpark on EMR | Large-scale distributed transforms |
| `SparkJarProcessor` | Scala Spark | JVM-based Spark transforms |
| `FrameworkProcessor` | Any custom framework | Custom Docker containers |
| `ScriptProcessor` | Custom Docker | Fully custom environment |

---

## 7. Data Quality & Validation

### SageMaker Data Quality Baseline

```python
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
)

# Create baseline statistics from training data
monitor.suggest_baseline(
    baseline_dataset='s3://my-bucket/training-data/train.csv',
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri='s3://my-bucket/baseline/',
)
```

### Data Quality Checks

| Check | Description | Tool |
|-------|-------------|------|
| **Schema validation** | Correct dtypes, column names | Glue, Deequ (open-source) |
| **Completeness** | % non-null per column | Pandas, Deequ |
| **Range validation** | Values within expected bounds | SageMaker Data Wrangler |
| **Statistical drift** | Distribution shift from historical | SageMaker Model Monitor |
| **Referential integrity** | Foreign keys exist in lookup tables | Glue, Athena |
| **Uniqueness** | No duplicate records | Pandas, Deequ |

### AWS Deequ — Data Quality at Scale

Deequ is an open-source library built on Apache Spark for defining **data quality constraints**:

```python
# Typical Deequ constraints (Scala/PySpark on Glue or EMR)
from pydeequ.checks import Check, CheckLevel
from pydeequ.verification import VerificationSuite

check = Check(spark, CheckLevel.Warning, "Data Quality") \
    .hasSize(lambda x: x >= 1000) \
    .isComplete("customer_id") \
    .isUnique("customer_id") \
    .isContainedIn("status", ["active", "inactive", "pending"]) \
    .isNonNegative("purchase_amount") \
    .hasCompleteness("email", lambda x: x >= 0.95)
```

---

## 8. Streaming Data for ML

### Amazon Kinesis Services

| Service | Purpose | ML Use Case |
|---------|---------|-------------|
| **Kinesis Data Streams** | Real-time data ingestion | Stream clickstream/IoT data for online ML |
| **Kinesis Data Firehose** | Managed delivery to S3/Redshift/OpenSearch | Buffer streaming data to S3 for batch training |
| **Kinesis Data Analytics** | SQL/Flink on streaming data | Real-time feature computation |
| **Kinesis Video Streams** | Video ingestion | Computer vision model input |

### Streaming ML Pipeline Pattern

```
IoT Sensors / Application
         │
         ▼
  Kinesis Data Streams
         │
         ├──► Kinesis Analytics (real-time aggregations → online features)
         │              └──► DynamoDB (feature cache for inference)
         │
         └──► Kinesis Firehose
                    └──► S3 (historical data → batch training)
                              └──► SageMaker Training Job
```

### Real-Time Inference with Streaming Triggers

```python
# Lambda function triggered by Kinesis → calls SageMaker endpoint
import json
import boto3
import base64

sagemaker_runtime = boto3.client('sagemaker-runtime')

def lambda_handler(event, context):
    for record in event['Records']:
        payload = json.loads(base64.b64decode(record['kinesis']['data']))
        
        # Format features for inference
        features = f"{payload['feature1']},{payload['feature2']},{payload['feature3']}"
        
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName='my-ml-endpoint',
            ContentType='text/csv',
            Body=features
        )
        
        prediction = json.loads(response['Body'].read())
        print(f"Prediction: {prediction}")
```

---

## 9. Key Facts & Exam Tips

### Critical Facts for Domain 1

- **SageMaker Data Wrangler** → visual, no-code data prep, part of SageMaker Studio
- **Glue DataBrew** → standalone visual data prep (NOT part of SageMaker Studio)
- **Feature Store Online** → DynamoDB-backed, < 10ms latency, latest values only
- **Feature Store Offline** → S3-backed Parquet, time-travel, for training
- **FastFile Mode** > Pipe Mode > File Mode for large datasets
- **RecordIO-Protobuf** → most efficient format for SageMaker built-in algorithms
- **Point-in-Time queries** → always use event_time to prevent data leakage
- **SageMaker Processing** → use for medium-large preprocessing; runs on managed infra

### Common Exam Scenarios

| Scenario | Best Solution |
|---------|--------------|
| Need visual data prep in SageMaker Studio | SageMaker Data Wrangler |
| Need consistent features across training & real-time serving | SageMaker Feature Store |
| Need to process large datasets (TBs) with Spark | SageMaker Processing with PySparkProcessor |
| Need ETL pipeline with scheduling and lineage | AWS Glue |
| Need serverless SQL analysis of S3 data | Amazon Athena |
| Need to detect data quality issues automatically | SageMaker Model Monitor (baseline) |
| Need real-time feature updates from streaming | Kinesis → Feature Store (put_record) |
| Training data drifts over time | Retrain with SageMaker Pipelines + Model Monitor trigger |

### Watch Out For

- **Data leakage**: Always split train/test BEFORE fitting scalers/encoders
- **Training-serving skew**: Use Feature Store to avoid different preprocessing in training vs inference
- **Imbalanced classes**: Use stratified splits; evaluate with F1/AUC not accuracy
- **S3 partitioning**: Partition by date/region for Athena query performance
- **Glue vs Data Wrangler**: Glue = ETL at scale with scheduling; Data Wrangler = interactive prep in Studio
