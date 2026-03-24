# Lab Practice — Domain 1: Data Preparation for ML
## MLA-C01 | Hands-On Simulations | 6 Labs

> **Format:** Each lab gives you a realistic scenario, a setup context, step-by-step tasks to execute, and expected outcomes to verify. Labs are designed to be run in a SageMaker Studio or notebook environment.

---

## Lab 1: S3 Input Modes — FastFile vs Pipe vs File Mode

### Scenario
You have a 50 GB image dataset stored in S3. You need to benchmark training startup time and throughput using different S3 input modes.

### Prerequisites
- AWS account with SageMaker access
- S3 bucket with a sample dataset (use the SageMaker sample MNIST data as a proxy)
- IAM role with `AmazonSageMakerFullAccess`

### Tasks

**Step 1 — Set up the S3 dataset**
```python
import boto3, sagemaker
from sagemaker import get_execution_role

session = sagemaker.Session()
bucket = session.default_bucket()
role = get_execution_role()

# Upload a sample CSV or image dataset to S3
import numpy as np
import pandas as pd

# Generate synthetic training data
df = pd.DataFrame(np.random.randn(10000, 20), columns=[f"feature_{i}" for i in range(20)])
df["label"] = np.random.randint(0, 2, size=10000)
df.to_csv("train.csv", index=False)

s3_path = session.upload_data("train.csv", bucket=bucket, key_prefix="lab1/data")
print(f"Data uploaded to: {s3_path}")
```

**Step 2 — Launch training with FILE mode**
```python
from sagemaker.inputs import TrainingInput
from sagemaker.sklearn.estimator import SKLearn

file_input = TrainingInput(
    s3_data=f"s3://{bucket}/lab1/data/",
    input_mode="File"           # Downloads data before training starts
)

estimator = SKLearn(
    entry_point="train.py",     # your training script
    role=role,
    instance_type="ml.m5.xlarge",
    framework_version="1.0-1",
    instance_count=1,
)
# estimator.fit({"train": file_input})
print("File mode: downloads entire dataset to /opt/ml/input/data/train/ before training")
```

**Step 3 — Switch to FastFile mode**
```python
fastfile_input = TrainingInput(
    s3_data=f"s3://{bucket}/lab1/data/",
    input_mode="FastFile"       # POSIX-like access, no full download needed
)
print("FastFile mode: files appear as local paths immediately; data streamed on read")
print("Best for: large datasets where only a subset of data is accessed per epoch")
```

**Step 4 — Switch to Pipe mode**
```python
pipe_input = TrainingInput(
    s3_data=f"s3://{bucket}/lab1/data/",
    input_mode="Pipe"           # Streams data via named pipe (FIFO)
)
print("Pipe mode: requires training script to read from a named pipe")
print("Best for: RecordIO/TFRecord format with very large datasets")
```

### Verification Checklist
- [ ] `File` mode: check that `/opt/ml/input/data/train/` is fully populated at training start
- [ ] `FastFile` mode: training starts within seconds regardless of dataset size
- [ ] `Pipe` mode: training script must open `fifo` path, not a regular file path
- [ ] Compare CloudWatch training job duration metrics between modes

### Key Insight
> FastFile mode is the **default best choice** unless you specifically need Pipe mode for RecordIO streaming. File mode is safest for small datasets (<10 GB).

---

## Lab 2: SageMaker Feature Store — Ingestion and Retrieval

### Scenario
Build a Feature Store for a loan approval model. Store applicant features in both online (real-time serving) and offline (batch training) stores.

### Prerequisites
- SageMaker Studio or notebook with execution role that has `AmazonSageMakerFeatureStoreAccess`

### Tasks

**Step 1 — Create a Feature Group**
```python
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.feature_definition import (
    FeatureDefinition, FeatureTypeEnum
)
import pandas as pd, time

session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = session.default_bucket()
region = session.boto_region_name

feature_group_name = "loan-applicant-features"

feature_group = FeatureGroup(
    name=feature_group_name,
    sagemaker_session=session
)

# Step 2 — Define features
applicant_features = [
    FeatureDefinition("applicant_id",      FeatureTypeEnum.STRING),
    FeatureDefinition("age",               FeatureTypeEnum.INTEGRAL),
    FeatureDefinition("income",            FeatureTypeEnum.FRACTIONAL),
    FeatureDefinition("credit_score",      FeatureTypeEnum.INTEGRAL),
    FeatureDefinition("debt_to_income",    FeatureTypeEnum.FRACTIONAL),
    FeatureDefinition("event_time",        FeatureTypeEnum.FRACTIONAL),  # REQUIRED
]

feature_group.feature_definitions = applicant_features
```

**Step 3 — Create the Feature Group**
```python
feature_group.create(
    s3_uri=f"s3://{bucket}/feature-store/",
    record_identifier_name="applicant_id",    # unique key
    event_time_feature_name="event_time",     # required timestamp
    role_arn=role,
    enable_online_store=True,                 # for real-time lookups
    description="Loan applicant features for approval model"
)

# Wait for creation
import time
while feature_group.describe()["FeatureGroupStatus"] != "Created":
    print("Waiting for Feature Group creation...")
    time.sleep(5)
print("Feature Group created!")
```

**Step 4 — Ingest records (batch)**
```python
# Generate sample data
data = pd.DataFrame({
    "applicant_id": [f"APP_{i:05d}" for i in range(100)],
    "age":           [25 + i % 40 for i in range(100)],
    "income":        [30000.0 + i * 500 for i in range(100)],
    "credit_score":  [580 + i % 220 for i in range(100)],
    "debt_to_income":[0.2 + (i % 50) * 0.01 for i in range(100)],
    "event_time":    [time.time() for _ in range(100)],
})

# Batch ingest
feature_group.ingest(
    data_frame=data,
    max_workers=4,
    wait=True
)
print("Ingestion complete")
```

**Step 5 — Online Store lookup (real-time)**
```python
featurestore_runtime = session.boto_session.client(
    "sagemaker-featurestore-runtime",
    region_name=region
)

record = featurestore_runtime.get_record(
    FeatureGroupName=feature_group_name,
    RecordIdentifierValueAsString="APP_00042"
)
print("Online record:", record["Record"])

# Expected: returns the latest feature values for APP_00042 in <10ms
```

**Step 6 — Offline Store query via Athena**
```python
# Wait ~15 minutes for offline store sync after ingestion
query = feature_group.athena_query()
query_string = f"""
    SELECT applicant_id, credit_score, income
    FROM "{query.table_name}"
    WHERE credit_score > 700
    ORDER BY credit_score DESC
    LIMIT 10
"""

query.run(
    query_string=query_string,
    output_location=f"s3://{bucket}/athena-results/"
)
query.wait()
results = query.as_dataframe()
print(results)
```

**Step 7 — Point-in-time query**
```python
# Demonstrate the event_time filter — retrieve feature state AS OF a past timestamp
event_time_filter = "2024-01-15 00:00:00"
query_string_pit = f"""
    SELECT applicant_id, income, credit_score, event_time
    FROM "{query.table_name}"
    WHERE event_time <= to_unixtime(timestamp '{event_time_filter}')
"""
# This prevents data leakage by ensuring features only reflect info available at that time
print("Point-in-time queries prevent future data from leaking into past training labels")
```

### Verification Checklist
- [ ] Feature Group status shows `Created`
- [ ] Online `get_record` returns the correct applicant record in <50ms
- [ ] Offline Athena query returns filtered results correctly
- [ ] Point-in-time query correctly excludes records with `event_time` after the filter

---

## Lab 3: SageMaker Processing Job — Data Preprocessing at Scale

### Scenario
Your raw training data is in S3 as a 5 GB CSV file. You need to run a scikit-learn preprocessing pipeline (imputation, normalization, one-hot encoding) and output the processed train/validation/test splits back to S3.

### Tasks

**Step 1 — Write the preprocessing script**

Save as `preprocessing.py`:
```python
# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import argparse, os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-test-split-ratio", type=float, default=0.2)
    args = parser.parse_args()

    # SageMaker Processing Job input path
    input_path = "/opt/ml/processing/input/data.csv"
    df = pd.read_csv(input_path)

    print(f"Loaded {len(df)} rows")

    # Drop nulls in target, impute features
    df = df.dropna(subset=["label"])
    features = [c for c in df.columns if c != "label"]

    imputer = SimpleImputer(strategy="median")
    df[features] = imputer.fit_transform(df[features])

    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Split
    train, temp = train_test_split(df, test_size=args.train_test_split_ratio * 2, random_state=42)
    val, test   = train_test_split(temp, test_size=0.5, random_state=42)

    # SageMaker Processing Job output paths
    os.makedirs("/opt/ml/processing/output/train", exist_ok=True)
    os.makedirs("/opt/ml/processing/output/validation", exist_ok=True)
    os.makedirs("/opt/ml/processing/output/test", exist_ok=True)

    train.to_csv("/opt/ml/processing/output/train/train.csv",           index=False)
    val.to_csv(  "/opt/ml/processing/output/validation/validation.csv", index=False)
    test.to_csv( "/opt/ml/processing/output/test/test.csv",             index=False)

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
```

**Step 2 — Launch the Processing Job**
```python
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

processor = SKLearnProcessor(
    framework_version="1.0-1",
    role=role,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    base_job_name="lab3-preprocessing"
)

processor.run(
    code="preprocessing.py",
    inputs=[
        ProcessingInput(
            source=f"s3://{bucket}/lab3/raw/data.csv",
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output/train",
            destination=f"s3://{bucket}/lab3/processed/train",
            output_name="train"
        ),
        ProcessingOutput(
            source="/opt/ml/processing/output/validation",
            destination=f"s3://{bucket}/lab3/processed/validation",
            output_name="validation"
        ),
        ProcessingOutput(
            source="/opt/ml/processing/output/test",
            destination=f"s3://{bucket}/lab3/processed/test",
            output_name="test"
        ),
    ],
    arguments=["--train-test-split-ratio", "0.2"]
)
```

**Step 3 — Inspect outputs**
```python
# After job completes:
outputs = processor.jobs[-1].outputs
for output in outputs:
    print(f"{output.output_name}: {output.destination}")

# Verify files exist in S3
s3 = boto3.client("s3")
for prefix in ["lab3/processed/train/", "lab3/processed/validation/", "lab3/processed/test/"]:
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    for obj in response.get("Contents", []):
        print(obj["Key"], obj["Size"])
```

### Verification Checklist
- [ ] Processing Job completes with `Completed` status
- [ ] Three output folders exist in S3: `train/`, `validation/`, `test/`
- [ ] Row counts sum to original dataset size
- [ ] No nulls remain in output CSV (imputation worked)
- [ ] Feature values are approximately N(0,1) distributed (StandardScaler worked)

---

## Lab 4: AWS Glue ETL + SageMaker Data Wrangler — Feature Engineering Pipeline

### Scenario
Raw e-commerce transaction data is cataloged in Glue Data Catalog. You need to use Glue ETL for heavy transformation and then use Data Wrangler to interactively explore and engineer features.

### Tasks

**Step 1 — Create a Glue Database and Crawler**
```bash
# AWS CLI
aws glue create-database --database-input '{"Name": "ecommerce_lab"}'

aws glue create-crawler \
  --name ecommerce-crawler \
  --role arn:aws:iam::ACCOUNT_ID:role/GlueServiceRole \
  --database-name ecommerce_lab \
  --targets '{"S3Targets": [{"Path": "s3://YOUR_BUCKET/lab4/raw/"}]}'

aws glue start-crawler --name ecommerce-crawler
# Wait for crawler to complete (RUNNING → READY)
aws glue get-crawler --name ecommerce-crawler --query "Crawler.State"
```

**Step 2 — Write a Glue ETL job script**

Save as `glue_etl.py`:
```python
# glue_etl.py — Glue ETL PySpark job
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql import functions as F

args = getResolvedOptions(sys.argv, ["JOB_NAME", "output_path"])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

# Read from Glue Data Catalog
datasource = glueContext.create_dynamic_frame.from_catalog(
    database="ecommerce_lab",
    table_name="raw_transactions"
)

df = datasource.toDF()

# Feature engineering
df = df.withColumn("days_since_last_purchase",
        F.datediff(F.current_date(), F.col("last_purchase_date")))

df = df.withColumn("log_transaction_amount",
        F.log1p(F.col("transaction_amount")))     # log-transform skewed feature

df = df.withColumn("is_weekend",
        (F.dayofweek(F.col("transaction_date")).isin([1, 7])).cast("int"))

# Remove outliers (transaction_amount > 99th percentile)
p99 = df.approxQuantile("transaction_amount", [0.99], 0.01)[0]
df = df.filter(F.col("transaction_amount") <= p99)

# Write output as Parquet
output_df = DynamicFrame.fromDF(df, glueContext, "output")
glueContext.write_dynamic_frame.from_options(
    frame=output_df,
    connection_type="s3",
    connection_options={"path": args["output_path"]},
    format="parquet"
)
print(f"Wrote {df.count()} rows to {args['output_path']}")
```

**Step 3 — Create and run the Glue job**
```bash
aws glue create-job \
  --name ecommerce-feature-engineering \
  --role arn:aws:iam::ACCOUNT_ID:role/GlueServiceRole \
  --command '{"Name":"glueetl","ScriptLocation":"s3://YOUR_BUCKET/scripts/glue_etl.py","PythonVersion":"3"}' \
  --default-arguments '{"--output_path":"s3://YOUR_BUCKET/lab4/processed/","--job-language":"python"}'

aws glue start-job-run \
  --job-name ecommerce-feature-engineering \
  --arguments '{"--output_path":"s3://YOUR_BUCKET/lab4/processed/"}'
```

**Step 4 — Explore in Data Wrangler (SageMaker Studio)**
1. Open SageMaker Studio → Data Wrangler → New Flow
2. Add data source → Amazon S3 → select `lab4/processed/*.parquet`
3. In the Data Flow, add a Transform:
   - **Handle missing values** → Fill with median for `log_transaction_amount`
   - **Encode categorical** → One-hot encode `product_category`
   - **Feature importance** → Run quick model to rank features (uses Random Forest internally)
4. Click **Analyze** → Run **Bias Report**:
   - Target column: `conversion` (0/1)
   - Facet: `customer_gender`
   - Review Class Imbalance (CI) and DPL metrics
5. **Export** the flow as a Processing Job to S3

### Verification Checklist
- [ ] Glue Crawler creates a table in `ecommerce_lab` database
- [ ] Glue ETL job runs successfully (status: `SUCCEEDED`)
- [ ] Output Parquet files appear in `lab4/processed/`
- [ ] Data Wrangler flow loads parquet data without errors
- [ ] Bias Report shows metrics for the selected facet column
- [ ] Feature importance chart ranks `log_transaction_amount` highly

---

## Lab 5: Kinesis Data Streams — Real-Time Feature Ingestion

### Scenario
A fraud detection model needs features computed from live transaction events. Create a Kinesis stream, produce events, and consume/process them to update the Feature Store in near-real-time.

### Tasks

**Step 1 — Create a Kinesis stream**
```bash
aws kinesis create-stream \
  --stream-name transaction-events \
  --shard-count 2

# Wait for stream to become ACTIVE
aws kinesis describe-stream-summary \
  --stream-name transaction-events \
  --query "StreamDescriptionSummary.StreamStatus"
```

**Step 2 — Produce events (Python producer)**
```python
import boto3, json, time, random

kinesis = boto3.client("kinesis", region_name="us-east-1")

def produce_transaction(stream_name, count=100):
    for i in range(count):
        record = {
            "transaction_id": f"TXN-{i:06d}",
            "customer_id":    f"CUST-{random.randint(1, 1000):04d}",
            "amount":         round(random.uniform(5.0, 5000.0), 2),
            "merchant_category": random.choice(["grocery", "travel", "online", "restaurant"]),
            "timestamp":      time.time()
        }
        kinesis.put_record(
            StreamName=stream_name,
            Data=json.dumps(record),
            PartitionKey=record["customer_id"]   # routes same customer to same shard
        )
        if i % 10 == 0:
            print(f"Produced {i+1} records")
        time.sleep(0.05)

produce_transaction("transaction-events", count=50)
```

**Step 3 — Consume and process with Lambda**

Lambda function code (deploy via console or SAM):
```python
# lambda_handler.py
import json, boto3, time

featurestore = boto3.client("sagemaker-featurestore-runtime")
FEATURE_GROUP = "transaction-features"

def lambda_handler(event, context):
    records_processed = 0
    for record in event["Records"]:
        # Decode Kinesis record
        payload = json.loads(record["kinesis"]["data"])   # base64-decoded automatically

        # Compute derived features
        is_high_value = 1 if payload["amount"] > 1000 else 0
        hour_of_day   = int(payload["timestamp"] % 86400 / 3600)

        # Upsert into Feature Store online store
        featurestore.put_record(
            FeatureGroupName=FEATURE_GROUP,
            Record=[
                {"FeatureName": "customer_id",        "ValueAsString": payload["customer_id"]},
                {"FeatureName": "last_amount",         "ValueAsString": str(payload["amount"])},
                {"FeatureName": "is_high_value",       "ValueAsString": str(is_high_value)},
                {"FeatureName": "hour_of_day",         "ValueAsString": str(hour_of_day)},
                {"FeatureName": "merchant_category",   "ValueAsString": payload["merchant_category"]},
                {"FeatureName": "event_time",          "ValueAsString": str(payload["timestamp"])},
            ]
        )
        records_processed += 1

    return {"statusCode": 200, "body": f"Processed {records_processed} records"}
```

**Step 4 — Add Kinesis trigger to Lambda**
```bash
aws lambda create-event-source-mapping \
  --function-name transaction-feature-updater \
  --event-source-arn arn:aws:kinesis:us-east-1:ACCOUNT_ID:stream/transaction-events \
  --starting-position LATEST \
  --batch-size 25 \
  --bisect-batch-on-function-error true     # split batch if Lambda errors
```

### Verification Checklist
- [ ] Kinesis stream shows `ACTIVE` status with 2 shards
- [ ] Producer sends 50 records without errors
- [ ] Lambda function invocations visible in CloudWatch Logs
- [ ] Feature Store `get_record` for an ingested customer returns updated `last_amount`
- [ ] CloudWatch `GetRecords.IteratorAgeMilliseconds` stays near 0 (no lag)

---

## Lab 6: Amazon Athena — Querying ML Training Data

### Scenario
Your processed data lives in S3 as Parquet files partitioned by date. Use Athena to run analytical queries that will feed ML training data selection.

### Tasks

**Step 1 — Register data in Glue Data Catalog**
```sql
-- Run in Athena Query Editor
CREATE EXTERNAL TABLE IF NOT EXISTS ecommerce_lab.processed_transactions (
    customer_id         STRING,
    transaction_amount  DOUBLE,
    log_amount         DOUBLE,
    product_category    STRING,
    is_weekend          INT,
    conversion          INT
)
PARTITIONED BY (transaction_date STRING)
STORED AS PARQUET
LOCATION 's3://YOUR_BUCKET/lab4/processed/'
TBLPROPERTIES ('parquet.compress'='SNAPPY');

-- Load partitions automatically
MSCK REPAIR TABLE ecommerce_lab.processed_transactions;
```

**Step 2 — Exploratory queries for ML**
```sql
-- Label distribution check
SELECT conversion, COUNT(*) as count,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS pct
FROM ecommerce_lab.processed_transactions
GROUP BY conversion;

-- Feature statistics by class
SELECT
    conversion,
    AVG(transaction_amount)    AS avg_amount,
    STDDEV(transaction_amount) AS std_amount,
    PERCENTILE_APPROX(transaction_amount, 0.5) AS median_amount
FROM ecommerce_lab.processed_transactions
GROUP BY conversion;

-- Missing value audit
SELECT
    COUNT(*) AS total,
    SUM(CASE WHEN transaction_amount IS NULL THEN 1 ELSE 0 END) AS null_amount,
    SUM(CASE WHEN product_category  IS NULL THEN 1 ELSE 0 END) AS null_category
FROM ecommerce_lab.processed_transactions;

-- Sample training data export back to S3 (CTAS)
CREATE TABLE ecommerce_lab.training_set
WITH (
    format='PARQUET',
    external_location='s3://YOUR_BUCKET/training-set/',
    partitioned_by=ARRAY['transaction_date']
)
AS
SELECT * FROM ecommerce_lab.processed_transactions
WHERE transaction_date BETWEEN '2024-01-01' AND '2024-12-31';
```

**Step 3 — Verify partition pruning (cost optimization)**
```sql
-- Good: uses partition column in WHERE → Athena scans only matching partitions
SELECT COUNT(*) FROM ecommerce_lab.processed_transactions
WHERE transaction_date = '2024-06-15';

-- Bad: no partition filter → full table scan
SELECT COUNT(*) FROM ecommerce_lab.processed_transactions
WHERE MONTH(CAST(transaction_date AS DATE)) = 6;

-- Check bytes scanned in Athena console → compare both queries
```

### Verification Checklist
- [ ] `MSCK REPAIR TABLE` loads all partitions correctly
- [ ] Label distribution query shows class ratios (flag if >90% imbalanced)
- [ ] CTAS creates a new `training_set` table in S3
- [ ] Partition-pruned query scans significantly fewer bytes than full-scan query
- [ ] No `HIVE_PARTITION_SCHEMA_MISMATCH` errors (column types match Parquet schema)

---

## Summary — Domain 1 Lab Skills Matrix

| Lab | Service | Skills Practiced |
|-----|---------|-----------------|
| 1 | S3 + SageMaker Training | File / Pipe / FastFile input modes |
| 2 | SageMaker Feature Store | Feature Group creation, online/offline ingestion, point-in-time queries |
| 3 | SageMaker Processing Jobs | SKLearnProcessor, input/output paths, preprocessing pipeline |
| 4 | Glue ETL + Data Wrangler | Crawlers, PySpark transforms, interactive feature engineering, bias report |
| 5 | Kinesis + Lambda + Feature Store | Real-time event streaming → feature updates |
| 6 | Athena + Glue Data Catalog | SQL analysis, CTAS, partition pruning |

### Common Mistakes to Avoid
- **Feature Store**: Forgetting `event_time` column → `FeatureGroup.create()` fails
- **Processing Jobs**: Writing to `/tmp/` instead of `/opt/ml/processing/output/` → outputs not uploaded to S3
- **Athena**: Not running `MSCK REPAIR TABLE` after adding new partitions → queries return 0 rows
- **Kinesis**: Using the same `PartitionKey` for all records → hot shard problem, uneven load
- **FastFile mode**: Using with very small files (<1 MB each) → S3 LIST overhead negates the benefit
