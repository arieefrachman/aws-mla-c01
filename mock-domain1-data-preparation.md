# Mock Exam — Domain 1: Data Preparation for Machine Learning
## MLA-C01 | 20 Questions | ~52 minutes

> **Instructions:** Choose the single best answer unless the question says "Choose TWO." 
> Answers and explanations are at the bottom — don't peek!

---

### Question 1
A data scientist is working with a 2 TB tabular dataset stored in Amazon S3. They need to train a SageMaker XGBoost model. The training instance has 64 GB of RAM. Which S3 input mode should they use to avoid out-of-memory errors during training?

- A) File Mode — download the entire dataset before training
- B) Pipe Mode — stream data using RecordIO format
- C) FastFile Mode — stream data via POSIX-compatible interface
- D) Transfer Acceleration — speed up the S3 download

---

### Question 2
A machine learning team stores raw clickstream data in Amazon S3, partitioned by date. They need to run ad-hoc SQL queries to explore feature distributions and check for null values before training. The solution must require no upfront infrastructure provisioning. Which service is MOST appropriate?

- A) Amazon Redshift with Spectrum
- B) Amazon EMR with PySpark
- C) Amazon Athena
- D) AWS Glue Studio

---

### Question 3
A team uses SageMaker Feature Store to serve features for real-time fraud detection. A transaction arrives, and the system must retrieve the customer's latest feature values in under 10 milliseconds. Which Feature Store component handles this?

- A) Offline Store (backed by Amazon S3)
- B) Online Store (backed by Amazon DynamoDB)
- C) Glue Data Catalog (metadata store)
- D) SageMaker Processing Job output

---

### Question 4
A company ingests sensor readings every 5 minutes into Amazon Kinesis Data Streams. Data must be delivered to Amazon S3 for downstream ML training with no custom code required for the delivery mechanism. Which service should be used between Kinesis Data Streams and S3?

- A) Amazon Kinesis Data Analytics
- B) AWS Lambda
- C) Amazon Kinesis Data Firehose
- D) Amazon EventBridge Pipes

---

### Question 5
A data engineer notices that a training dataset has a feature `annual_income` with a right-skewed distribution (skewness = 3.2). The team plans to use a linear regression model. Which transformation should they apply?

- A) One-hot encoding
- B) Log transform: `log(annual_income + 1)`
- C) Min-Max normalization to [0, 1]
- D) Ordinal encoding

---

### Question 6
A data scientist is building a churn prediction model. They fit a `StandardScaler` on the full dataset (train + test combined) and then split into train/test sets. A senior ML engineer flags this as a problem. What issue does this introduce?

- A) The model will train slower due to scaled features
- B) Data leakage — test set statistics influenced the scaler fit
- C) The model will underfit because features are normalized
- D) StandardScaler does not work on combined datasets

---

### Question 7
A team wants to run large-scale PySpark data transformation jobs on a 10 TB dataset stored in S3. The jobs run nightly and the team wants minimal infrastructure management. Which tool is MOST appropriate?

- A) SageMaker Processing Job with `SKLearnProcessor`
- B) Amazon EMR with a persistent cluster
- C) AWS Glue ETL job with PySpark
- D) AWS Lambda with pandas

---

### Question 8
A company is building a training dataset for a loan approval model. The dataset contains 80% male applicants and 20% female applicants. The data scientist wants to measure this representation imbalance before training. Which SageMaker tool generates a pre-training bias report?

- A) SageMaker Debugger
- B) SageMaker Model Monitor
- C) SageMaker Data Wrangler (Analyze → Bias Report)
- D) SageMaker Experiments

---

### Question 9
A data science team uses SageMaker Feature Store. They want to train a model using customer features as of January 1, 2024 — not the current values. Which mechanism allows this?

- A) Read from the Online Store with a timestamp filter
- B) Use the Offline Store with an Athena query filtering by `WHERE event_time <= '2024-01-01'`
- C) Create a snapshot of the Online Store on January 1, 2024
- D) Use the GetRecord API with a timestamp parameter

---

### Question 10
A team is building an NLP classifier. Their raw text column has 50,000 unique words. They want to convert texts to fixed-size numerical vectors while capturing word frequency and reducing the importance of common words like "the" and "is." Which encoding strategy should they use?

- A) One-hot encoding on the vocabulary
- B) Label encoding each unique word
- C) TF-IDF vectorization
- D) Ordinal encoding based on word frequency rank

---

### Question 11
A SageMaker Processing Job needs to read data from S3, apply feature engineering, and write train/validation/test splits back to S3. The Python script uses scikit-learn. Where must the script write its output files so that SageMaker can upload them to S3?

- A) `/tmp/output/`
- B) `/opt/ml/output/`
- C) The paths specified in `ProcessingOutput(source=...)` declarations
- D) `/opt/ml/input/data/`

---

### Question 12
A data engineer maintains a data catalog where hundreds of tables in Amazon S3 are automatically discovered and schema changes are tracked. Which AWS service provides this functionality?

- A) Amazon Athena
- B) AWS Glue Crawlers + Data Catalog
- C) AWS Lake Formation table definitions
- D) Amazon S3 Object Metadata

---

### Question 13
An ML team stores training datasets in Amazon S3 and needs to ensure that historical versions of datasets are preserved for reproducibility — even when new data is uploaded with the same key. Which S3 feature should they enable?

- A) S3 Object Replication
- B) S3 Transfer Acceleration
- C) S3 Versioning
- D) S3 Intelligent-Tiering

---

### Question 14
A classification training dataset has a 97/3 class imbalance. A data scientist evaluates the model using raw accuracy and reports 97% accuracy. A senior ML engineer says this metric is misleading. Which metric should be used instead?

- A) Mean Absolute Error (MAE)
- B) R² Score
- C) F1-Score or AUC-ROC
- D) Root Mean Square Error (RMSE)

---

### Question 15
A company needs to visually prepare data — joining tables, removing outliers, encoding categoricals, and detecting bias — as part of their SageMaker Studio workflow. The output should feed directly into SageMaker Pipelines. Which tool is designed for this?

- A) AWS Glue DataBrew
- B) Amazon Athena with CTAS queries
- C) SageMaker Data Wrangler
- D) SageMaker Ground Truth

---

### Question 16
A data scientist wants to eliminate highly correlated features before training a neural network. They compute a Pearson correlation matrix and find that features `feature_A` and `feature_B` have a correlation coefficient of 0.97. What should they do?

- A) Keep both features — high correlation doesn't affect neural networks
- B) Apply PCA to all features to orthogonalize them
- C) Remove one of the two correlated features to reduce multicollinearity
- D) Apply log transform to both features

---

### Question 17
For a SageMaker training job, a team chooses the **RecordIO-Protobuf** format for their dataset instead of CSV. What is the PRIMARY advantage of this format?

- A) RecordIO-Protobuf supports more column types than CSV
- B) It is more human-readable and easier to debug
- C) It is a compact binary format that enables faster data loading, especially with Pipe Mode
- D) It is the only format that SageMaker XGBoost supports

---

### Question 18
A company is ingesting IoT sensor data via Amazon Kinesis Data Streams. A Lambda function consumes from the stream, computes rolling averages over the last 5 minutes, and writes features to SageMaker Feature Store for real-time inference. The feature computation requires stateful windowing over multiple records. Which service is BETTER suited for this stateful aggregation than Lambda?

- A) Amazon SNS
- B) Amazon Kinesis Data Analytics (Apache Flink)
- C) Amazon S3 with lifecycle rules
- D) Amazon SQS FIFO queues

---

### Question 19
A team is preparing a dataset with a datetime column `purchase_date`. They want to capture the cyclical nature of months (so that December→January is treated as a small step, not a large jump). Which transformation should they apply?

- A) Ordinal encode the month as integers 1–12
- B) One-hot encode each month as 12 binary columns
- C) Apply sine and cosine transforms: `sin(2π × month / 12)` and `cos(2π × month / 12)`
- D) Extract the month as a raw integer feature

---

### Question 20
A company wants to ingest new features into SageMaker Feature Store in real time as transactions occur (one record at a time, from an application server). Which API / method should they use?

- A) `feature_group.ingest(data_frame=df, ...)` — batch ingestion
- B) `featurestore_runtime.put_record(FeatureGroupName=..., Record=[...])` — real-time ingestion
- C) AWS Glue job writing to the Feature Group S3 path directly
- D) SageMaker Processing Job with the feature group as output

---

## ✅ Answers & Explanations

---

**Q1 — Answer: C (FastFile Mode)**
FastFile Mode streams data from S3 with a POSIX-like file interface, supporting any format, without downloading the full dataset to instance storage. Pipe Mode (B) also streams but is limited to RecordIO format and is the legacy option. File Mode (A) would require downloading all 2 TB first. Transfer Acceleration (D) is an S3 feature for upload speed, not a SageMaker input mode.

---

**Q2 — Answer: C (Amazon Athena)**
Athena is serverless SQL that queries S3 data directly with zero infrastructure provisioning. Redshift Spectrum (A) requires a Redshift cluster. EMR (B) requires cluster start. Glue Studio (D) is for ETL pipeline building, not ad-hoc queries.

---

**Q3 — Answer: B (Online Store backed by DynamoDB)**
The Online Store is DynamoDB-backed and provides < 10ms point-in-time lookups of the **latest** feature values per record identifier — designed for real-time inference. The Offline Store (A) is S3-backed for historical/batch queries. Glue Data Catalog (C) stores metadata only.

---

**Q4 — Answer: C (Amazon Kinesis Data Firehose)**
Kinesis Data Firehose is a fully managed service that reliably delivers streaming data from Kinesis Data Streams to destinations like S3, Redshift, or OpenSearch — zero custom code. Lambda (B) would require coding. Kinesis Analytics (A) is for stream processing, not delivery. EventBridge Pipes (D) is for routing events, not streaming delivery.

---

**Q5 — Answer: B (Log transform)**
Log transform compresses the right tail of skewed distributions, making them more normally distributed — important for linear models that assume normality. One-hot (A) and ordinal (D) are for categorical data. Min-Max (C) normalizes range but does not reduce skewness.

---

**Q6 — Answer: B (Data Leakage)**
Fitting the scaler on combined train+test data "leaks" test set statistics (mean, std) into the training process. The model effectively "sees" the test set during training, causing overly optimistic evaluation results. Always fit transformers on training data ONLY, then apply them to test data.

---

**Q7 — Answer: C (AWS Glue ETL with PySpark)**
Glue ETL jobs run serverless PySpark — no cluster provisioning or management. EMR (B) is more flexible but requires cluster lifecycle management. SageMaker Processing with SKLearnProcessor (A) runs sklearn/pandas, not PySpark natively at this scale (though `PySparkProcessor` exists, Glue is more purpose-built for nightly ETL). Lambda (D) has a 15-min timeout and can't handle 10 TB.

---

**Q8 — Answer: C (SageMaker Data Wrangler)**
SageMaker Data Wrangler has an **Analyze** tab that generates pre-training bias reports using Clarify metrics (CI, DPL, etc.) directly on the training dataset, before a model is trained. Debugger (A) monitors training, not data. Model Monitor (B) monitors deployed endpoints. Experiments (D) tracks training runs.

---

**Q9 — Answer: B (Offline Store Athena query with event_time filter)**
The Offline Store in SageMaker Feature Store is S3-backed with Parquet files that include `event_time`. Athena queries support point-in-time filtering via `WHERE event_time <= '2024-01-01'`, retrieving the latest value as of that date. The Online Store (A) only holds the most recent values — no time-travel. `GetRecord` API (D) does not accept a timestamp parameter.

---

**Q10 — Answer: C (TF-IDF)**
TF-IDF (Term Frequency-Inverse Document Frequency) converts text to vectors that reflect word frequency in a document but down-weight common words that appear across all documents (high IDF penalty). One-hot (A) would create a 50,000-dimensional sparse vector per word — not text-level. Label encoding (B) is for categories, not NLP. Ordinal (D) loses semantic meaning.

---

**Q11 — Answer: C (Paths specified in ProcessingOutput source)**
SageMaker Processing Jobs upload files from the paths declared in `ProcessingOutput(source='/opt/ml/processing/train', ...)`. You define these source paths, and anything written there is uploaded to the specified S3 destination. `/opt/ml/output/` is for training job failure files, not processing output.

---

**Q12 — Answer: B (AWS Glue Crawlers + Data Catalog)**
Glue Crawlers automatically traverse S3 paths, infer schemas, and register table definitions in the Glue Data Catalog. The catalog is then queryable by Athena, EMR, and Redshift Spectrum. Athena (A) is a query engine, not a catalog service. Lake Formation (C) adds governance on top of the catalog but doesn't provide crawling.

---

**Q13 — Answer: C (S3 Versioning)**
S3 Versioning preserves every version of an object, even when overwritten with the same key. This is critical for dataset reproducibility in ML — you can reference a specific version of a training dataset by its VersionId. Replication (A) copies to another bucket but doesn't version. Intelligent-Tiering (D) manages storage costs, not versioning.

---

**Q14 — Answer: C (F1-Score or AUC-ROC)**
With 97/3 imbalance, a model that predicts the majority class for every example achieves 97% accuracy automatically — it's meaningless. F1-Score balances precision and recall on the minority class. AUC-ROC measures ranking quality independent of threshold. MAE and RMSE (A, D) are regression metrics. R² (B) is for regression.

---

**Q15 — Answer: C (SageMaker Data Wrangler)**
Data Wrangler is the tool within SageMaker Studio for visual, interactive data preparation. It natively exports `.flow` configurations to SageMaker Pipelines as ProcessingSteps. Glue DataBrew (A) is a valid alternative but is NOT integrated into SageMaker Studio. Athena (B) is SQL only. Ground Truth (D) is for data labeling.

---

**Q16 — Answer: C (Remove one correlated feature)**
Correlation of 0.97 indicates near-redundant information. While neural networks can sometimes handle this, removing one prevents multicollinearity, reduces noise, and speeds up training. PCA (B) is a valid approach but more complex; removing one feature is simpler and equally effective for a single pair. A (keeping both) is incorrect best practice.

---

**Q17 — Answer: C (Compact binary format, faster loading)**
RecordIO-Protobuf is a binary format that is smaller on disk and faster to read than CSV, particularly when used with Pipe or FastFile Mode — the training instance can process data faster because I/O is lower. It is NOT the only format XGBoost supports (D is false — XGBoost supports CSV and LibSVM as well).

---

**Q18 — Answer: B (Amazon Kinesis Data Analytics / Apache Flink)**
Kinesis Data Analytics with Apache Flink provides stateful stream processing with native windowing functions (tumbling, sliding, session windows) — designed for exactly this use case: aggregations over time windows. Lambda (A would be Lambda but it's not listed — C is S3) is stateless and cannot maintain state across invocations without external storage.

---

**Q19 — Answer: C (Sine and Cosine transforms)**
Cyclical encoding with sin/cos preserves the circular nature of time: December (month 12) sin/cos values are close to January (month 1) values. Ordinal encoding (A) creates a false large gap between Dec (12) and Jan (1). One-hot (B) doesn't capture proximity. Raw integer (D) has the same problem as ordinal.

---

**Q20 — Answer: B (`featurestore_runtime.put_record()`)**
Real-time, single-record ingestion uses the `sagemaker-featurestore-runtime` boto3 client's `put_record` API. This writes to the Online Store immediately. `feature_group.ingest()` (A) is batch ingestion of DataFrames. Writing to the S3 path directly (C) bypasses the Feature Store APIs and would corrupt the format. Processing Jobs (D) are batch and not real-time.
