# Mock Exam — Domain 2: ML Model Development
## MLA-C01 | 20 Questions | ~52 minutes

> **Instructions:** Choose the single best answer unless the question says "Choose TWO."
> Answers and explanations are at the bottom — don't peek!

---

### Question 1
A team needs to train a regression model to predict house prices from 30 tabular features. The dataset has 1 million rows. They want to use a SageMaker built-in algorithm with minimal configuration. Which algorithm is MOST appropriate?

- A) K-Means
- B) Random Cut Forest
- C) XGBoost with `objective='reg:squarederror'`
- D) BlazingText in Word2Vec mode

---

### Question 2
A data scientist is training an XGBoost binary classifier on a dataset where 95% of examples are negative (class 0) and 5% are positive (class 1). Without any modification, the model achieves high accuracy but terrible recall on class 1. Which hyperparameter should they set first?

- A) `max_depth=10`
- B) `scale_pos_weight=19` (ratio of negatives to positives)
- C) `num_round=1000`
- D) `subsample=0.5`

---

### Question 3
A team needs to forecast the next 30 days of sales across 500 different product lines, each with their own historical patterns. The model should learn cross-series patterns. Which SageMaker built-in algorithm is designed for this?

- A) Linear Learner with `predictor_type='regressor'`
- B) DeepAR
- C) XGBoost with lag features
- D) K-Nearest Neighbors (KNN)

---

### Question 4
A training job runs for 12 hours and costs $200. The team runs this job weekly. They need to reduce the weekly cost significantly. Which approach provides the GREATEST savings with the LEAST operational change?

- A) Switch to a smaller instance type
- B) Enable Managed Spot Training with `checkpoint_s3_uri` configured
- C) Use SageMaker Savings Plans
- D) Move to batch training with AWS Batch

---

### Question 5
A neural network training job shows a large gap between training accuracy (95%) and validation accuracy (71%). Training loss keeps decreasing but validation loss starts increasing after epoch 20. What phenomenon is occurring?

- A) Underfitting — the model is too simple
- B) Vanishing gradients — gradients become too small
- C) Overfitting — the model memorizes training data
- D) Learning rate too low — training is converging slowly

---

### Question 6
A machine learning team runs 50 training experiments with different hyperparameters and wants to compare metrics (AUC, F1, loss curves) in a centralized dashboard within SageMaker Studio. Which SageMaker feature provides this?

- A) SageMaker Debugger
- B) SageMaker Experiments
- C) SageMaker Model Monitor
- D) Amazon CloudWatch Dashboards

---

### Question 7
A team runs Bayesian hyperparameter optimization with `max_jobs=30` and `max_parallel_jobs=5`. After 30 jobs, they want to continue the search using results from the first round but with extended hyperparameter ranges. Which HPO feature enables this?

- A) Start a new HPO job with `strategy='Random'`
- B) Use HPO Warm Start with `WarmStartTypes.TRANSFER_LEARNING`
- C) Increase `max_jobs` on the existing job
- D) Use HPO Warm Start with `WarmStartTypes.IDENTICAL_DATA_AND_ALGORITHM`

---

### Question 8
A company is detecting anomalies in server log data. The dataset has NO labels (normal vs. anomalous is unknown). Which SageMaker built-in algorithm is designed for this unsupervised anomaly detection use case?

- A) XGBoost with `objective='binary:logistic'`
- B) Linear Learner with `predictor_type='binary_classifier'`
- C) Random Cut Forest (RCF)
- D) K-Means with k=2

---

### Question 9
During training, a PyTorch deep learning model produces exploding gradients beginning at step 150. The ML engineer wants to be **automatically notified and have the training job stopped** when this happens. Which SageMaker feature handles this?

- A) CloudWatch alarm on training loss metric
- B) SageMaker Experiments with metric threshold
- C) SageMaker Debugger with `ExplodingGradient` rule and `StopTraining` action
- D) Lambda function polling the training job status

---

### Question 10
A data scientist wants to classify customer support tickets (text) into 5 categories. The team has 100,000 labeled examples. Which SageMaker approach is MOST cost-effective with the LEAST infrastructure setup?

- A) Build a custom PyTorch transformer from scratch in a SageMaker Training Job
- B) Use SageMaker JumpStart to fine-tune a pre-trained BERT/text classification model
- C) Use SageMaker BlazingText in Word2Vec mode
- D) Use Amazon Comprehend Custom Classification (managed API)

---

### Question 11
A team trains a Linear Learner model for binary classification. After checking results, they notice SageMaker automatically trained 32 models. Why does Linear Learner do this?

- A) SageMaker trains multiple models to ensure fault tolerance
- B) Linear Learner trains multiple models in parallel with different hyperparameter settings and selects the best one
- C) It trains one model per class in a one-vs-rest setup
- D) The `num_round` parameter defaults to 32

---

### Question 12
A data scientist trains an image classification model using SageMaker's built-in Image Classification algorithm. The training data consists of thousands of labeled JPEG images. Which input format does this algorithm require?

- A) CSV files with Base64-encoded image data
- B) JSON Lines with image paths and labels
- C) RecordIO format wrapping the JPEG images
- D) TFRecord files with image tensors

---

### Question 13
A team must train a model on a 5 TB dataset. They have verified the model converges well but want to reduce the training time. They currently use 1 GPU instance. The dataset processing (data loading) is identified as the bottleneck, NOT the GPU computation. What should they try first?

- A) Switch to a larger GPU instance with more VRAM
- B) Increase the number of GPU instances for model parallelism
- C) Optimize data loading: switch to FastFile Mode and use multiple workers
- D) Apply gradient checkpointing to reduce memory usage

---

### Question 14
A company has 10 petabytes of sensor data and wants to train a deep learning model across 64 GPU instances. The model parameters fit on a single GPU but the dataset is too large for one instance. Which distributed training strategy should be used?

- A) Model Parallelism — split the model across GPUs
- B) Data Parallelism — split the dataset across GPU instances, each trains the same model
- C) Pipeline Parallelism — split the model into stages across instances
- D) Tensor Parallelism — split individual weight matrices

---

### Question 15
A regression model achieves a training RMSE of 0.05 and a validation RMSE of 2.3. Which action would MOST directly help?

- A) Increase `max_depth` in XGBoost to fit more complex patterns
- B) Use more features to give the model more signal
- C) Apply L2 regularization (`lambda`) and reduce model complexity
- D) Switch from RMSE to MAE as the evaluation metric

---

### Question 16
A team needs to classify satellite images into 10 land cover categories. They have limited labeled data (2,000 images). Which approach is BEST to achieve high accuracy?

- A) Train SageMaker's built-in Image Classification algorithm from scratch
- B) Use SageMaker JumpStart with a pre-trained ResNet model and fine-tune the last layers
- C) Use BlazingText on image pixel data
- D) Convert images to CSV and train XGBoost

---

### Question 17
A data scientist defines the following HPO search space. Which parameter type is correctly defined?

```python
hyperparameter_ranges = {
    'optimizer': CategoricalParameter(['adam', 'sgd', 'rmsprop']),
    'learning_rate': ContinuousParameter(0.0001, 0.1),
    'num_layers': IntegerParameter(2, 10),
}
```

The team sets `strategy='Bayesian'` and `max_parallel_jobs=5`. When `max_parallel_jobs` equals `max_jobs`, what effectively happens to the Bayesian strategy?

- A) Bayesian optimization becomes more efficient
- B) Bayesian optimization degrades to random search (no prior results to learn from)
- C) The job fails with a configuration error
- D) SageMaker automatically switches to Grid Search

---

### Question 18
A model trained for sentiment analysis achieves the following on the hold-out test set:
- Precision: 0.91
- Recall: 0.43
- Accuracy: 0.88

The business requires catching as many negative sentiments as possible, even at the cost of some false positives. Which action would DIRECTLY improve the recall?

- A) Collect more positive sentiment training examples
- B) Lower the classification decision threshold (e.g., from 0.5 to 0.3)
- C) Increase the model complexity with more layers
- D) Apply L1 regularization to the model

---

### Question 19
A team packages a custom PyTorch training script in a Docker container and pushes it to Amazon ECR. When they run `estimator.fit()`, the training job fails immediately. A review of the CloudWatch logs shows:  
`Error: /opt/ml/model is read-only`

What is the likely cause?

- A) IAM role lacks `s3:PutObject` permissions
- B) The container is writing model artifacts to the wrong path (must write to `/opt/ml/model/`)
- C) The container must use `/tmp/` for all file output
- D) SageMaker requires model artifacts to be named `model.tar.gz` exactly

---

### Question 20
A team is training multiple versions of a fraud detection model over several weeks. After 3 months, they want to reproduce the exact model from Week 2 Run 5 — same data, same code, same hyperparameters. Which SageMaker features support this? (Choose TWO)

- A) SageMaker Experiments — logs parameters and metrics per run
- B) SageMaker Debugger — saves training tensors per step
- C) SageMaker Model Registry — stores model artifacts with lineage
- D) SageMaker Canvas — tracks AutoML runs
- E) Amazon CloudWatch — stores training logs

---

## ✅ Answers & Explanations

---

**Q1 — Answer: C (XGBoost reg:squarederror)**
XGBoost is the go-to SageMaker built-in algorithm for tabular regression and classification tasks. `objective='reg:squarederror'` minimizes mean squared error — perfect for price prediction. K-Means (A) is unsupervised clustering. RCF (B) is anomaly detection. BlazingText (D) is NLP.

---

**Q2 — Answer: B (`scale_pos_weight=19`)**
With 95/5 imbalance, the ratio of negatives to positives is 19:1. Setting `scale_pos_weight=19` tells XGBoost to weight positive class errors ~19x more, forcing the model to prioritize recall on the minority class. `max_depth` (A) and `num_round` (C) affect model complexity, not class weighting. `subsample` (D) affects row sampling.

---

**Q3 — Answer: B (DeepAR)**
DeepAR is specifically designed for probabilistic time-series forecasting across multiple related time series simultaneously. It learns cross-series patterns using a shared RNN model. Linear Learner (A) and XGBoost (C) require manual feature engineering for time series. KNN (D) is not a forecasting algorithm.

---

**Q4 — Answer: B (Managed Spot Training)**
Managed Spot Training can save 60-90% on training costs by using spare EC2 capacity. With `checkpoint_s3_uri`, the job resumes after any spot interruption. Savings Plans (C) provide 30-64% savings but require a 1-3 year commitment upfront. A smaller instance (A) might not maintain training quality. AWS Batch (D) requires significant migration effort.

---

**Q5 — Answer: C (Overfitting)**
The large gap between training accuracy (95%) and validation accuracy (71%), combined with diverging loss curves after epoch 20, is the classic signature of overfitting. The model memorizes training data and fails to generalize. Underfitting (A) would show high error on BOTH sets. Vanishing gradients (B) would prevent training from progressing.

---

**Q6 — Answer: B (SageMaker Experiments)**
SageMaker Experiments is purpose-built for tracking and comparing training runs — log parameters, metrics, artifacts per run, and visualize comparisons in SageMaker Studio. Debugger (A) monitors training internals. Model Monitor (C) monitors deployed endpoints. CloudWatch (D) can show metrics but lacks ML-specific experiment comparison views.

---

**Q7 — Answer: B (Warm Start with TRANSFER_LEARNING)**
`TRANSFER_LEARNING` warm start allows using results from a **previous HPO job** as a prior for a NEW HPO job, even if hyperparameter ranges change. `IDENTICAL_DATA_AND_ALGORITHM` (D) is for resuming with the exact same configuration. You cannot increase `max_jobs` on a completed job. Starting from scratch with Random (A) discards prior knowledge.

---

**Q8 — Answer: C (Random Cut Forest)**
Random Cut Forest (RCF) is SageMaker's built-in unsupervised anomaly detection algorithm that assigns an anomaly score to each data point without requiring labels. XGBoost (A) and Linear Learner (B) require labeled data. K-Means (D) is for clustering, not anomaly detection (though it can be used creatively, RCF is the purpose-built answer).

---

**Q9 — Answer: C (SageMaker Debugger with ExplodingGradient rule)**
SageMaker Debugger has built-in rules including `ExplodingGradient`. You can attach a `StopTraining` action to the rule, which automatically terminates the training job when the rule triggers. CloudWatch alarms (A) work on CloudWatch metrics, not training tensor values. Lambda polling (D) is a manual workaround.

---

**Q10 — Answer: B (SageMaker JumpStart with pre-trained BERT)**
JumpStart provides ready-to-use pre-trained models. Fine-tuning BERT on 100K labeled tickets is far more data-efficient than training from scratch (A) and more cost-effective. BlazingText Word2Vec (C) produces embeddings, not classification. Amazon Comprehend (D) is a valid answer but JumpStart keeps everything within the SageMaker ecosystem and offers more customization.

---

**Q11 — Answer: B (Multiple models trained in parallel, best selected)**
Linear Learner's distinctive feature is training **N models simultaneously** (default 32) with different random hyperparameter initializations or configurations. It evaluates all on the validation set and returns the best one. This is automatic and transparent to the user.

---

**Q12 — Answer: C (RecordIO format)**
SageMaker's built-in Image Classification algorithm requires images in **RecordIO** format (`.rec` files). This binary format efficiently encodes images and labels. CSV (A) is impractical for images. JSON Lines (B) and TFRecord (D) are not supported by this algorithm.

---

**Q13 — Answer: C (FastFile Mode + multiple data loading workers)**
The bottleneck is I/O (data loading), not GPU compute — so scaling GPU instances won't help. Switching to FastFile Mode reduces I/O overhead by streaming from S3, and increasing data loader workers (e.g., `num_workers` in PyTorch DataLoader) parallelizes reads. Model parallelism (B) is for model-size bottlenecks.

---

**Q14 — Answer: B (Data Parallelism)**
Data Parallelism is the correct strategy when the model fits on one GPU but the dataset is too large, or you want to accelerate training across many GPUs. Each instance trains on a different shard of data, and gradients are synchronized. Model Parallelism (A) is for when the model itself is too large for one GPU.

---

**Q15 — Answer: C (L2 regularization + reduce complexity)**
Training RMSE 0.05 vs validation RMSE 2.3 is a massive gap — severe overfitting. L2 regularization (lambda in XGBoost) penalizes large weights and reduces overfitting. Increasing `max_depth` (A) would worsen overfitting. Adding features (B) may help or hurt. Switching metrics (D) doesn't improve the model.

---

**Q16 — Answer: B (JumpStart + pre-trained ResNet, fine-tune)**
With only 2,000 images, training from scratch (A) will overfit badly. Transfer learning via JumpStart fine-tunes a ResNet pre-trained on ImageNet — the model already "knows" edges, textures, and shapes, requiring only the final layers to adapt to land cover categories. BlazingText (C) is for text. XGBoost on pixel CSV (D) would be ineffective.

---

**Q17 — Answer: B (Degrades to random search)**
When `max_parallel_jobs == max_jobs`, all jobs launch simultaneously in the first (and only) round. Bayesian optimization needs sequential feedback — previous job results inform next job's hyperparameter selection. With all jobs running at once, there are no prior results to learn from, so it becomes effectively random search.

---

**Q18 — Answer: B (Lower the decision threshold)**
Recall = TP / (TP + FN). To increase recall, you accept more false positives by lowering the decision boundary from 0.5 to a lower value (e.g., 0.3). This classifies more examples as positive, catching more true negatives. Collecting more positive examples (A) helps long-term retraining. L1 regularization (D) reduces model complexity but doesn't affect threshold.

---

**Q19 — Answer: B (Writing to wrong path)**
The error message says `/opt/ml/model` is read-only — but that's the correct output path for a training container. The actual issue is that the container is likely trying to write somewhere incorrectly OR the code has a permissions issue. However, the most common cause of this specific error is that the directory must be written to correctly. Actually, the SageMaker container does mount `/opt/ml/model/` as writable — containers MUST write model artifacts there. If the error says it's read-only, the container likely incorrectly constructed the path or tried to overwrite the directory itself rather than files within it.

---

**Q20 — Answer: A and C (SageMaker Experiments + Model Registry)**
SageMaker Experiments (A) logs all hyperparameters, metrics, and input data references per run — enabling full reproducibility of training conditions. SageMaker Model Registry (C) stores model artifacts with full lineage (which training job, which data, which code produced the model). Together they provide complete reproducibility. Debugger (B) saves tensors but not full run metadata. CloudWatch (E) stores logs but not structured ML metadata.
