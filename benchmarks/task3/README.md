# Task 3: Context Aware Anomaly Classification

## Overview

Task 3 evaluates whether Large Language Models (LLMs) can determine whether an observed traffic deviation represents a normal condition, a minor anomaly, or a major anomaly after considering the surrounding urban context.

Unlike a simple traffic deviation classification task, the model must interpret the observed and expected traffic volumes together with contextual factors such as weather conditions, nearby events, road crashes, public holidays, school holidays, and the surrounding urban environment.

## Repository Contents

The `benchmarks/task3` directory contains the scripts, documentation, and example data used to construct and evaluate Task 3.

```text
task3/
├── Examples/
│   └── anomaly_benchmark_station_sample.csv
├── .gitkeep
├── README.md
├── build_task3_benchmark.py
├── feature_task3.py
└── llm_eval_task3.py
```

### `feature_task3.py`

Performs the feature engineering required for Task 3. It reads the station-hour master context table and constructs:

- Expected traffic volume
- Traffic change and percentage deviation
- Traffic direction
- Rain, crash, event, wind, and holiday severity scores
- Context severity score
- Context complexity
- Context-aware anomaly labels
- Anomaly reasons
- Natural-language scenario descriptions

Main outputs:

```text
master_context_table_with_anomaly_features.csv
task3_anomaly_station_sample.csv
```

### `build_task3_benchmark.py`

Constructs the final balanced Task 3 benchmark from the anomaly-labelled observations. It creates the LLM classification questions, maps anomaly classes to `A`, `B`, and `C`, and samples the final evaluation set.

Main output:

```text
task3_anomaly_classification.csv
```

### `llm_eval_task3.py`

Evaluates the eight selected LLMs on the Task 3 benchmark. It handles API calls, response parsing, checkpoints, resume/overwrite behaviour, aggregate metrics, per-class metrics, and confusion matrices.

### `Examples/`

Contains example Task 3 data generated during benchmark construction.

#### `anomaly_benchmark_station_sample.csv`

A station-representative anomaly sample used to inspect the structure and labels produced during Task 3 feature engineering. It is an example dataset and is separate from the final balanced 600-instance evaluation benchmark.

### `README.md`

Documents the Task 3 methodology, repository structure, anomaly framework, benchmark construction, evaluated models, evaluation protocol, metrics, and execution instructions.

### `.gitkeep`

A placeholder originally used to keep the `task3` directory tracked when it was empty. It contains no benchmark logic and is not required to run Task 3.

## Task Definition

Each benchmark instance is constructed from a single traffic observation extracted from the master context table.

Every observation is converted into a structured natural-language description containing information such as:

- Traffic monitoring station and road location
- Date and time
- Observed traffic volume
- Expected traffic volume
- Weather conditions
- Nearby road crashes
- Nearby events
- Public holiday status
- School holiday status
- Surrounding urban characteristics

### Traffic Deviation

Traffic deviation is calculated as:

```text
traffic_change_pct = ((observed_volume - expected_volume) / expected_volume) × 100
```

A positive deviation indicates traffic volume above the historical baseline, while a negative deviation indicates traffic volume below the historical baseline.

## Classification Labels

Each benchmark instance is presented as a three-class classification problem.

| Label | Classification |
|---|---|
| **A** | Normal |
| **B** | Minor Anomaly |
| **C** | Major Anomaly |

The model is instructed to return exactly one character: **A, B, or C**.

## Ground Truth Construction

Ground-truth labels are generated automatically using a rule-based framework that combines traffic deviation with contextual severity.

The context severity score is:

```text
S = rainfall severity + crash severity + event severity + wind severity + holiday severity
```

The components represent:

- Rainfall severity
- Crash severity
- Nearby event severity
- Strong-wind conditions
- Holiday conditions

Instead of using one fixed anomaly threshold, the classification thresholds are adjusted according to contextual severity.

| Context Severity | Minor Anomaly Threshold | Major Anomaly Threshold |
|---|---:|---:|
| `S = 0` | 15% | 30% |
| `1 <= S <= 2` | 22% | 40% |
| `3 <= S <= 4` | 30% | 55% |
| `S > 4` | 40% | 70% |

### Final Anomaly Label

The final class is assigned using the absolute traffic deviation and the thresholds corresponding to the observation's context severity.

| Condition | Classification |
|---|---|
| Absolute deviation >= major threshold | **Major Anomaly** |
| Absolute deviation >= minor threshold and < major threshold | **Minor Anomaly** |
| Absolute deviation < minor threshold | **Normal** |

For example, when context severity is `S = 2`, the minor threshold is **22%** and the major threshold is **40%**. A traffic deviation of **18%** is therefore Normal, **30%** is a Minor Anomaly, and **45%** is a Major Anomaly.

This approach allows the same traffic deviation to receive different anomaly classifications depending on the surrounding urban context. A large deviation during severe weather or a major event may be considered less anomalous than the same deviation under otherwise normal conditions.

## Benchmark Dataset

The final Task 3 evaluation benchmark contains **600 instances**, balanced across the three target classes.

| Class | Number of Instances |
|---|---:|
| Normal | 200 |
| Minor Anomaly | 200 |
| Major Anomaly | 200 |
| **Total** | **600** |

## Evaluated Models

The following eight LLMs are included in the final Task 3 evaluation:

1. **Llama 3.3 70B**
2. **Gemini 2.5 Flash**
3. **DeepSeek R1 Distill 70B**
4. **Gemma 2 27B**
5. **Qwen3 32B**
6. **Mixtral 8×22B Instruct**
7. **Gemini 2.5 Pro**
8. **Llama 3.1 8B**

## Evaluation Protocol

All models are evaluated using the same benchmark instances under a **zero-shot, single-turn classification setting**.

For every benchmark instance, the model receives the traffic scenario and is instructed to return exactly one classification label:

```text
A
B
C
```

No explanation or reasoning is requested as part of the final response. Invalid or empty outputs are recorded separately from valid predictions.

## Evaluation Metrics

The evaluation reports:

- Accuracy
- Balanced Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Weighted Precision, Recall, and F1
- Cohen's Kappa
- Per-class Precision, Recall, F1, and Support
- Confusion Matrix
- Invalid/error rate
- Average response time

## Model Aliases

The evaluation script uses these eight aliases:

```text
llama_3_3_70b
gemini_2_5_flash
deepseek_r1_distill
gemma_2_27b
qwen3_32b
mixtral_8x22b
gemini_2_5_pro
llama_3_1_8b
```

## Running the Evaluation

Display the configured models:

```bash
python llm_eval_task3.py --list
```

Evaluate one model:

```bash
python llm_eval_task3.py --model llama_3_3_70b
```

Run a small balanced test:

```bash
python llm_eval_task3.py --model llama_3_3_70b --limit 6 --sample-mode balanced
```

Resume an interrupted evaluation:

```bash
python llm_eval_task3.py --model llama_3_3_70b --resume
```

Restart an evaluation and replace previously saved results:

```bash
python llm_eval_task3.py --model llama_3_3_70b --overwrite
```

## Output Files

The evaluation pipeline produces:

```text
task3_anomaly_classification.csv
llm_predictions_task3_anomaly.csv
llm_model_metrics_task3_anomaly.csv
llm_per_class_metrics_task3_anomaly.csv
confusion_matrix_task3_anomaly_<model_alias>.csv
```

### Predictions

`llm_predictions_task3_anomaly.csv` stores each model prediction together with the expected answer, predicted class, correctness, response time, raw model output, and recorded errors.

### Model Metrics

`llm_model_metrics_task3_anomaly.csv` contains aggregate evaluation metrics for each model.

### Per-Class Metrics

`llm_per_class_metrics_task3_anomaly.csv` contains precision, recall, F1 score, and support for each anomaly class.

### Confusion Matrices

A separate confusion matrix is generated for every model with valid predictions.

## Evaluation Workflow

1. Load the Task 3 benchmark dataset.
2. Validate the benchmark labels.
3. Select one of the eight configured LLMs.
4. Construct the classification prompt for each benchmark instance.
5. Send the prompt to the selected model.
6. Parse the response into A, B, or C.
7. Compare the prediction with the ground-truth label.
8. Save predictions periodically using checkpoints.
9. Calculate aggregate and per-class evaluation metrics.
10. Generate the model confusion matrix.
11. Compare performance across the eight evaluated models.

## Urban Context Benchmark

Task 3 forms part of the broader **Urban Context Benchmark**, which evaluates the ability of LLMs to understand and reason about urban mobility using real-world contextual information.

The complete benchmark contains five complementary tasks:

1. Context Understanding
2. Contextual Traffic Prediction
3. Anomaly Classification
4. Contextual Reasoning
5. Region Sensitivity

Task 3 specifically focuses on whether an LLM can distinguish expected traffic variation from genuine anomalies after accounting for the surrounding urban context.
