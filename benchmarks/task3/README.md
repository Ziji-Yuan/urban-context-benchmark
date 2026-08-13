# Task 3: Context Aware Anomaly Classification

## Overview

Task 3 evaluates whether Large Language Models (LLMs) can determine whether an observed traffic deviation represents a normal condition, a minor anomaly, or a major anomaly after considering the surrounding urban context.

Unlike a simple traffic deviation classification task, the model must interpret the observed and expected traffic volumes together with contextual factors such as weather conditions, nearby events, road crashes, public holidays, school holidays, and the surrounding urban environment.

## Task Definition

Each benchmark instance is constructed from a single traffic observation extracted from the master context table.

Every observation is converted into a structured natural language description containing information such as:

* Traffic monitoring station and road location
* Date and time
* Observed traffic volume
* Expected traffic volume
* Weather conditions
* Nearby road crashes
* Nearby events
* Public holiday status
* School holiday status
* Surrounding urban characteristics

Traffic deviation is calculated as the percentage difference between observed and expected traffic volume:

[
\Delta_i = \frac{V_i^{obs} - V_i^{exp}}{V_i^{exp}} \times 100
]

where (V_i^{obs}) represents the observed traffic volume and (V_i^{exp}) represents the expected traffic volume.

A positive deviation indicates traffic volume above the historical baseline, while a negative deviation indicates traffic volume below the historical baseline.

## Classification Labels

Each benchmark instance is presented as a three class classification problem.

| Label | Classification |
| ----- | -------------- |
| **A** | Normal         |
| **B** | Minor Anomaly  |
| **C** | Major Anomaly  |

The model is instructed to return exactly one character: **A, B, or C**.

## Ground Truth Construction

Ground truth labels are generated automatically using a rule based framework that combines traffic deviation with contextual severity.

The context severity score is calculated as:

[
S_i = R_i + C_i + E_i + W_i + H_i
]

where:

* (R_i) represents rainfall severity
* (C_i) represents crash severity
* (E_i) represents nearby event severity
* (W_i) represents strong wind conditions
* (H_i) represents holiday conditions

Instead of using one fixed anomaly threshold, the classification thresholds are adjusted according to contextual severity.

| Context Severity    | Minor Anomaly Threshold | Major Anomaly Threshold |
| ------------------- | ----------------------: | ----------------------: |
| (S_i = 0)           |                     15% |                     30% |
| (1 \leq S_i \leq 2) |                     22% |                     40% |
| (3 \leq S_i \leq 4) |                     30% |                     55% |
| (S_i > 4)           |                     40% |                     70% |

The final anomaly label is determined using:

[
y_i =
\begin{cases}
\text{Major Anomaly}, & |\Delta_i| \geq T_M(S_i) \
\text{Minor Anomaly}, & T_m(S_i) \leq |\Delta_i| < T_M(S_i) \
\text{Normal}, & |\Delta_i| < T_m(S_i)
\end{cases}
]

This approach allows the same traffic deviation to receive different anomaly classifications depending on the surrounding urban context.

For example, a large traffic deviation occurring during severe weather or a major event may be considered less anomalous than the same deviation occurring under otherwise normal conditions.

## Benchmark Dataset

The final Task 3 evaluation benchmark contains **600 instances**.

The benchmark is balanced across the three target classes:

| Class         | Number of Instances |
| ------------- | ------------------: |
| Normal        |                 200 |
| Minor Anomaly |                 200 |
| Major Anomaly |                 200 |
| **Total**     |             **600** |

The balanced class distribution ensures that the evaluation is not dominated by one anomaly category.

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

The models represent different developers, architectures, parameter scales, and training approaches.

## Evaluation Protocol

All models are evaluated using the same benchmark instances under a **zero shot, single turn classification setting**.

For every benchmark instance, the model receives the traffic scenario and is instructed to return exactly one classification label:

```text
A
B
C
```

No explanation or reasoning is requested as part of the final response.

The evaluation script parses the model response and records invalid or empty outputs separately from valid predictions.

## Evaluation Metrics

Model performance is evaluated using several complementary metrics.

### Accuracy

Accuracy measures the proportion of correctly classified benchmark instances.

### Balanced Accuracy

Balanced accuracy calculates the average recall across the three target classes.

Because the final benchmark contains 200 observations from each class, accuracy and balanced accuracy are directly comparable.

### Macro Precision

Macro Precision calculates precision independently for each class and then gives each class equal weight.

### Macro Recall

Macro Recall calculates recall independently for each class and averages the results across all three classes.

### Macro F1 Score

Macro F1 calculates the F1 score independently for Normal, Minor Anomaly, and Major Anomaly before averaging the three scores.

This metric is particularly useful for determining whether a model performs consistently across all anomaly categories.

### Weighted Metrics

Weighted precision, recall, and F1 score are also calculated using class support.

### Cohen's Kappa

Cohen's Kappa measures agreement between model predictions and ground truth labels while accounting for agreement that could occur by chance.

### Per Class Metrics

Precision, recall, F1 score, and support are calculated separately for:

* Normal
* Minor Anomaly
* Major Anomaly

A confusion matrix is also generated for each evaluated model.

## Evaluation Script

The evaluation is implemented in:

```text
llm_eval_task3.py
```

The script supports:

* Evaluation of one model at a time
* Balanced sampling for development and testing
* Random sampling
* Limited test runs
* API model availability checks
* Automatic retry handling
* Rate limit handling
* Periodic checkpoints
* Resuming interrupted evaluations
* Overwriting previous model results
* Model response validation
* Automatic metric calculation
* Per class metric generation
* Confusion matrix generation

## Model Aliases

The following aliases are used by the evaluation script:

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

To display the configured models:

```bash
python llm_eval_task3.py --list
```

To evaluate one model:

```bash
python llm_eval_task3.py --model llama_3_3_70b
```

To perform a small balanced test:

```bash
python llm_eval_task3.py --model llama_3_3_70b --limit 6 --sample-mode balanced
```

To resume an interrupted evaluation:

```bash
python llm_eval_task3.py --model llama_3_3_70b --resume
```

To restart an evaluation and replace the previously saved results:

```bash
python llm_eval_task3.py --model llama_3_3_70b --overwrite
```

## Output Files

The evaluation script produces the following files:

```text
task3_anomaly_classification.csv
llm_predictions_task3_anomaly.csv
llm_model_metrics_task3_anomaly.csv
llm_per_class_metrics_task3_anomaly.csv
confusion_matrix_task3_anomaly_<model_alias>.csv
```

### Predictions

`llm_predictions_task3_anomaly.csv` stores the predictions generated by each model together with the expected answer, predicted class, correctness, response time, raw model output, and any recorded errors.

### Model Metrics

`llm_model_metrics_task3_anomaly.csv` contains the aggregate evaluation metrics for each model.

### Per Class Metrics

`llm_per_class_metrics_task3_anomaly.csv` contains precision, recall, F1 score, and support for each anomaly class.

### Confusion Matrices

A separate confusion matrix is generated for every model with valid predictions.

## Evaluation Workflow

1. Load the Task 3 benchmark dataset.
2. Validate the benchmark labels.
3. Select one of the eight configured LLMs.
4. Construct the classification prompt for each benchmark instance.
5. Send the prompt to the selected model.
6. Parse the model response into A, B, or C.
7. Compare the prediction with the ground truth label.
8. Save predictions periodically using checkpoints.
9. Calculate aggregate and per class evaluation metrics.
10. Generate the model confusion matrix.
11. Compare performance across the eight evaluated models.

## Urban Context Benchmark

Task 3 forms part of the broader **Urban Context Benchmark**, which evaluates the ability of LLMs to understand and reason about urban mobility using real world contextual information.

The complete benchmark contains five complementary tasks:

1. Context Understanding
2. Contextual Traffic Prediction
3. Anomaly Classification
4. Contextual Reasoning
5. Region Sensitivity

Task 3 specifically focuses on whether an LLM can distinguish expected traffic variation from genuine anomalies after accounting for the surrounding urban context.
