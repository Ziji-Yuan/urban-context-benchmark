# Task 5: Region Sensitivity

## Objective

Task 5 evaluates whether large language models can identify regions whose traffic is more sensitive to adverse weather conditions using regional traffic and contextual information.

The benchmark is constructed at the Local Government Area (LGA) level. Regional traffic behaviour under different weather conditions is summarised and used to generate multiple-choice question-answer pairs for model evaluation.

## Benchmark Tasks

The benchmark contains three question types:

1. **Pairwise Comparison**  
   Compares two candidate LGAs and asks which region shows greater weather-sensitive traffic behaviour.

2. **Top Sensitive Region**  
   Presents four candidate LGAs and asks the model to identify the region with the strongest weather-sensitive traffic response.

3. **Management Priority**  
   Presents four candidate LGAs and asks which region should receive the highest management priority based on regional sensitivity and supporting traffic evidence.

The final evaluation dataset contains **1,800 QA pairs**, with **600 questions for each task type**.

## Directory Structure

```text
task5/
├── README.md
├── qa_generation/
│   ├── 01_region_sensitivity_table_generation.ipynb
│   └── 02_region_sensitivity_qa_generation.ipynb
├── data/
│   ├── region_sensitivity_table.csv
│   └── region_sensitivity_qa_pairs_context_v2_1800_eval.json
└── model_evaluation/
    ├── gemini2_5_flash_evaluation.ipynb
    ├── gemini2_5_pro_evaluation.ipynb
    ├── llama3_1_8b_openrouter_evaluation.ipynb
    ├── llama3_3_70b_openrouter_evaluation.ipynb
    ├── qwen3_32b_openrouter_evaluation.ipynb
    ├── mixtral_8x22b_openrouter_evaluation.ipynb
    ├── gemma2_27b_openrouter_evaluation.ipynb
    └── deepseek_r1_distill_llama_70b_openrouter_evaluation.ipynb
```

## QA Generation

### 1. Region Sensitivity Table

`qa_generation/01_region_sensitivity_table_generation.ipynb`

This notebook constructs the LGA-level region sensitivity table from the benchmark master data. It aggregates traffic behaviour under different weather conditions and produces:

`data/region_sensitivity_table.csv`

### 2. QA Generation and Ground Truth

`qa_generation/02_region_sensitivity_qa_generation.ipynb`

This notebook generates the Task 5 multiple-choice QA benchmark from the region sensitivity table. It constructs the three Task 5 question types and their corresponding ground-truth answers.

The final evaluation dataset is:

`data/region_sensitivity_qa_pairs_context_v2_1800_eval.json`

Each QA instance contains the task type, system prompt, question, candidate options, ground-truth answer, ground-truth region, weather condition, and candidate LGAs.

## Model Evaluation

The `model_evaluation/` directory contains the notebooks used to evaluate the eight language models included in Task 5:

- Gemini 2.5 Flash
- Gemini 2.5 Pro
- Llama 3.1 8B
- Llama 3.3 70B
- Qwen3 32B
- Mixtral 8x22B
- Gemma 2 27B
- DeepSeek-R1-Distill-Llama-70B

Gemini models are evaluated through the Gemini API, while the remaining models are evaluated through OpenRouter.

Each notebook contains the model-specific evaluation pipeline, answer extraction, result generation, and metric verification.

## Evaluation Metrics

Model performance is evaluated using:

- Accuracy
- Macro-F1
- Task-level Accuracy
- Task-level Macro-F1
- Option-level Recall and F1
- Prediction distribution

Metrics are calculated from model predictions against the ground-truth answers contained in the Task 5 QA dataset.

## Reproduction

The recommended execution order is:

1. Run `qa_generation/01_region_sensitivity_table_generation.ipynb`.
2. Run `qa_generation/02_region_sensitivity_qa_generation.ipynb`.
3. Run the required notebook in `model_evaluation/`.
4. Use the metric verification section at the end of each model notebook to reproduce the reported evaluation metrics.

The first QA-generation notebook requires the shared benchmark master table as input. The master table is a common project-level resource and is therefore not duplicated inside the Task 5 folder.

API credentials are required for model evaluation and are not included in this repository.
