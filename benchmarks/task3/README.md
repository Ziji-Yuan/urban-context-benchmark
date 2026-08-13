# Task 3: Context-Aware Anomaly Classification

## Overview

Task 3 evaluates the ability of Large Language Models (LLMs) to identify and classify anomalous urban traffic conditions using contextual information.

The task is part of the **Urban Context Benchmark**, which investigates whether incorporating external urban context—such as weather, holidays, events, and disruptions—improves a model's ability to understand and reason about urban mobility patterns.

Rather than relying only on traffic measurements, Task 3 progressively provides additional contextual information to evaluate how different types of context affect anomaly classification performance.

## Objective

Given traffic observations and associated urban context, the model is asked to determine whether the observed traffic condition represents an anomaly and classify it into the appropriate anomaly category.

The benchmark is designed to evaluate:

* Anomaly recognition
* Context-aware classification
* Urban traffic reasoning
* The effect of increasing contextual information
* Performance differences across LLMs

## Context Levels

Task 3 uses incremental context levels from **L0 to L4**.

| Level  | Context                                                           |
| ------ | ----------------------------------------------------------------- |
| **L0** | Traffic information only                                          |
| **L1** | Traffic + temporal context                                        |
| **L2** | Traffic + temporal + weather context                              |
| **L3** | Traffic + temporal + weather + event/holiday context              |
| **L4** | Full contextual information, including relevant urban disruptions |

This incremental design allows us to measure whether additional contextual information improves anomaly classification.

## Dataset

The Task 3 dataset contains approximately **10,000 benchmark instances** derived from urban traffic observations and external contextual datasets.

Contextual information may include:

* Traffic conditions
* Date and time
* Weather conditions
* Public holidays
* School terms
* Major events
* Traffic incidents and disruptions
* Other relevant urban context

The benchmark uses adaptive criteria to identify anomalous observations while accounting for differences across locations and traffic patterns.

## Evaluation

Models are evaluated in a **zero-shot, single-turn setting**.

The primary evaluation metrics include:

* **Accuracy**
* **Macro F1-score**
* **Cohen's Kappa**

Macro F1 is particularly useful for evaluating performance across anomaly classes when the class distribution is imbalanced, while Cohen's Kappa measures agreement beyond what would be expected by chance.

## Models

Task 3 has been evaluated using multiple LLM families, including:

* Llama
* DeepSeek
* Qwen
* Mixtral
* Gemma
* Gemini

The benchmark enables comparison of model performance across different context levels and model architectures.

## Repository Structure

```text
benchmarks/
└── task3/
    ├── README.md
    ├── data/
    ├── prompts/
    ├── results/
    └── evaluation/
```

The exact files available in this directory may vary as the benchmark is updated.

## Running Task 3

The general workflow is:

1. Load the Task 3 benchmark dataset.
2. Construct the prompt using the required context level (L0–L4).
3. Send each benchmark instance to the selected model.
4. Record the model's predicted anomaly class.
5. Compare predictions against the ground-truth labels.
6. Calculate evaluation metrics.
7. Compare performance across models and context levels.

## Urban Context Benchmark

Task 3 is one component of the broader **Urban Context Benchmark**, which evaluates the ability of LLMs to understand and reason about urban mobility data under different contextual conditions.

The benchmark includes tasks covering context understanding, traffic prediction, anomaly classification, contextual reasoning, and downstream spatial analysis.
