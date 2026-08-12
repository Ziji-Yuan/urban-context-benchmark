# Urban Context Benchmark

## Project Overview

The Urban Context Benchmark is an academic benchmark for evaluating language-model reasoning over integrated urban traffic observations. It links traffic measurements with weather, planned activities, crashes, urban form, and calendar context so that models can be tested on questions that require more than isolated text or tabular lookup.

## Overall Benchmark Workflow

```text
Urban context data sources
        ↓
Spatio-temporal data integration
        ↓
Station-hour Master Table
        ↓
Ground-truth and label construction
        ↓
Individual benchmark tasks
        ↓
QA generation and model inference
        ↓
Task-specific evaluation
```

The Master Table provides a common station-hour representation. Ground-truth and task builders transform those integrated observations into benchmark-specific labels and questions, after which model responses are parsed and evaluated within each task.

## Data Context

The benchmark integrates six main forms of context:

- hourly traffic volume and monitoring-station metadata;
- weather conditions, including precipitation and temperature;
- nearby planned activities and calendar effects;
- crash and disruption information;
- static urban context such as land use, points of interest, and buildings;
- public-holiday, school-holiday, weekday, weekend, and hourly context.

Large source datasets and generated benchmark outputs are intentionally not stored in this repository.

## Data Preparation

The benchmark combines traffic and crash records, weather conditions, event information, and station-level urban-context features. Dataset-specific collection, processing, and alignment are documented in [`data_pre/README.md`](data_pre/README.md).

## Benchmark Structure

The benchmark is organised into Task 1, Task 2, Task 3, and Task 4 modules. Each task has its own construction and evaluation workflow. Task 4 is the contextual/contrastive reasoning benchmark: it presents two controlled traffic scenarios and asks whether traffic volume in Scenario B is **Higher**, **Lower**, or **Similar** to Scenario A.

## Repository Structure

```text
urban-context-benchmark/
├── master_table/          # Master Table integration methodology
├── ground_truth/          # Ground-truth construction module
├── benchmarks/
│   ├── task1/
│   ├── task2/
│   ├── task3/
│   └── task4/             # Contextual/contrastive reasoning implementation
│   └── task5/
├── requirements.txt
└── README.md
```

## Reproducibility

1. Prepare source data using consistent station identifiers, local dates, and hours.
2. Integrate traffic, spatial, weather, crash, event, and calendar context into the Master Table.
3. Apply the ground-truth construction required by each benchmark task.
4. Run the relevant task builder to create benchmark questions.
5. Configure model credentials through environment variables and run inference.
6. Parse responses and calculate task-specific metrics.

