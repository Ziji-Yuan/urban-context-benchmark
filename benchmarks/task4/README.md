# Task 4: Contextual and Contrastive Reasoning

## Objective

Task 4 evaluates whether a language model can reason about traffic changes between two comparable urban scenarios. Given Scenario A and Scenario B, the model selects whether traffic volume in Scenario B is **Higher**, **Lower**, or **Similar** relative to Scenario A.

The implementation covers three context types: rainfall sensitivity, crash-disruption sensitivity, and planned-activity/calendar context.

## Inputs

The QA builders consume a labelled benchmark-ready Master Table. Depending on the context type, the builders use station and time keys, traffic volume, rainfall, temperature, events, crashes, public and school holidays, and urban-context descriptions. The large input table is not included in the repository.

## Scenario Pair Construction

All three builders match observations at the same station and hour while controlling available contextual factors.

- **Rainfall:** both pools exclude nearby events, holidays, and major crash contexts. Matching prioritises the same station, hour, day type, month or season, and temperature bucket. Pairs from the same timestamp, the same source row, or effectively identical rainfall conditions are rejected.
- **Crash disruption:** Scenario A has no crash and Scenario B has one or more crashes. Events and holidays are excluded. Matching controls station, hour, day type, rainfall bucket, temperature bucket, and month or season.
- **Planned activity/calendar:** pairs differ in event or holiday context. Matching controls station, hour, day type, rainfall bucket, temperature bucket, crash bucket, and month or season. A pair is retained only when at least one activity/calendar attribute changes.

Candidate pools are sampled with fixed random seeds. The final evaluation sets are balanced across the three labels when sufficient candidates are available.

## Labels

For a pair with observed volumes `volume_A` and `volume_B`, the reference change is:

```text
delta = (volume_B - volume_A) / volume_A
```

- **Higher / A:** `delta > 0.10`
- **Lower / B:** `delta < -0.10`
- **Similar / C:** `-0.10 ≤ delta ≤ 0.10`

The source scripts use the semantic aliases `increase`, `decrease`, and `normal`; evaluation standardises them to `higher`, `lower`, and `similar`.

## QA Generation

Each builder writes scenario-pair records and model-ready questions. Questions contain station and urban context, the controlled Scenario A/B attributes, and the three answer choices. Three small representative records are included in [`examples/sample_questions.jsonl`](examples/sample_questions.jsonl).

## Model Execution

Model definitions are stored in `model_evaluation/config_models.py`. Credentials are read from environment variables; secrets must never be placed in source code.

```bash
cd benchmarks/task4/model_evaluation
cp .env.example .env
# Set OPENROUTER_API_KEY in .env
python run_api_eval.py --input ../qa_generation/generated/rainfall/rain_numeric_context_qa_eval600.csv --models all --dry-run
python run_api_eval.py --input ../qa_generation/generated/rainfall/rain_numeric_context_qa_eval600.csv --models all --resume
```

The runner supports dry-run prompt inspection, per-model resumable predictions, response parsing, and report generation. Generated predictions and reports are ignored by Git.

## Metrics and Evaluation

`evaluation/analyze_task4_metrics.py` reads CSV, JSON, or JSONL prediction files from an external Task 4 run directory. It standardises common field names and labels, recomputes correctness from predicted and gold options, retains invalid predictions, and reports:

- strict and valid-only accuracy;
- pooled Macro-F1;
- class precision, recall, F1, and support;
- performance by context type;
- prediction distributions and confusion matrices;
- optional human-evaluated contextual-reasoning scores.

Strict accuracy, in which an invalid prediction is counted as incorrect, is the primary metric.

## Workflow

From the repository root:

```bash
# Rainfall builder (portable CLI arguments)
python benchmarks/task4/qa_generation/build_rainfall_qa.py \
  --input_csv path/to/master_table_station_hour_2022_2024_benchmark_labeled.csv

# Crash and activity builders use the same configurable input variable
export TASK4_INPUT_CSV=path/to/master_table_station_hour_2022_2024_benchmark_labeled.csv
python benchmarks/task4/qa_generation/build_crash_disruption_qa.py
python benchmarks/task4/qa_generation/build_planned_activity_calendar_qa.py

# Model inference: pass the generated evaluation CSV explicitly
python benchmarks/task4/model_evaluation/run_api_eval.py --input path/to/qa_eval600.csv --models all --dry-run

# Unified evaluation: point --root at the external run directory
python benchmarks/task4/evaluation/analyze_task4_metrics.py --root path/to/task4_run
```

`TASK4_OUTPUT_DIR` can be set for the crash and activity builders. Rainfall output can be configured with `--output_dir`.
