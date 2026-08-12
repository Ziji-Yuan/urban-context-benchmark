# Task 2: Single-Scenario Traffic Volume Prediction

## Objective

Task 2 evaluates whether a language model can predict traffic volume change direction given a single urban-context scenario. The model selects whether the actual traffic volume at a given station and hour is **Lower than typical (A)**, **Close to typical (B)**, or **Higher than typical (C)**.

The task uses a **five-level progressive information ablation** (L0 → L4), where each level provides the model with more contextual fields, from minimal station/time data up to full urban context (weather, events, crashes, holidays, land use, and POI categories).

## Inputs

The QA builders consume a labelled benchmark-ready Master Table produced by the ground-truth pipeline. The table contains station-hour-level observations with traffic volume, expected volume, weather, events, crashes, public and school holidays, and urban-context descriptions. The large input CSV is **not included** in this repository.

```
labeled_data/
  master_table_station_hour_2022_2024_benchmark_labeled.csv   # labelled input (not in repo)
  task1_qa_pairs.jsonl                                        # generated QA pairs
  task1_qa_stats.json                                         # sampling statistics
```

## Ground Truth Labeling

Traffic volume labels are derived from a z-score relative to the station's historical hourly expected volume. With ±10% threshold:

```
delta = (actual_volume - expected_volume) / expected_volume

decrease / A:   delta < -0.10
normal / B:     -0.10 ≤ delta ≤ 0.10
increase / C:   delta > 0.10
```

The labelling script (`ground_label_construction/generate_labels.py`) computes z-scores per station-hour and assigns three-class labels. Labels are embedded in the Master Table CSVs consumed by QA generation.

## Information Levels (L0 → L4)

Each level adds more context fields. All levels include station identity, location, time, and expected volume.

| Level | Fields Added | Count |
|-------|-------------|-------|
| **L0** | station_name, road_type, lga, date, day_of_week, hour, expected_volume | 7 |
| **L1** | + weather: rain_description, temperature, humidity, visibility, wind, cloud | 13 |
| **L2** | + calendar: is_weekend, school_holiday, holiday_text | 14 |
| **L3** | + events: event_text | 15 |
| **L4** | + full context: crash_text, land_use_description, top_poi_categories | 18 |

## QA Generation

`qa_generation/generate_qa_pairs.py` produces 3,000 QA pairs from the labelled master table using a two-pass stratified sampling approach:

- **Pass 1** chunk-reads lightweight metadata columns to build a sampling frame
- **Pass 2** loads full context for selected rows and generates natural-language descriptions

Sampling constraints:
- **~1,000 per label** (decrease / normal / increase)
- **Balanced across years** (2022–2024, ±10% tolerance)
- **≥300 per condition** (rain, event, crash>1)
- **Fixed random seed** (42) for reproducibility

Natural-language description functions convert structured fields into readable text for weather, events, crashes, holidays, land use, and nearby POIs.

## Model Execution

`model_evaluation/benchmark_soft_prompt_openrouter_noCoT.py` is the primary benchmark runner. It uses a **soft (conversational) prompt** that allows models to briefly reason before answering — no few-shot examples, no forced completion like "Output exactly one letter."

### Setup

```bash
cd benchmarks/task2/model_evaluation
cp ../../.env.example .env
# Set OPENROUTER_API_KEY in .env (https://openrouter.ai/keys)
```

### CLI Usage

```bash
# Full benchmark (all 8 models, all 5 levels, 600 per level)
python benchmark_soft_prompt_openrouter_noCoT.py

# Single model, limited pairs
python benchmark_soft_prompt_openrouter_noCoT.py \
  --models deepseek/deepseek-r1-distill-llama-70b \
  --max 100 --seed 42

# Custom output directory
python benchmark_soft_prompt_openrouter_noCoT.py \
  --output-dir labeled_data/results_adjust
```

### Queue Runner (3-models-at-a-time)

```bash
python queue_runner.py    # launch all queued models
python queue_resume.py    # resume after partial completion
```

### Default Models

`deepseek/deepseek-r1-distill-qwen-32b`, `mistralai/mixtral-8x22b-instruct`, `google/gemma-2-27b-it`, `meta-llama/llama-3.3-70b-instruct`, `meta-llama/llama-3.1-8b-instruct`, `qwen/qwen3-32b`, `google/gemini-2.5-flash`, `google/gemini-2.5-pro`

### Resume & Incremental Runs

Each benchmark run tracks `levels_run` in the output JSON. Re-running with the same model and output directory skips completed levels. To run a fresh benchmark, delete the existing output file or change `--output-dir`.

## Metrics and Evaluation

All metrics are computed **inline** during benchmarking by `compute_metrics()` (line 330) and written to the output JSON. No separate evaluation script is needed.

### Primary Metric

**Strict accuracy** — the extracted option letter must exactly match the ground-truth option. Extraction failures (unparseable responses) count as incorrect.

### Reported Metrics

| Metric | Description |
|--------|-------------|
| `overall_accuracy` | Strict correct / total QA pairs |
| `per_label_accuracy` | Accuracy broken down by decrease / normal / increase |
| `per_condition_accuracy` | Accuracy for rain hours, event hours, crash>1 hours, school holidays |
| `confusion_matrix` | True label → extracted option (A/B/C) |
| `extraction_failures` | Responses where no A/B/C option could be parsed |
| `soft_scoring` | Same structure as above, but boundary-adjacent predictions (±1.0% band around ±10% threshold) are rescued |

### Soft Scoring

When enabled (default `--soft-band 1.0`), samples whose `traffic_change_pct` falls within ±1.0% of the ±10% decision boundary are treated leniently: predicting **B (normal)** is accepted even if the strict label is A or C. This avoids penalising models for reasonable predictions on borderline cases.

## API Configuration

- **Provider**: OpenRouter (`https://openrouter.ai/api/v1`)
- **Temperature**: 0.0 (deterministic)
- **Max output tokens**: 1024
- **Rate limit**: 0.5s inter-request delay, exponential backoff on 429s
- **Retries**: up to 5 per request

## Workflow

From the repository root:

```bash
# 1. Generate ground-truth labels (requires master table CSV)
python benchmarks/task2/ground_label_construction/generate_labels.py

# 2. Generate QA pairs (~3,000 stratified pairs)
python benchmarks/task2/qa_generation/generate_qa_pairs.py

# 3. Run benchmark (all models, all levels)
python benchmarks/task2/model_evaluation/benchmark_soft_prompt_openrouter_noCoT.py

# 4. Queue multiple models concurrently
python benchmarks/task2/model_evaluation/queue_runner.py

# 5. Generate example prompts for inspection
python benchmarks/task2/qa_generation/gen_sample_prompts.py
```

## Output Structure

```
model_evaluation/
  benchmark_soft_prompt_openrouter_noCoT.py   # primary benchmark runner
  queue_runner.py                             # concurrent queue launcher
  queue_resume.py                             # resume partial runs

qa_generation/
  generate_qa_pairs.py                        # two-pass stratified QA generation
  gen_sample_prompts.py                       # generate example prompts

examples/
  prompt_all_levels.md                        # one QA pair rendered at all 5 levels
  sample_l4_prompts.md                        # one L4 prompt per label
  fields_by_level.txt                         # field reference per level
  qa_pair_snippet.json                        # raw JSON of a single QA pair
```
