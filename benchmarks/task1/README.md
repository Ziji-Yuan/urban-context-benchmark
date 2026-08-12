# Task 1: Context Understanding

Tests whether an LLM can judge, from context alone (time, place, weather,
events, no historical traffic number given), whether a station-hour counts
as "high traffic" or "normal traffic".

## Files
- `task1_benchmark_evaluation.py` - sends the QA pairs to 8 models via OpenRouter and scores the results.
- `task1_qa_pairs.jsonl` - the 600 QA pairs actually used for evaluation (300 high_traffic + 300 normal_traffic, class-balanced).
- `eval_results/` - one JSONL file per model with every raw response (question, predicted answer, ground truth, correctness, response time).

## Workflow

### Run the evaluation

```
pip install requests
export OPENROUTER_API_KEY="sk-or-v1-..."
python3 task1_benchmark_evaluation.py
```
Things to be noted:
- Needs an OpenRouter account with credit (https://openrouter.ai) and the API Key is unique to every user. 
- Writes results to `eval_results/`. 
- The script checkpoints as it goes so if it's interrupted, it will skips questions that already answered instead of starting over when rerunning.

## Results Overview

| Model | Completion | Accuracy | Macro-F1 | Avg. Time (s) |
|---|---|---|---|---|
| Gemini 2.5 Pro | 600/600 | 70.8% | 0.708 | 3.592 |
| Gemini 2.5 Flash | 600/600 | 67.7% | 0.676 | 2.891 |
| DeepSeek-R1-Distill | 600/600 | 62.8% | 0.625 | 65.641 |
| Llama 3.3 70B | 600/600 | 56.5% | 0.560 | 13.674 |
| Qwen3 32B | 600/600 | 55.2% | 0.511 | 20.848 |
| Mixtral 8x22B | 600/600 | 53.2% | 0.443 | 0.829 |
| Gemma 2 27B | 600/600 | 50.0% | 0.333 | 3.991 |
| Llama 3.1 8B | 600/600 | 48.5% | 0.443 | 4.731 |