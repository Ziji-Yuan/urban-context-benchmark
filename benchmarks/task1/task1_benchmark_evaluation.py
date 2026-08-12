import json
import os
import re
import time

import requests

# Benchmark Evaluation for Task 1 (Context Understanding)
# Sends each QA pair to every model via OpenRouter, checks the answer, and writes per-model results.

INPUT_JSONL = "task1_qa_pairs.jsonl"
RESULTS_DIR = "eval_results"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SAMPLE_SIZE = None  # Initially set as None because 600 balanced pairs already in the file

MODELS = {
    "Gemini 2.5 Flash": "google/gemini-2.5-flash",
    "Gemini 2.5 Pro": "google/gemini-2.5-pro",
    "Gemma 2 27B": "google/gemma-2-27b-it",
    "Llama 3.1 8B": "meta-llama/llama-3.1-8b-instruct",
    "Llama 3.3 70B": "meta-llama/llama-3.3-70b-instruct",
    "Mixtral 8x22B": "mistralai/mixtral-8x22b-instruct",
    "Qwen3 32B": "qwen/qwen3-32b",
    "DeepSeek-R1-Distill": "deepseek/deepseek-r1-distill-llama-70b",
}

REQUEST_TIMEOUT = 60
RETRIES = 3
RETRY_BACKOFF = 5


# Loading QA Pairs
def load_qa_pairs(path, sample_size=None):
    pairs = []
    with open(path) as f:
        for line in f:
            pairs.append(json.loads(line))
    if sample_size:
        pairs = pairs[:sample_size]
    return pairs


# Calling the Model via OpenRouter
def call_model(model_slug, question_text):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_slug,
        "messages": [{"role": "user", "content": question_text}],
        "temperature": 0,
        "max_tokens": 3000,  # reasoning models spend tokens thinking before they answer
        "reasoning": {"effort": "low"},
    }

    last_error = None
    attempt = 1
    while attempt <= RETRIES:
        try:
            resp = requests.post(headers=headers, url=OPENROUTER_URL, json=payload, timeout=REQUEST_TIMEOUT)
            if not resp.ok:
                raise RuntimeError(f"{resp.status_code} {resp.reason}: {resp.text[:300]}")
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_error = e
            if attempt < RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
        attempt += 1

    raise RuntimeError(f"Failed after {RETRIES} attempts: {last_error}")


# Extracting the Answer Letter A/B in the reply (models tend to give explanations first before giving the A/B)
def extract_answer_letter(raw_reply):
    if not raw_reply:
        return None
    matches = re.findall(r"\b([AB])\b", raw_reply.strip().upper())
    if not matches:
        return None
    return matches[-1]


# Loading Existing Results (so a run can resume)
def load_latest_results(out_path):
    latest = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                latest[r["qa_id"]] = r
    return latest


# Running Evaluation for One Model
def run_model_evaluation(model_name, model_slug, qa_pairs):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, model_name.replace(" ", "_") + ".jsonl")

    latest = load_latest_results(out_path)
    already_done = set()
    for qa_id, r in latest.items():
        if r.get("predicted") is not None:
            already_done.add(qa_id)

    print(f"\n=== {model_name} ===")
    print(f"{len(already_done)}/{len(qa_pairs)} already done")

    out_f = open(out_path, "a")
    for i, qa in enumerate(qa_pairs):
        if qa["qa_id"] in already_done:
            continue

        start = time.time()
        try:
            raw_reply = call_model(model_slug, qa["question"])
            predicted = extract_answer_letter(raw_reply)
            correct = predicted == qa["answer"]
        except Exception as e:
            raw_reply = None
            predicted = None
            correct = False
            print(f"  [{i}] ERROR on {qa['qa_id']}: {e}")
        elapsed_s = round(time.time() - start, 3)

        result = {
            "qa_id": qa["qa_id"],
            "ground_truth": qa["answer"],
            "predicted": predicted,
            "correct": correct,
            "raw_reply": raw_reply,
            "elapsed_s": elapsed_s,
            "meta": qa["meta"],
        }
        out_f.write(json.dumps(result) + "\n")
        out_f.flush()

        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(qa_pairs)}")

    out_f.close()
    return out_path


# Computing Macro-F1 for the A/B classes
def macro_f1_binary(results):
    f1_scores = []
    for cls in ("A", "B"):
        tp = 0
        fp = 0
        fn = 0
        for r in results:
            if r["predicted"] == cls and r["ground_truth"] == cls:
                tp += 1
            elif r["predicted"] == cls and r["ground_truth"] != cls:
                fp += 1
            elif r["predicted"] != cls and r["ground_truth"] == cls:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)


# Summarizing Results Across Models
def summarize_results(valid_qa_ids=None):
    summary_rows = []

    for model_name in MODELS:
        out_path = os.path.join(RESULTS_DIR, model_name.replace(" ", "_") + ".jsonl")
        if not os.path.exists(out_path):
            continue

        results = list(load_latest_results(out_path).values())
        if valid_qa_ids is not None:
            filtered = []
            for r in results:
                if r["qa_id"] in valid_qa_ids:
                    filtered.append(r)
            results = filtered

        n = len(results)
        if n == 0:
            continue

        n_completed = 0
        n_correct = 0
        times = []
        for r in results:
            if r["predicted"] is not None:
                n_completed += 1
            if r["correct"]:
                n_correct += 1
            if r.get("elapsed_s") is not None:
                times.append(r["elapsed_s"])

        row = {
            "model": model_name,
            "n": n,
            "completion": f"{n_completed}/{n}",
            "accuracy": round(n_correct / n, 4),
            "macro_f1": round(macro_f1_binary(results), 4),
            "avg_time_s": round(sum(times) / len(times), 3) if times else "n/a",
        }

        # accuracy broken down by context condition
        for cond in ["has_rain", "has_event", "has_crash", "is_holiday"]:
            subset = []
            for r in results:
                if r["meta"].get(cond):
                    subset.append(r)
            if subset:
                n_correct_subset = 0
                for r in subset:
                    if r["correct"]:
                        n_correct_subset += 1
                row[f"accuracy_{cond}"] = round(n_correct_subset / len(subset), 4)

        summary_rows.append(row)

    if not summary_rows:
        print("No results to summarize yet.")
        return

    print("\n=== Summary ===")
    for row in summary_rows:
        print(row["model"], "\n", 
              "-> Completion:", row["completion"], "|", "Accuracy:", row["accuracy"], "|", "Macro F1:", row["macro_f1"], "|", "Average Time (s):", row["avg_time_s"])


def main():
    if not OPENROUTER_API_KEY:
        raise SystemExit("OPENROUTER_API_KEY is not set.")

    qa_pairs = load_qa_pairs(INPUT_JSONL, sample_size=SAMPLE_SIZE)
    print(f"Loaded {len(qa_pairs)} QA pairs")

    for model_name, model_slug in MODELS.items():
        run_model_evaluation(model_name, model_slug, qa_pairs)

    valid_qa_ids = set()
    for qa in qa_pairs:
        valid_qa_ids.add(qa["qa_id"])
    summarize_results(valid_qa_ids=valid_qa_ids)


main()
