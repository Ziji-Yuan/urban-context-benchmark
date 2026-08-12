#!/usr/bin/env python3
"""
benchmark_soft_prompt.py — Task 1 Benchmark Runner (Gentle Prompt, No CoT).

For models that refuse the aggressive "Output exactly one letter" directive
(distill-qwen-32b, mixtral-8x22b,"qwen3-32b",). Uses a natural conversational prompt that
allows brief reasoning, then extracts the answer from the response.

Usage:
    python benchmark_soft_prompt.py --max 100 --seed 42
    python benchmark_soft_prompt.py --models deepseek/deepseek-r1-distill-qwen-32b

API key: OPENROUTER_API_KEY  (https://openrouter.ai/keys)
"""

import json
import os
import random
import re
import sys
import time
import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from openai import OpenAI

# ============================================================================
# Auto-load .env file
# ============================================================================

def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and value and key not in os.environ:
                os.environ[key] = value

_load_dotenv()

# ============================================================================
# Paths & Constants
# ============================================================================

ROOT = Path(__file__).resolve().parent.parent
INPUT_JSONL = ROOT / "labeled_data" / "task1_qa_pairs.jsonl"
OUTPUT_RESULTS_DIR = ROOT / "labeled_data" / "results_no_cot"

DEFAULT_MODELS = [
    "deepseek/deepseek-r1-distill-qwen-32b",
    "mistralai/mixtral-8x22b-instruct",
    "google/gemma-2-27b-it", 
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "qwen/qwen3-32b",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
]
ALL_LEVELS = ["L0", "L1", "L2", "L3", "L4"]

# API settings
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
RATE_LIMIT_DELAY = 0.5
MAX_RETRIES = 5
RETRY_DELAY = 2.0
API_TEMPERATURE = 0.0
API_MAX_OUTPUT_TOKENS = 1024  # Allow brief reasoning before answer

# Soft scoring
DEFAULT_SOFT_BOUNDARY_BAND = 1.0

_client: OpenAI | None = None
_last_call_time: float = 0.0


def init_client() -> str:
    """Initialize OpenRouter client."""
    global _client
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY environment variable.")
        print("  Free key: https://openrouter.ai/keys")
        sys.exit(1)
    _client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/data5925/benchmark",
            "X-Title": "NSW Traffic Benchmark (Soft Prompt)",
        },
    )
    return api_key


# Gentle system prompt — allows reasoning, just requests a final answer
SYSTEM_PROMPT = (
    "You are a traffic analyst. Given the context, briefly reason about "
    "whether traffic volume is likely lower, close to, or higher than typical. "
    "End your response with exactly one of these on its own line: (A), (B), or (C)."
)


def call_llm(prompt: str, model: str) -> tuple[str | None, dict]:
    """Call model via OpenRouter API (OpenAI-compatible)."""
    global _client, _last_call_time

    for attempt in range(MAX_RETRIES):
        elapsed = time.time() - _last_call_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)

        try:
            response = _client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=API_TEMPERATURE,
                max_tokens=API_MAX_OUTPUT_TOKENS,
                timeout=120,
            )
            _last_call_time = time.time()
            text = (response.choices[0].message.content or "") if response.choices else ""
            # Gemini on OpenRouter: reasoning may consume tokens, fallback to reasoning text
            if not text.strip() and response.choices:
                text = getattr(response.choices[0].message, "reasoning", "") or ""
            if not text.strip():
                time.sleep(RETRY_DELAY * (2 ** attempt))
                continue
            option = extract_option(text)
            if option is None:
                # Got text but couldn't parse — retry
                time.sleep(RETRY_DELAY * (2 ** attempt))
                continue
            return option, {"attempt": attempt + 1, "raw_text": text[:300]}

        except Exception as e:
            _last_call_time = time.time()
            err_msg = str(e)
            if "429" in err_msg or "rate_limit" in err_msg.lower():
                wait = RATE_LIMIT_DELAY * (2 ** attempt)
            elif attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (2 ** attempt)
            else:
                return None, {"attempt": attempt + 1, "error": err_msg[:200]}
            time.sleep(wait)

    return None, {"attempt": MAX_RETRIES, "error": "max_retries_exceeded"}


# ============================================================================
# Prompt Rendering (gentle — no forced "Answer: (", no aggressive constraint)
# ============================================================================

OPTIONS_TEXT = """(A) Lower than typical
    (traffic volume more than 10% below the expected level for this station at this time)
(B) Close to typical
    (traffic volume within ±10% of the expected level — normal fluctuation)
(C) Higher than typical
    (traffic volume more than 10% above the expected level for this station at this time)"""

DIRECT_INSTRUCTION = (
    "Based on the information above, is the actual traffic volume most likely "
    "(A) lower than typical, (B) close to typical, or (C) higher than typical?\n\n"
    "Briefly explain your reasoning, then write your final answer as (A), (B), or (C) on a new line."
)

LEVEL_FIELD_MAP = {
    "L0": {"station_name", "road_type", "lga", "day_of_week", "date", "hour",
           "expected_volume"},
    "L1": {"station_name", "road_type", "lga", "day_of_week", "date", "hour",
           "expected_volume",
           "rain_description", "temperature_c", "apparent_temperature_c",
           "relative_humidity_2m_pct", "visibility", "windspeed_10m_kmh",
           "cloud_category"},
    "L2": {"station_name", "road_type", "lga", "day_of_week", "date", "hour",
           "expected_volume",
           "rain_description", "temperature_c", "relative_humidity_2m_pct",
           "visibility",
           "is_weekend", "school_holiday", "holiday_text"},
    "L3": {"station_name", "road_type", "lga", "day_of_week", "date", "hour",
           "expected_volume",
           "rain_description", "temperature_c", "relative_humidity_2m_pct",
           "visibility",
           "is_weekend", "school_holiday", "holiday_text",
           "event_text"},
    "L4": {"station_name", "road_type", "lga", "day_of_week", "date", "hour",
           "expected_volume",
           "rain_description", "temperature_c", "relative_humidity_2m_pct",
           "visibility",
           "is_weekend", "school_holiday", "holiday_text",
           "event_text",
           "crash_text", "land_use_description", "top_poi_categories"},
}


def render_prompt(sample_data: dict, level: str) -> str:
    """Render a Task 1 prompt (gentle — no CoT, no few-shot, no forced completion)."""
    d = sample_data
    fields = LEVEL_FIELD_MAP.get(level, LEVEL_FIELD_MAP["L4"])
    parts = []

    parts.append(
        f'A traffic monitoring station "{d["station_name"]}" on '
        f'{d["road_type"]} in {d["lga"]} area.'
    )

    if "land_use_description" in fields:
        lu = d.get("land_use_description", "")
        poi = d.get("top_poi_categories", "")
        if lu and poi:
            parts.append(
                f"The surrounding area is characterized by: {lu}, "
                f"with nearby facilities including {poi}."
            )
        elif lu:
            parts.append(f"The surrounding area is characterized by: {lu}.")

    parts.append(f'\nTime: {d["day_of_week"]}, {d["date"]} at {d["hour"]:02d}:00.')

    if "rain_description" in fields:
        rd = d.get("rain_description", "")
        tc = d.get("temperature_c", "")
        rh = d.get("relative_humidity_2m_pct", "")
        vis = d.get("visibility", "")
        parts.append(
            f"Weather: {rd}, temperature {tc:.0f}°C, "
            f"humidity {rh:.0f}%, visibility {vis}."
        )

    if "event_text" in fields:
        txt = d.get("event_text", "")
        if txt:
            parts.append(txt)

    if "crash_text" in fields:
        txt = d.get("crash_text", "")
        if txt:
            parts.append(txt)

    if "holiday_text" in fields:
        txt = d.get("holiday_text", "")
        if txt:
            parts.append(txt)

    parts.append(
        f'\nThe typical traffic volume for this station at this time is '
        f'approximately {d["expected_volume"]:.0f} vehicles per hour.'
    )

    parts.append(f"\n{OPTIONS_TEXT}")
    parts.append(f"\n{DIRECT_INSTRUCTION}")

    return "\n".join(parts)


# ============================================================================
# Answer Extraction — more robust for free-text responses
# ============================================================================

def extract_option(response_text: str) -> str | None:
    """Extract the answer option (A/B/C) from free-text model response.

    Priority: look for (A)/(B)/(C) on its own line near the end first,
    then fall back to broader patterns.
    """
    text = response_text.strip()

    # 1) Last line: "(A)", "(B)", or "(C)" alone
    lines = text.split("\n")
    for line in reversed(lines):
        m = re.match(r"^\s*\(([A-C])\s*\)\s*$", line)
        if m:
            return m.group(1).upper()

    # 2) "Answer: (A)" or "Final answer: (B)" pattern
    m = re.search(r"(?:final\s+)?answer\s*:?\s*\(([A-C])\)", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 3) Leading "A)" / "B)" / "C)" on any line
    m = re.search(r"^\s*([A-C])\s*\)", text, re.MULTILINE)
    if m:
        return m.group(1).upper()

    # 4) Last "(A)", "(B)", "(C)" in the text
    matches = list(re.finditer(r"\(\s*([A-C])\s*\)", text))
    if matches:
        return matches[-1].group(1).upper()

    # 5) "lower than typical" / "close to typical" / "higher than typical"
    if re.search(r"lower\s+than\s+typical", text, re.IGNORECASE):
        return "A"
    if re.search(r"higher\s+than\s+typical", text, re.IGNORECASE):
        return "C"
    if re.search(r"close\s+to\s+typical", text, re.IGNORECASE):
        return "B"

    # 6) Last standalone A/B/C letter
    matches = list(re.finditer(r"\b([A-C])\b", text))
    if matches:
        return matches[-1].group(1).upper()

    return None


# ============================================================================
# Metrics Computation (shared logic)
# ============================================================================

def is_boundary_sample(pct: float | None, band: float) -> bool:
    if pct is None:
        return False
    near_decrease = abs(pct - (-10.0)) < band
    near_increase = abs(pct - 10.0) < band
    return near_decrease or near_increase


def compute_metrics(results: list[dict], soft_boundary_band: float = 0.0) -> dict:
    valid = [r for r in results if r["extracted_option"] is not None]
    n_total = len(results)
    n_valid = len(valid)

    strict_correct = [r for r in valid if r["extracted_option"] == r["answer_option"]]
    n_strict = len(strict_correct)
    strict_acc = n_strict / n_total if n_total > 0 else 0.0

    boundary_count = 0
    soft_correct_count = 0
    for r in valid:
        pct = r.get("sample_data", {}).get("traffic_change_pct", None)
        on_boundary = is_boundary_sample(pct, soft_boundary_band) if soft_boundary_band > 0 else False
        if on_boundary:
            boundary_count += 1
        is_correct_strict = (r["extracted_option"] == r["answer_option"])
        is_correct_soft = is_correct_strict or (on_boundary and r["extracted_option"] == "B")
        if is_correct_soft:
            soft_correct_count += 1
        r["_on_boundary"] = on_boundary
        r["_correct_soft"] = is_correct_soft

    soft_acc = soft_correct_count / n_total if n_total > 0 else 0.0

    per_label = defaultdict(lambda: {"total": 0, "correct": 0, "correct_soft": 0})
    for r in results:
        lbl = r["answer_label"]
        per_label[lbl]["total"] += 1
        if r.get("extracted_option") == r["answer_option"]:
            per_label[lbl]["correct"] += 1
        if r.get("_correct_soft", False):
            per_label[lbl]["correct_soft"] += 1
    per_label_acc = {
        lbl: {
            "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0,
            "total": v["total"],
            "correct": v["correct"],
        }
        for lbl, v in sorted(per_label.items())
    }

    per_cond = {}
    for cond_name in ["is_rain_hour", "has_event_3km", "crash_gt1", "school_holiday"]:
        cond_results = [r for r in results if r["sample_data"].get(cond_name, False)]
        if cond_name == "crash_gt1":
            cond_results = [r for r in results
                          if r["sample_data"].get("crash_count", 0) > 1]
        cond_correct = [r for r in cond_results
                        if r["extracted_option"] == r["answer_option"]]
        per_cond[cond_name] = {
            "accuracy": len(cond_correct) / len(cond_results) if cond_results else 0,
            "total": len(cond_results),
            "correct": len(cond_correct),
        }

    confusion = defaultdict(lambda: defaultdict(int))
    for r in valid:
        confusion[r["answer_label"]][r["extracted_option"]] += 1

    metrics = {
        "overall_accuracy": round(strict_acc, 4),
        "total_qa_pairs": n_total,
        "valid_responses": n_valid,
        "extraction_failures": n_total - n_valid,
        "correct": n_strict,
        "per_label_accuracy": per_label_acc,
        "per_condition_accuracy": per_cond,
        "confusion_matrix": {k: dict(v) for k, v in sorted(confusion.items())},
    }

    if soft_boundary_band > 0:
        per_label_soft = {
            lbl: {
                "accuracy": round(v["correct_soft"] / v["total"], 4) if v["total"] > 0 else 0,
                "correct": v["correct_soft"],
                "total": v["total"],
            }
            for lbl, v in sorted(per_label.items())
        }
        per_cond_soft = {}
        for cond_name in ["is_rain_hour", "has_event_3km", "crash_gt1", "school_holiday"]:
            cond_results = [r for r in results if r["sample_data"].get(cond_name, False)]
            if cond_name == "crash_gt1":
                cond_results = [r for r in results
                              if r["sample_data"].get("crash_count", 0) > 1]
            cond_correct_soft = [r for r in cond_results
                                 if r.get("_correct_soft", False)]
            per_cond_soft[cond_name] = {
                "accuracy": round(len(cond_correct_soft) / len(cond_results), 4) if cond_results else 0,
                "correct": len(cond_correct_soft),
                "total": len(cond_results),
            }

        metrics["soft_scoring"] = {
            "enabled": True,
            "band_pct": soft_boundary_band,
            "boundary_samples": boundary_count,
            "boundary_rescued": sum(
                1 for r in valid
                if r.get("_on_boundary")
                and r.get("_correct_soft", False)
                and r["extracted_option"] != r["answer_option"]
            ),
            "overall_accuracy": round(soft_acc, 4),
            "correct": soft_correct_count,
            "per_label_accuracy": per_label_soft,
            "per_condition_accuracy": per_cond_soft,
        }

    for r in results:
        r.pop("_on_boundary", None)
        r.pop("_correct_soft", None)

    return metrics


# ============================================================================
# Main Pipeline
# ============================================================================

def load_qa_pairs(path: Path, max_pairs: int | None = None,
                  seed: int | None = None) -> list[dict]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))

    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(pairs)

    if max_pairs:
        pairs = pairs[:max_pairs]
    return pairs


def run_benchmark(qa_pairs: list[dict], levels: list[str],
                  model_name: str, output_path: Path | None = None,
                  soft_boundary_band: float = 0.0) -> dict:
    all_results = {}
    interactive = sys.stdin.isatty()

    if output_path and output_path.exists():
        try:
            with open(output_path, encoding="utf-8") as f:
                existing = json.load(f)
            if "results_by_level" in existing:
                for lv in existing["results_by_level"]:
                    all_results[lv] = existing["results_by_level"][lv]
                    m = all_results[lv]
                    print(f"  [Resume] Level {lv} already done: "
                          f"acc={m['overall_accuracy']:.3f} ({m['correct']}/{m['total_qa_pairs']})")
        except (json.JSONDecodeError, KeyError):
            pass

    for level in levels:
        if level in all_results:
            continue

        print(f"\n{'='*60}")
        print(f"Level {level}: {len(qa_pairs)} QA pairs")
        print(f"{'='*60}")

        results = []
        t0 = time.time()

        for i, qa in enumerate(qa_pairs):
            d = qa["sample_data"]
            prompt = render_prompt(d, level)

            option, meta = call_llm(prompt, model_name)
            correct = (option == qa["task_config"]["answer_option"])

            result = {
                "qa_id": qa["qa_id"],
                "level": level,
                "answer_label": qa["task_config"]["answer_label"],
                "answer_option": qa["task_config"]["answer_option"],
                "extracted_option": option,
                "correct": correct,
                "sample_data": d,
                "api_metadata": meta,
            }
            results.append(result)

            acc_so_far = sum(1 for r in results if r["correct"]) / len(results)
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(qa_pairs) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:4d}/{len(qa_pairs)}] "
                  f"acc={acc_so_far:.3f} "
                  f"rate={rate:.1f}/s "
                  f"ETA={eta:.0f}s"
                  + (f"  ERROR: {meta.get('error','')[:60]}" if option is None else ""),
                  end="\r", flush=True)

        elapsed = time.time() - t0
        metrics = compute_metrics(results, soft_boundary_band=soft_boundary_band)
        metrics["runtime_seconds"] = round(elapsed, 1)
        all_results[level] = metrics

        print(f"\n  Done in {elapsed:.0f}s. "
              f"Accuracy={metrics['overall_accuracy']:.3f} "
              f"({metrics['correct']}/{metrics['total_qa_pairs']})")
        if soft_boundary_band > 0 and "soft_scoring" in metrics:
            ss = metrics["soft_scoring"]
            print(f"  Soft accuracy (±{soft_boundary_band:.1f}% band): "
                  f"{ss['overall_accuracy']:.3f} ({ss['correct']}/{metrics['total_qa_pairs']}) "
                  f"[{ss['boundary_samples']} boundary, {ss['boundary_rescued']} rescued]")
        print(f"  Extraction failures: {metrics['extraction_failures']}")

        for lbl, info in metrics["per_label_accuracy"].items():
            print(f"    {lbl:15s}: {info['accuracy']:.3f} "
                  f"({info['correct']}/{info['total']})")

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output = {
                "model": model_name,
                "backend": "openrouter",
                "backend_name": "OpenRouter",
                "prompt_mode": "soft",
                "timestamp": datetime.now().isoformat(),
                "levels_run": sorted(all_results.keys()),
                "total_qa_pairs": len(qa_pairs),
                "results_by_level": dict(all_results),
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"  [Saved] {output_path}")

        if level != levels[-1]:
            remaining = [l for l in levels if l not in all_results]
            if interactive:
                try:
                    print(f"\n  Remaining levels: {remaining}")
                    user_input = input("  Press Enter to continue (q to quit): ").strip().lower()
                    if user_input == "q":
                        print("  User requested quit. Exiting.")
                        break
                except (EOFError, OSError):
                    interactive = False
            if not interactive:
                print(f"  Next: {remaining}")

    return all_results


def _model_to_filename(model_id: str) -> str:
    safe = model_id.replace("/", "__").replace(":", "_")
    return f"soft_{safe}.json"


def _run_one_model(model_name: str, args, levels: list[str]) -> None:
    output_path = Path(args.output_dir) / _model_to_filename(model_name)
    qa_pairs = load_qa_pairs(Path(args.input), max_pairs=args.max, seed=args.seed)

    print(f"\n{'#'*60}")
    print(f"Model: {model_name}")
    print(f"Output: {output_path}")
    print(f"{'#'*60}")

    soft_band = 0.0 if args.no_soft_scoring else args.soft_boundary
    results = run_benchmark(qa_pairs, levels, model_name, output_path=output_path,
                            soft_boundary_band=soft_band)

    done_levels = [l for l in levels if l in results]
    if done_levels:
        print(f"\n  {'Level':<6} {'Acc':>8} {'Corr':>6} {'Total':>6} {'Fail':>5}")
        for lv in done_levels:
            m = results[lv]
            print(f"  {lv:<6} {m['overall_accuracy']:>8.4f} {m['correct']:>6} "
                  f"{m['total_qa_pairs']:>6} {m['extraction_failures']:>5}")
        if soft_band > 0 and "soft_scoring" in results[done_levels[-1]]:
            ss = results[done_levels[-1]]["soft_scoring"]
            print(f"  Soft (±{soft_band:.1f}%): {ss['overall_accuracy']:.4f} "
                  f"boundary={ss['boundary_samples']} rescued={ss['boundary_rescued']}")


def main():
    default_models_str = ",".join(DEFAULT_MODELS)
    parser = argparse.ArgumentParser(
        description="Task 1 Benchmark Runner — Gentle Prompt (No CoT)"
    )
    parser.add_argument("--models", default=default_models_str,
                        help=f"Comma-separated model IDs (default: 2 models)")
    parser.add_argument("--levels", default="L0,L1,L2,L3,L4",
                        help="Comma-separated levels to run (default: all)")
    parser.add_argument("--max", type=int, default=None,
                        help="Max QA pairs to run")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible shuffling (e.g. 42)")
    parser.add_argument("--input", default=str(INPUT_JSONL),
                        help="Path to QA JSONL file")
    parser.add_argument("--output-dir", default=str(OUTPUT_RESULTS_DIR),
                        help="Directory for result JSON files")
    parser.add_argument("--soft-boundary", type=float, default=DEFAULT_SOFT_BOUNDARY_BAND,
                        help="Soft scoring band (±N%% around ±10%% threshold)")
    parser.add_argument("--no-soft-scoring", action="store_true",
                        help="Disable soft scoring")
    parser.add_argument("--delay", type=float, default=RATE_LIMIT_DELAY,
                        help=f"Rate limit delay between calls in seconds (default: {RATE_LIMIT_DELAY})")
    parser.add_argument("--max-tokens", type=int, default=API_MAX_OUTPUT_TOKENS,
                        help=f"Max output tokens per API call (default: {API_MAX_OUTPUT_TOKENS})")
    args = parser.parse_args()

    # Override module-level rate limit delay
    globals()['RATE_LIMIT_DELAY'] = args.delay
    globals()['API_MAX_OUTPUT_TOKENS'] = args.max_tokens

    init_client()

    levels = [l.strip() for l in args.levels.split(",")]
    for lv in levels:
        if lv not in ALL_LEVELS:
            print(f"ERROR: Unknown level '{lv}'. Choose from {ALL_LEVELS}")
            sys.exit(1)

    models = [m.strip() for m in args.models.split(",")]

    print("=" * 60)
    print("Task 1 Benchmark Runner — Gentle Prompt (No CoT)")
    print("=" * 60)
    print(f"Models:          {len(models)}")
    for i, m in enumerate(models):
        print(f"  [{i+1}] {m}")
    print(f"Levels:          {levels}")
    print(f"Base URL:        {OPENROUTER_BASE_URL}")
    print(f"Prompt mode:     gentle (allows reasoning, no forced completion)")
    print(f"Max output:      {args.max_tokens} tokens")
    if args.seed is not None:
        print(f"Random seed:     {args.seed}")
    if args.max:
        print(f"QA pairs:        {args.max}")
    print(f"Output dir:      {args.output_dir}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    for i, model in enumerate(models):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(models)}] {model}")
        print(f"{'='*60}")
        _run_one_model(model, args, levels)

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"All {len(models)} models done in {total_time:.0f}s")
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
