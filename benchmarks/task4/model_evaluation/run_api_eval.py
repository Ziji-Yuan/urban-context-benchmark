"""
Run API evaluation for the Task 4 contrastive traffic QA benchmark.

The script supports:
- dry-run prompt construction without API calls
- resumable per-model predictions
- Google Gemini through google-generativeai
- OpenAI-compatible providers such as OpenRouter/Together/DeepSeek
- metrics and an HTML report
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

try:
    from sklearn.metrics import accuracy_score, f1_score
except ImportError:
    accuracy_score = None
    f1_score = None

from config_models import MODEL_CONFIGS


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR.parent / "qa_generation" / "generated" / "crash_disruption" / "crash_disruption_qa_eval600.csv"
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"
STANDARD_LABELS = ["increase", "decrease", "normal"]
OPTION_LETTERS = ["A", "B", "C"]


def infer_column(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    for col in columns:
        simplified = col.lower().replace("_", "").replace(".", "")
        for candidate in candidates:
            if candidate.lower().replace("_", "").replace(".", "") == simplified:
                return col
    return None


def inspect_schema(df: pd.DataFrame) -> dict[str, str | None]:
    columns = list(df.columns)
    return {
        "question": infer_column(columns, ["question", "prompt", "question_text", "input"]),
        "answer": infer_column(
            columns, ["answer", "correct_answer", "label", "gold_label", "target"]
        ),
        "context_type": infer_column(
            columns,
            [
                "context_type",
                "condition_type",
                "contrast_type",
                "metadata.contrast_type",
                "task_type",
            ],
        ),
        "id": infer_column(columns, ["id", "question_id", "pair_id", "metadata.pair_id"]),
        "options": infer_column(columns, ["options", "choices", "answer_options"]),
    }


def option_columns(df: pd.DataFrame) -> list[str]:
    opts = []
    for letter in OPTION_LETTERS:
        for candidate in [f"options.{letter}", f"option_{letter}", f"{letter}"]:
            if candidate in df.columns:
                opts.append(candidate)
                break
    return opts


def options_text_from_row(row: pd.Series, schema: dict[str, str | None], opt_cols: list[str]) -> str:
    if schema.get("options") and pd.notna(row.get(schema["options"])):
        return str(row.get(schema["options"]))
    if opt_cols:
        lines = []
        for col in opt_cols:
            label = col.split(".")[-1].replace("option_", "").upper()
            lines.append(f"({label}) {row.get(col)}")
        return "\n".join(lines)
    return "(A) Higher than Scenario A\n(B) Lower than Scenario A\n(C) Similar to Scenario A"


def question_contains_options(question_text: str) -> bool:
    return all(re.search(rf"\({letter}\)", question_text, flags=re.IGNORECASE) for letter in OPTION_LETTERS)


def normalize_percent_display(text: str) -> str:
    def replace_percent(match: re.Match) -> str:
        value = float(match.group(1).replace(",", ""))
        if 100 < value <= 10000:
            value = value / 100
        if value.is_integer():
            return f"{value:.0f}%"
        return f"{value:.2f}".rstrip("0").rstrip(".") + "%"

    return re.sub(r"(\d+(?:,\d{3})*(?:\.\d+)?)%", replace_percent, text)


def sanitize_question_text(question_text: str) -> str:
    text = normalize_percent_display(question_text)
    text = re.sub(
        r"\n*Please briefly explain your reasoning, then answer with one option\.\s*"
        r"Please end your response with:\s*Final answer:\s*A, B, or C\s*",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return text.strip()


def text_to_standard_label(text: Any) -> str | None:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None
    lowered = str(text).strip().lower()
    if not lowered:
        return None

    increase_terms = ["increase", "higher", "rise", "rises", "more traffic", "go up"]
    decrease_terms = [
        "decrease",
        "lower",
        "reduced",
        "reduce",
        "decline",
        "less traffic",
        "go down",
    ]
    normal_terms = [
        "normal",
        "similar",
        "same",
        "no change",
        "unchanged",
        "stable",
        "little change",
    ]

    if any(term in lowered for term in increase_terms):
        return "increase"
    if any(term in lowered for term in decrease_terms):
        return "decrease"
    if any(term in lowered for term in normal_terms):
        return "normal"
    return None


def option_label_map_from_row(
    row: pd.Series, schema: dict[str, str | None], opt_cols: list[str]
) -> dict[str, str]:
    # This benchmark has a fixed option order; do not infer labels from long
    # option sentences because "Scenario A" can cause false keyword matches.
    return {"A": "increase", "B": "decrease", "C": "normal"}

    # Retained for compatibility if this script is later generalized.
    mapping: dict[str, str] = {}
    for col in opt_cols:
        letter = col.split(".")[-1].replace("option_", "").upper()
        if letter in OPTION_LETTERS:
            label = text_to_standard_label(row.get(col))
            if label:
                mapping[letter] = label

    if mapping:
        return mapping

    options_text = options_text_from_row(row, schema, opt_cols)
    for letter in OPTION_LETTERS:
        match = re.search(
            rf"\(?{letter}\)?\s*[:\.\-]?\s*(.+?)(?=\n\s*\(?[ABC]\)?\s*[:\.\-]?|\Z)",
            options_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            label = text_to_standard_label(match.group(1))
            if label:
                mapping[letter] = label

    if mapping:
        return mapping

    question_text = str(row.get(schema["question"], "")) if schema.get("question") else ""
    for letter in OPTION_LETTERS:
        match = re.search(
            rf"\({letter}\)\s*(.+?)(?=\n\s*\([ABC]\)|\Z)",
            question_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            label = text_to_standard_label(match.group(1))
            if label:
                mapping[letter] = label
    return mapping


def inverse_option_map(option_map: dict[str, str]) -> dict[str, str]:
    return {label: letter for letter, label in option_map.items()}


def normalize_gold_answer(value: Any, option_map: dict[str, str]) -> dict[str, str]:
    if pd.isna(value):
        return {"gold_option": "", "gold_label": "invalid", "gold_answer_type": "invalid"}
    text = str(value).strip()
    upper = text.upper()
    if upper in OPTION_LETTERS:
        return {
            "gold_option": upper,
            "gold_label": option_map.get(upper, "invalid"),
            "gold_answer_type": "option",
        }
    label = text_to_standard_label(text) or "invalid"
    return {
        "gold_option": inverse_option_map(option_map).get(label, ""),
        "gold_label": label,
        "gold_answer_type": "label",
    }


def build_prompt(row: pd.Series, schema: dict[str, str | None], opt_cols: list[str]) -> str:
    question_col = schema.get("question")
    raw_question_text = str(row.get(question_col, "")) if question_col else ""
    question_text = sanitize_question_text(raw_question_text)
    options_text = options_text_from_row(row, schema, opt_cols)
    options_block = "" if question_contains_options(question_text) else f"\n\nOptions:\n{options_text}"
    prompt = (
        "You are answering a traffic reasoning benchmark question.\n\n"
        f"Question:\n{question_text}"
        f"{options_block}\n\n"
        "Please answer with exactly one letter: A, B, or C.\n"
        "Do not provide explanation or any other text."
    )
    return normalize_percent_display(prompt)


def parse_prediction(raw_output: Any, option_map: dict[str, str] | None = None) -> dict[str, str]:
    option_map = option_map or {}
    if raw_output is None or (isinstance(raw_output, float) and np.isnan(raw_output)):
        return {"option": "", "label": "invalid"}
    text = str(raw_output).strip().lower()
    if not text:
        return {"option": "", "label": "invalid"}

    first_line = text.splitlines()[0].strip()
    letter_match = re.search(r"\b([abc])\b", first_line)
    if letter_match:
        option = letter_match.group(1).upper()
        return {"option": option, "label": option_map.get(option, "invalid")}

    final_match = re.search(r"final\s*answer\s*[:\-]?\s*([abc])\b", text)
    if final_match:
        option = final_match.group(1).upper()
        return {"option": option, "label": option_map.get(option, "invalid")}

    label = text_to_standard_label(text)
    if label:
        return {"option": inverse_option_map(option_map).get(label, ""), "label": label}
    return {"option": "", "label": "invalid"}


def load_data(input_path: Path, limit: int | None) -> tuple[pd.DataFrame, dict[str, str | None], list[str]]:
    df = pd.read_csv(input_path)
    if limit is not None:
        df = df.head(limit).copy()
    schema = inspect_schema(df)
    opt_cols = option_columns(df)

    if not schema["question"]:
        raise ValueError("Could not infer question/prompt column.")
    if not schema["answer"]:
        raise ValueError("Could not infer answer/gold label column.")

    df["question_id_norm"] = (
        df[schema["id"]].astype(str)
        if schema["id"]
        else [f"q_{i:04d}" for i in range(len(df))]
    )
    df["context_type_norm"] = (
        df[schema["context_type"]].astype(str) if schema["context_type"] else "unknown"
    )
    df["options_norm"] = df.apply(lambda row: options_text_from_row(row, schema, opt_cols), axis=1)
    df["option_label_map"] = df.apply(
        lambda row: option_label_map_from_row(row, schema, opt_cols), axis=1
    )
    gold_norm = df.apply(
        lambda row: normalize_gold_answer(row[schema["answer"]], row["option_label_map"]),
        axis=1,
        result_type="expand",
    )
    df["gold_option_norm"] = gold_norm["gold_option"]
    df["gold_label_norm"] = gold_norm["gold_label"]
    df["gold_answer_type"] = gold_norm["gold_answer_type"]
    df["prompt_norm"] = df.apply(lambda row: build_prompt(row, schema, opt_cols), axis=1)
    return df, schema, opt_cols


def selected_models(model_arg: str) -> list[str]:
    if model_arg.strip().lower() == "all":
        return [key for key, cfg in MODEL_CONFIGS.items() if cfg.get("enabled", True)]
    keys = [m.strip() for m in model_arg.split(",") if m.strip()]
    unknown = [key for key in keys if key not in MODEL_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown model key(s): {unknown}")
    return keys


def call_google_model(config: dict, prompt: str, temperature: float) -> str:
    import google.generativeai as genai

    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {config['api_key_env']}")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(config["model_id"])
    response = model.generate_content(
        prompt,
        generation_config={"temperature": temperature, "max_output_tokens": 32},
    )
    return getattr(response, "text", "") or ""


def call_openai_compatible(config: dict, prompt: str, temperature: float) -> str:
    from openai import OpenAI

    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {config['api_key_env']}")
    client = OpenAI(api_key=api_key, base_url=config.get("base_url"))
    response = client.chat.completions.create(
        model=config["model_id"],
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=256,
    )
    return response.choices[0].message.content or ""


def call_model(config: dict, prompt: str, temperature: float, retries: int = 3) -> str:
    provider = config["provider"]
    for attempt in range(retries):
        try:
            if provider == "google":
                return call_google_model(config, prompt, temperature)
            if provider == "openai_compatible":
                return call_openai_compatible(config, prompt, temperature)
            raise ValueError(f"Unsupported provider: {provider}")
        except (requests.RequestException, TimeoutError, RuntimeError) as exc:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def prediction_path(model_key: str) -> Path:
    return RESULTS_DIR / f"{model_key}_predictions.csv"


def load_existing_predictions(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def save_predictions(rows: list[dict], path: Path) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def run_model_eval(
    df: pd.DataFrame,
    model_key: str,
    resume: bool,
    temperature: float,
) -> None:
    config = MODEL_CONFIGS[model_key]
    out_path = prediction_path(model_key)
    existing = load_existing_predictions(out_path)
    rows = existing.to_dict("records") if resume and not existing.empty else []
    done_ids = {
        str(row.get("question_id"))
        for row in rows
        if row.get("parsed_prediction") in STANDARD_LABELS + ["invalid"]
    }

    for _, row in df.iterrows():
        qid = str(row["question_id_norm"])
        if resume and qid in done_ids:
            continue

        raw_output = ""
        error_message = ""
        try:
            raw_output = call_model(config, row["prompt_norm"], temperature)
        except Exception as exc:  # API errors should not stop the whole run.
            error_message = repr(exc)

        parsed = parse_prediction(raw_output, row["option_label_map"])
        predicted_option = parsed["option"]
        predicted_label = parsed["label"]
        if row["gold_answer_type"] == "option":
            correct = predicted_option == row["gold_option_norm"]
        else:
            correct = predicted_label == row["gold_label_norm"]
        rows.append(
            {
                "question_id": qid,
                "model_key": model_key,
                "model_name": config["display_name"],
                "provider": config["provider"],
                "context_type": row["context_type_norm"],
                "question": row.get("question", ""),
                "options": row["options_norm"],
                "option_label_map": json.dumps(row["option_label_map"], sort_keys=True),
                "correct_answer": row["gold_label_norm"],
                "correct_option": row["gold_option_norm"],
                "gold_answer_type": row["gold_answer_type"],
                "raw_model_output": raw_output,
                "predicted_option": predicted_option,
                "parsed_prediction": predicted_label,
                "is_correct": bool(correct),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "error_message": error_message,
            }
        )
        save_predictions(rows, out_path)
        print(f"{model_key}: saved prediction {len(rows)} -> {out_path.name}")


def merge_report_rows(
    existing_path: Path,
    new_df: pd.DataFrame,
    model_keys: list[str] | None,
    write_global: bool,
    sort_columns: list[str],
) -> pd.DataFrame:
    if write_global or model_keys is None or not existing_path.exists():
        merged = new_df.copy()
    else:
        existing = pd.read_csv(existing_path)
        if existing.empty or "model_key" not in existing.columns:
            merged = new_df.copy()
        else:
            replace_keys = set(model_keys)
            kept = existing[~existing["model_key"].astype(str).isin(replace_keys)]
            merged = pd.concat([kept, new_df], ignore_index=True)
    if not merged.empty:
        available = [col for col in sort_columns if col in merged.columns]
        if available:
            merged = merged.sort_values(available).reset_index(drop=True)
    return merged


def load_all_predictions() -> pd.DataFrame:
    prediction_files = sorted(RESULTS_DIR.glob("*_predictions.csv"))
    frames = [pd.read_csv(path) for path in prediction_files]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compute_metrics(
    model_keys: list[str] | None = None,
    write_global: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if accuracy_score is None or f1_score is None:
        raise RuntimeError("scikit-learn is required for metrics. Install requirements.txt.")

    if model_keys is None:
        prediction_files = sorted(RESULTS_DIR.glob("*_predictions.csv"))
    else:
        prediction_files = [prediction_path(key) for key in model_keys if prediction_path(key).exists()]
    frames = [pd.read_csv(path) for path in prediction_files]
    if not frames:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    preds = pd.concat(frames, ignore_index=True)
    metric_rows = []
    condition_rows = []
    bias_rows = []

    for model_key, group in preds.groupby("model_key"):
        y_true = group["correct_answer"].astype(str)
        y_pred = group["parsed_prediction"].astype(str)
        metric_rows.append(
            {
                "model_key": model_key,
                "model_name": group["model_name"].iloc[0],
                "provider": group["provider"].iloc[0],
                "n": len(group),
                "contrastive_accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(
                    y_true, y_pred, labels=STANDARD_LABELS, average="macro", zero_division=0
                ),
            }
        )
        for context, ctx_group in group.groupby("context_type"):
            condition_rows.append(
                {
                    "model_key": model_key,
                    "context_type": context,
                    "n": len(ctx_group),
                    "accuracy": accuracy_score(
                        ctx_group["correct_answer"].astype(str),
                        ctx_group["parsed_prediction"].astype(str),
                    ),
                }
            )
        counts = y_pred.value_counts(normalize=True).to_dict()
        bias_rows.append(
            {
                "model_key": model_key,
                "pct_increase": counts.get("increase", 0.0),
                "pct_decrease": counts.get("decrease", 0.0),
                "pct_normal": counts.get("normal", 0.0),
                "pct_invalid": counts.get("invalid", 0.0),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    per_condition = pd.DataFrame(condition_rows)
    directional_bias = pd.DataFrame(bias_rows)
    errors = preds[(preds["is_correct"] != True) | (preds["parsed_prediction"] == "invalid")]
    updated_model_keys = sorted(metrics["model_key"].astype(str).unique()) if not metrics.empty else model_keys

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    merged_metrics = merge_report_rows(
        REPORTS_DIR / "metrics_summary.csv",
        metrics,
        updated_model_keys,
        write_global,
        ["model_key"],
    )
    merged_errors = merge_report_rows(
        REPORTS_DIR / "error_analysis.csv",
        errors,
        updated_model_keys,
        write_global,
        ["model_key", "question_id"],
    )
    merged_condition = merge_report_rows(
        REPORTS_DIR / "per_condition_accuracy.csv",
        per_condition,
        updated_model_keys,
        write_global,
        ["model_key", "context_type"],
    )
    merged_bias = merge_report_rows(
        REPORTS_DIR / "directional_bias.csv",
        directional_bias,
        updated_model_keys,
        write_global,
        ["model_key"],
    )

    merged_metrics.to_csv(REPORTS_DIR / "metrics_summary.csv", index=False)
    merged_errors.to_csv(REPORTS_DIR / "error_analysis.csv", index=False)
    merged_condition.to_csv(REPORTS_DIR / "per_condition_accuracy.csv", index=False)
    merged_bias.to_csv(REPORTS_DIR / "directional_bias.csv", index=False)
    report_preds = load_all_predictions()
    generate_html_report(merged_metrics, merged_condition, merged_bias, merged_errors, report_preds)
    return merged_metrics, merged_condition, merged_bias


def generate_html_report(
    metrics: pd.DataFrame,
    per_condition: pd.DataFrame,
    directional_bias: pd.DataFrame,
    errors: pd.DataFrame,
    preds: pd.DataFrame,
) -> None:
    wrong_sample = errors.head(20).copy()
    invalid_sample = errors[errors["parsed_prediction"] == "invalid"].head(20).copy()
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Contrastive Traffic QA Benchmark Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.4; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f3f5f7; }}
    code {{ background: #f6f8fa; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Contrastive Traffic QA Benchmark Report</h1>
  <p><b>Generated:</b> {datetime.now().isoformat(timespec="seconds")}</p>
  <p><b>Input file:</b> <code>{html.escape(str(DEFAULT_INPUT))}</code></p>
  <p><b>Total predictions loaded:</b> {len(preds)}</p>

  <h2>Model Summary</h2>
  {metrics.to_html(index=False, escape=True) if not metrics.empty else "<p>No metrics yet.</p>"}

  <h2>Per-Condition Accuracy</h2>
  {per_condition.to_html(index=False, escape=True) if not per_condition.empty else "<p>No condition metrics yet.</p>"}

  <h2>Directional Bias</h2>
  {directional_bias.to_html(index=False, escape=True) if not directional_bias.empty else "<p>No bias metrics yet.</p>"}

  <h2>Sample Wrong Predictions</h2>
  {wrong_sample.to_html(index=False, escape=True) if not wrong_sample.empty else "<p>No wrong predictions.</p>"}

  <h2>Sample Invalid Predictions</h2>
  {invalid_sample.to_html(index=False, escape=True) if not invalid_sample.empty else "<p>No invalid predictions.</p>"}
</body>
</html>"""
    (REPORTS_DIR / "benchmark_report.html").write_text(html_text, encoding="utf-8")


def dry_run(df: pd.DataFrame, schema: dict[str, str | None], limit: int | None) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    n = min(limit or 3, len(df), 10)
    preview = df.head(n)[["question_id_norm", "context_type_norm", "gold_label_norm", "prompt_norm"]]
    preview.to_csv(REPORTS_DIR / "dry_run_prompts.csv", index=False)
    print(f"Number of rows loaded: {len(df)}")
    print(f"Column names: {list(df.columns)}")
    print(f"Inferred question column: {schema['question']}")
    print(f"Inferred answer/gold label column: {schema['answer']}")
    print(f"Inferred context type column: {schema['context_type']}")
    print("\nPreview of one constructed prompt:\n")
    print(df.iloc[0]["prompt_norm"])
    print(f"\nSaved dry-run prompts: {REPORTS_DIR / 'dry_run_prompts.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run contrastive QA API evaluation.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--models", "--model", dest="models", default="gemini_2_5_flash")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reports-only", action="store_true", help="Regenerate reports from existing results without API calls.")
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    load_dotenv(BASE_DIR / ".env")
    args = parse_args()
    model_keys = selected_models(args.models)
    write_global = args.models.strip().lower() == "all"

    if args.reports_only:
        metrics, _, _ = compute_metrics(None if write_global else model_keys, write_global=write_global)
        print(metrics if not metrics.empty else "No metrics generated.")
        return

    df, schema, _ = load_data(args.input, args.limit)

    if args.dry_run:
        dry_run(df, schema, args.limit)
        return

    print(f"Selected models: {model_keys}")
    for model_key in model_keys:
        run_model_eval(df, model_key, resume=args.resume, temperature=args.temperature)
    metrics, _, _ = compute_metrics(model_keys, write_global=write_global)
    print(metrics if not metrics.empty else "No metrics generated.")


if __name__ == "__main__":
    main()
