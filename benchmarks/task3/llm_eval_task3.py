import os
import sys
import time
import re
import argparse
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    cohen_kappa_score,
)


# ============================================================
# 1. MODEL CONFIG
# ============================================================
#
# IMPORTANT:
# - Groq may retire or rename hosted models.
# - The script checks Groq's live model list before evaluating.
# - It will stop with a clear message instead of silently replacing a
#   requested model with a different model.
#
# Run:
#   python llm_eval_task3.py --list
# to see the configured aliases.
#
# Run:
#   python llm_eval_task3.py --list-live-groq
# to see the Groq models currently available to your API key.

MODELS = {
    # Final 8 LLMs used in Task 3 evaluation

    # Google Gemini via OpenRouter
    "gemini_2_5_flash": {
        "provider": "openrouter",
        "model": "google/gemini-2.5-flash",
    },
    "gemini_2_5_pro": {
        "provider": "openrouter",
        "model": "google/gemini-2.5-pro",
    },

    # Meta Llama via OpenRouter
    "llama_3_1_8b": {
        "provider": "openrouter",
        "model": "meta-llama/llama-3.1-8b-instruct",
    },
    "llama_3_3_70b": {
        "provider": "openrouter",
        "model": "meta-llama/llama-3.3-70b-instruct",
    },

    # Qwen via Groq
    "qwen3_32b": {
        "provider": "groq",
        "model": "qwen/qwen3-32b",
    },
}

LABELS = ["A", "B", "C"]

BENCHMARK_FILE = "task3_anomaly_classification.csv"
PREDICTIONS_FILE = "llm_predictions_task3_anomaly.csv"
METRICS_FILE = "llm_model_metrics_task3_anomaly.csv"
PER_CLASS_FILE = "llm_per_class_metrics_task3_anomaly.csv"


# ============================================================
# 2. CLI ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one LLM at a time on the Task 3 anomaly benchmark."
    )

    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        help="Model alias to evaluate. Use --list to see all aliases.",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List configured model aliases and exit.",
    )

    parser.add_argument(
        "--list-live-groq",
        action="store_true",
        help="List Groq models currently available to your API key and exit.",
    )

    parser.add_argument(
        "--list-live-openrouter",
        action="store_true",
        help="List OpenRouter models currently available and exit.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace any previously saved results for this model.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted run by skipping row_id values already "
            "saved for the selected model."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only N benchmark rows for testing.",
    )

    parser.add_argument(
        "--sample-mode",
        choices=["balanced", "random", "first"],
        default="balanced",
        help=(
            "How --limit selects rows. Default: balanced, which samples "
            "approximately equal numbers of A, B, and C."
        ),
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.02,
        help="Seconds to pause between successful API requests. Default: 0.02.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries for temporary rate limits. Default: 3.",
    )

    parser.add_argument(
        "--retry-wait",
        type=float,
        default=60.0,
        help="Base seconds between retries. Default: 60.",
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Save partial results every N completed rows. Default: 25.",
    )

    return parser.parse_args()


# ============================================================
# 3. LOAD BENCHMARK
# ============================================================

def load_benchmark(limit=None, sample_mode="balanced", random_state=42):
    if not os.path.exists(BENCHMARK_FILE):
        raise FileNotFoundError(
            f"Benchmark file not found: {BENCHMARK_FILE}"
        )

    benchmark = pd.read_csv(BENCHMARK_FILE)

    required_cols = ["question", "answer"]
    missing = [col for col in required_cols if col not in benchmark.columns]

    if missing:
        raise ValueError(
            f"Missing required benchmark columns: {missing}"
        )

    benchmark["expected_answer"] = (
        benchmark["answer"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    invalid_expected = benchmark[
        ~benchmark["expected_answer"].isin(LABELS)
    ]

    if not invalid_expected.empty:
        bad_values = sorted(
            invalid_expected["expected_answer"]
            .dropna()
            .unique()
            .tolist()
        )
        raise ValueError(
            f"Benchmark contains invalid expected answers: {bad_values}. "
            f"Expected only {LABELS}."
        )

    if limit is None:
        return benchmark

    if limit <= 0:
        raise ValueError("--limit must be greater than 0.")

    limit = min(limit, len(benchmark))

    if sample_mode == "first":
        return benchmark.head(limit).copy()

    if sample_mode == "random":
        return benchmark.sample(
            n=limit,
            random_state=random_state,
        ).copy()

    if sample_mode != "balanced":
        raise ValueError(
            "--sample-mode must be one of: balanced, random, first."
        )

    per_class = limit // len(LABELS)
    remainder = limit % len(LABELS)
    selected_parts = []

    for position, label in enumerate(LABELS):
        class_rows = benchmark[
            benchmark["expected_answer"] == label
        ]

        class_n = per_class + (1 if position < remainder else 0)
        class_n = min(class_n, len(class_rows))

        if class_n > 0:
            selected_parts.append(
                class_rows.sample(
                    n=class_n,
                    random_state=random_state + position,
                )
            )

    if selected_parts:
        selected = pd.concat(
            selected_parts,
            ignore_index=False,
        )
    else:
        selected = benchmark.iloc[0:0].copy()

    # Fill a possible shortfall if one class has too few rows.
    if len(selected) < limit:
        candidates = benchmark[
            ~benchmark.index.isin(selected.index)
        ]

        extra_n = min(limit - len(selected), len(candidates))

        if extra_n > 0:
            selected = pd.concat(
                [
                    selected,
                    candidates.sample(
                        n=extra_n,
                        random_state=random_state + 99,
                    ),
                ],
                ignore_index=False,
            )

    return selected.sample(
        frac=1,
        random_state=random_state,
    ).copy()


# ============================================================
# 4. PROMPT
# ============================================================

def build_prompt(question):
    return f"""
You are evaluating an urban traffic anomaly benchmark.

Read the following traffic scenario carefully.

{question}

Return ONLY ONE character.

Allowed outputs:

A
B
C

If you output anything else,
your answer is considered WRONG.

Do NOT explain.

Do NOT think.

Do NOT output <think>.

Do NOT output reasoning.

Output exactly one character.

Do not write words.
Do not use punctuation.
""".strip()


# ============================================================
# 5. OUTPUT PARSER
# ============================================================

def parse_label(text):
    if text is None:
        return "invalid"

    text = str(text).strip()

    if not text:
        return "invalid"

    # Remove complete reasoning blocks.
    if re.search(r"<think>", text, flags=re.IGNORECASE) and re.search(
        r"</think>",
        text,
        flags=re.IGNORECASE,
    ):
        text = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

    # Do not extract a label from unfinished hidden reasoning.
    elif re.search(r"<think>", text, flags=re.IGNORECASE):
        before_think = re.split(
            r"<think>",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()

        if before_think:
            text = before_think
        else:
            return "invalid"

    if not text:
        return "invalid"

    upper = text.upper()
    lower = text.lower()

    # Accept textual labels if a model ignores the one-letter instruction.
    if "major_anomaly" in lower or "major anomaly" in lower:
        return "C"

    if "minor_anomaly" in lower or "minor anomaly" in lower:
        return "B"

    if re.search(r"\bnormal\b", lower):
        return "A"

    # Best case: exactly one valid letter.
    if upper in LABELS:
        return upper

    patterns = [
        r"\b(?:OPTION|ANSWER|CHOICE)\s*[:\-]?\s*([ABC])\b",
        r"\*\*([ABC])\*\*",
        r"\(([ABC])\)",
        r"\b([ABC])\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, upper)

        if match:
            return match.group(1)

    return "invalid"


# ============================================================
# 6. API CLIENTS AND MODEL AVAILABILITY
# ============================================================

def make_client(provider):
    """
    Create only the client needed for the selected provider.
    """

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "Missing GROQ_API_KEY. Set it in PowerShell before running."
            )

        try:
            from groq import Groq
        except ImportError as exc:
            raise ImportError(
                "The groq package is not installed. "
                "Run: pip install -U groq"
            ) from exc

        return Groq(api_key=api_key, max_retries=0)

    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "Missing GOOGLE_API_KEY. Set it in PowerShell before running."
            )

        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "The Google Gen AI SDK is not installed. "
                "Run: pip install -U google-genai"
            ) from exc

        return genai.Client(api_key=api_key)

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError(
                "Missing OPENROUTER_API_KEY. Set it in PowerShell before running."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The OpenAI SDK is not installed. "
                "Run: pip install -U openai"
            ) from exc

        default_headers = {
            "HTTP-Referer": os.getenv(
                "OPENROUTER_SITE_URL",
                "https://github.com/nesaugust",
            ),
            "X-Title": os.getenv(
                "OPENROUTER_APP_NAME",
                "Urban Context Benchmark",
            ),
        }

        return OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers=default_headers,
            max_retries=0,
        )

    raise ValueError(f"Unknown provider: {provider}")


def get_live_groq_model_ids(client):
    """
    Return model IDs currently visible to the user's Groq account.
    """

    response = client.models.list()
    data = getattr(response, "data", response)

    model_ids = []

    for item in data:
        model_id = getattr(item, "id", None)

        if model_id:
            model_ids.append(model_id)

    return sorted(set(model_ids))


def verify_groq_model_available(client, model_alias, model_name):
    """
    Stop cleanly when the configured Groq model is unavailable.

    The function does not silently substitute another model because doing so
    would make the benchmark results scientifically misleading.
    """
    live_models = get_live_groq_model_ids(client)

    if model_name in live_models:
        return

    # Show a few other general text-generation models visible to this API key.
    excluded_terms = (
        "whisper",
        "guard",
        "safeguard",
        "orpheus",
        "compound",
    )

    suggested = [
        model_id
        for model_id in live_models
        if model_id != model_name
        and not any(term in model_id.lower() for term in excluded_terms)
    ][:5]

    message = [
        "",
        f"Model alias: {model_alias}",
        f"Configured Groq model ID: {model_name}",
        "",
        "This model ID is not currently available to your Groq account.",
        "The evaluation has been stopped so another model is not used",
        "silently under the requested alias.",
    ]

    if suggested:
        message.extend(
            [
                "",
                "Other available general-purpose models:",
                *[f"  - {item}" for item in suggested],
            ]
        )

    message.extend(
        [
            "",
            "Check the complete live Groq catalogue with:",
            "  python llm_eval_task3.py --list-live-groq",
        ]
    )

    raise RuntimeError("\n".join(message))


def get_live_openrouter_model_ids(client):
    """Return model IDs currently listed by OpenRouter."""
    response = client.models.list()
    data = getattr(response, "data", response)

    model_ids = []
    for item in data:
        model_id = getattr(item, "id", None)
        if model_id:
            model_ids.append(model_id)

    return sorted(set(model_ids))


def verify_openrouter_model_available(client, model_alias, model_name):
    """Stop cleanly if an OpenRouter model ID is no longer listed."""
    live_models = get_live_openrouter_model_ids(client)

    if model_name in live_models:
        return

    raise RuntimeError(
        "\n".join(
            [
                "",
                f"Model alias: {model_alias}",
                f"Configured OpenRouter model ID: {model_name}",
                "",
                "This model ID is not currently listed by OpenRouter.",
                "Check available models with:",
                "  python llm_eval_task3.py --list-live-openrouter",
            ]
        )
    )


# ============================================================
# 7. MODEL CALLS
# ============================================================

def call_groq(client, model_name, prompt):
    request_kwargs = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": (
                    "You are a strict classification system. "
                    "Return exactly one character: A, B, or C. "
                    "Do not provide reasoning or explanation.\n\n"
                    + prompt
                ),
            },
        ],
        "temperature": 0,
        "max_completion_tokens": 64,
    }

    # Qwen 3.6 supports fully disabling reasoning.
    if model_name == "qwen/qwen3.6-27b":
        request_kwargs["reasoning_effort"] = "none"
        request_kwargs["temperature"] = 0.7
        request_kwargs["top_p"] = 0.8

    # GPT-OSS always reasons internally. With only 20 completion tokens, its
    # reasoning could consume the whole budget and leave message.content empty.
    # Use low effort, hide reasoning, and leave enough tokens for the final label.
    if model_name in [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    ]:
        request_kwargs["reasoning_effort"] = "low"
        request_kwargs["include_reasoning"] = False
        request_kwargs["temperature"] = 0.6
        request_kwargs["top_p"] = 0.95
        request_kwargs["max_completion_tokens"] = 128

    response = client.chat.completions.create(**request_kwargs)

    message = response.choices[0].message
    content = message.content

    return "" if content is None else content.strip()


def call_google(client, model_name, prompt):
    try:
        from google.genai import types
    except ImportError as exc:
        raise ImportError(
            "The Google Gen AI SDK is not installed. "
            "Run: pip install -U google-genai"
        ) from exc

    # Gemini 2.5 Flash supports fully disabling thinking.
    # Gemini 2.5 Pro does not support thinking_budget=0,
    # so it must use its minimum supported budget instead.
    if model_name == "gemini-2.5-flash":
        thinking_config = types.ThinkingConfig(
            thinking_budget=0,
            include_thoughts=False,
        )

    elif model_name == "gemini-2.5-pro":
        thinking_config = types.ThinkingConfig(
            thinking_budget=128,
            include_thoughts=False,
        )

    else:
        thinking_config = None

    config_kwargs = {
        "temperature": 0,
        "max_output_tokens": 10,
        "system_instruction": (
            "You are a strict classification system. "
            "Return exactly one character: A, B, or C. "
            "Do not explain your answer. "
            "Do not provide reasoning."
        ),
        "response_mime_type": "text/plain",
    }

    if thinking_config is not None:
        config_kwargs["thinking_config"] = thinking_config

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )

    if response.text is None:
        return ""

    return response.text.strip()


def call_openrouter(client, model_name, prompt):
    request_kwargs = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Classify the scenario using exactly one label.\n"
                    "Valid labels: A, B, C.\n"
                    "Return only the final label."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0,
        "max_tokens": 64,
    }

    # Gemini 2.5 Pro still uses reasoning tokens for short classification.
    # Reasoning tokens count toward max_tokens, so provide enough room
    # for reasoning plus the final visible A/B/C label.
    if model_name == "google/gemini-2.5-pro":
        request_kwargs["max_tokens"] = 2048
        request_kwargs["extra_body"] = {
            "reasoning": {
                "effort": "low",
                "exclude": True,
            },
            "provider": {
                "allow_fallbacks": True,
            },
        }

    # Gemini 2.5 Flash can run without reasoning for this task.
    if model_name == "google/gemini-2.5-flash":
        request_kwargs["max_tokens"] = 64
        request_kwargs["extra_body"] = {
            "reasoning": {
                "effort": "none",
                "exclude": True,
            },
            "provider": {
                "allow_fallbacks": True,
            },
        }

    # DeepSeek R1 Distill cannot reliably disable reasoning.
    if model_name == "deepseek/deepseek-r1-distill-llama-70b":
        request_kwargs["max_tokens"] = 4096
        request_kwargs["extra_body"] = {
            "reasoning": {
                "effort": "low",
                "exclude": True,
            },
            "provider": {
                "allow_fallbacks": True,
            },
        }

    response = client.chat.completions.create(**request_kwargs)

    choice = response.choices[0]
    message = choice.message
    content = message.content

    if content is None or not str(content).strip():
        finish_reason = getattr(choice, "finish_reason", None)

        usage = getattr(response, "usage", None)
        completion_tokens = getattr(
            usage,
            "completion_tokens",
            None,
        )

        reasoning_tokens = None

        if usage is not None:
            details = getattr(
                usage,
                "completion_tokens_details",
                None,
            )

            if details is not None:
                reasoning_tokens = getattr(
                    details,
                    "reasoning_tokens",
                    None,
                )

        print(
            "\n[OPENROUTER EMPTY RESPONSE]"
            f"\n  model: {model_name}"
            f"\n  finish_reason: {finish_reason}"
            f"\n  completion_tokens: {completion_tokens}"
            f"\n  reasoning_tokens: {reasoning_tokens}"
        )

        return ""

    return str(content).strip()


def call_model(client, provider, model_name, prompt):
    if provider == "groq":
        return call_groq(client, model_name, prompt)

    if provider == "google":
        return call_google(client, model_name, prompt)

    if provider == "openrouter":
        return call_openrouter(client, model_name, prompt)

    raise ValueError(f"Unknown provider: {provider}")


# ============================================================
# 8. SAVED RESULTS
# ============================================================

def load_existing_predictions():
    if not os.path.exists(PREDICTIONS_FILE):
        return pd.DataFrame()

    try:
        return pd.read_csv(PREDICTIONS_FILE)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def model_saved_row_count(existing, model_alias):
    if existing.empty or "model_alias" not in existing.columns:
        return 0

    return int((existing["model_alias"] == model_alias).sum())


def remove_previous_model_rows(existing, model_alias):
    if existing.empty or "model_alias" not in existing.columns:
        return existing

    return existing[
        existing["model_alias"] != model_alias
    ].copy()


def get_completed_row_ids(existing, model_alias):
    """Return benchmark row_id values already saved for a model."""
    if (
        existing.empty
        or "model_alias" not in existing.columns
        or "row_id" not in existing.columns
    ):
        return set()

    model_rows = existing[
        existing["model_alias"] == model_alias
    ].copy()

    if model_rows.empty:
        return set()

    row_ids = pd.to_numeric(
        model_rows["row_id"],
        errors="coerce",
    ).dropna()

    return set(row_ids.astype(int).tolist())


def keep_latest_model_rows(existing, model_alias):
    """Remove duplicate saved rows for one model, keeping the latest copy."""
    if (
        existing.empty
        or "model_alias" not in existing.columns
        or "row_id" not in existing.columns
    ):
        return existing

    other_models = existing[
        existing["model_alias"] != model_alias
    ].copy()

    model_rows = existing[
        existing["model_alias"] == model_alias
    ].copy()

    if model_rows.empty:
        return existing

    model_rows = model_rows.drop_duplicates(
        subset=["model_alias", "row_id"],
        keep="last",
    )

    return pd.concat(
        [other_models, model_rows],
        ignore_index=True,
    )


# ============================================================
# 9. RATE LIMIT HELPERS
# ============================================================

def is_rate_limit_error(text):
    text = str(text).lower()

    markers = [
        "429",
        "resource_exhausted",
        "rate limit",
        "too many requests",
        "quota",
    ]

    return any(marker in text for marker in markers)


def is_hard_quota_error(text):
    text = str(text).lower()

    markers = [
        "per day",
        "daily",
        "quota exceeded",
        "limit: 0",
        "billing",
        "insufficient quota",
    ]

    return any(marker in text for marker in markers)


def get_server_retry_wait(exc, fallback_seconds):
    """Use Groq's Retry-After header when available."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)

    if headers:
        retry_after = headers.get("retry-after")
        if retry_after is not None:
            try:
                return max(float(retry_after), 1.0)
            except (TypeError, ValueError):
                pass

        # Groq may expose a reset duration such as "1.23s".
        reset = headers.get("x-ratelimit-reset-tokens") or headers.get("x-ratelimit-reset-requests")
        if reset:
            match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(reset))
            if match:
                try:
                    return max(float(match.group(1)), 1.0)
                except ValueError:
                    pass

    return max(float(fallback_seconds), 1.0)


def save_checkpoint(existing, partial_results, model_alias):
    partial_df = pd.DataFrame(partial_results)

    if existing.empty:
        base = existing
    else:
        base = remove_previous_model_rows(
            existing,
            model_alias,
        )

    combined = pd.concat(
        [base, partial_df],
        ignore_index=True,
    )

    combined.to_csv(PREDICTIONS_FILE, index=False)
    return combined

# ============================================================
# 10. RUN ONE MODEL
# ============================================================

def run_model(model_alias, benchmark, client, existing, sleep_seconds=0.02, max_retries=3, retry_wait=60.0, checkpoint_every=25):
    config = MODELS[model_alias]
    provider = config["provider"]
    model_name = config["model"]
    print(f"\nEvaluating model: {model_alias} ({provider}/{model_name})")
    print(f"Rows to evaluate: {len(benchmark):,}")
    results = []
    stopped_early = False
    stop_reason = None

    for position, (row_index, row) in enumerate(benchmark.iterrows(), start=1):
        prompt = build_prompt(row["question"])
        expected = row["expected_answer"]
        raw_output = None
        prediction = "error"
        error = None
        response_time = None

        for attempt in range(max_retries + 1):
            start_time = time.perf_counter()
            try:
                raw_output = call_model(client, provider, model_name, prompt)

                # Remove completed reasoning blocks
                if raw_output:
                    raw_output = re.sub(
                        r"<think>.*?</think>",
                        "",
                        raw_output,
                        flags=re.DOTALL | re.IGNORECASE,
                    ).strip()

                # If reasoning is still unfinished, treat as invalid
                if raw_output and "<think>" in raw_output.lower():
                    print(f"\n[REASONING OUTPUT row {row_index}]")
                    prediction = "invalid"
                    error = "Incomplete reasoning output"
                    break

                prediction = parse_label(raw_output)
                response_time = time.perf_counter() - start_time
                if prediction == "invalid":
                    print(f"\n[INVALID OUTPUT row {row_index}, attempt {attempt+1}/{max_retries+1}] {raw_output!r}")
                    error = "Invalid or empty model output"

                    if attempt < max_retries:
                        wait_seconds = min(5.0 * (attempt + 1), 15.0)
                        print(
                            f"Waiting {wait_seconds:.0f} seconds before retrying "
                            "the same row..."
                        )
                        time.sleep(wait_seconds)
                        continue

                break
            except Exception as exc:
                response_time = time.perf_counter() - start_time
                error = f"{type(exc).__name__}: {exc}"
                if not is_rate_limit_error(error):
                    print(f"\n[API ERROR row {row_index}] {error}")
                    break
                print(f"\n[RATE LIMIT row {row_index}, attempt {attempt+1}/{max_retries+1}]\n{error}")
                if is_hard_quota_error(error):
                    stopped_early = True
                    stop_reason = "Daily or hard quota reached."
                    break
                if attempt >= max_retries:
                    stopped_early = True
                    stop_reason = f"Rate limit remained after {max_retries} retries."
                    break
                fallback_wait = retry_wait * (attempt + 1)
                wait_seconds = get_server_retry_wait(exc, fallback_wait)
                print(
                    f"Waiting {wait_seconds:.0f} seconds before retrying "
                    "the same row..."
                )
                time.sleep(wait_seconds)

        if stopped_early:
            print(f"\nStopping early: {stop_reason}")
            break

        results.append({
            "model_alias": model_alias, "provider": provider, "model_name": model_name,
            "row_id": row_index, "station_key": row.get("station_key", None),
            "station_id": row.get("station_id", None), "expected_answer": expected,
            "prediction": prediction, "correct": prediction == expected,
            "response_time_seconds": response_time, "raw_output": raw_output,
            "error": error, "question": row["question"],
        })

        if checkpoint_every > 0 and len(results) % checkpoint_every == 0:
            save_checkpoint(existing, results, model_alias)
            print(f"  Checkpoint saved at {len(results)}/{len(benchmark)}")
        if position % 50 == 0 or position == len(benchmark):
            print(f"  Completed {position}/{len(benchmark)}")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if results:
        save_checkpoint(existing, results, model_alias)
    return pd.DataFrame(results), stopped_early, stop_reason


# ============================================================
# 11. METRICS
# ============================================================

def compute_metrics(results):
    metrics_rows = []
    per_class_rows = []

    if results.empty:
        return pd.DataFrame(), pd.DataFrame()

    required_cols = {
        "model_alias",
        "prediction",
        "expected_answer",
        "response_time_seconds",
    }

    missing = required_cols.difference(results.columns)

    if missing:
        raise ValueError(
            f"Prediction file is missing columns: {sorted(missing)}"
        )

    for model_alias in results["model_alias"].dropna().unique():
        model_all = results[
            results["model_alias"] == model_alias
        ].copy()

        model_valid = model_all[
            model_all["prediction"].isin(LABELS)
        ].copy()

        invalid_count = len(model_all) - len(model_valid)

        base_metrics = {
            "model_alias": model_alias,
            "n_total_rows": len(model_all),
            "n_evaluated": len(model_valid),
            "invalid_or_error_count": invalid_count,
            "invalid_or_error_rate": (
                invalid_count / len(model_all)
                if len(model_all) > 0
                else None
            ),
            "avg_response_time_seconds": (
                model_all["response_time_seconds"].mean()
            ),
        }

        if model_valid.empty:
            metrics_rows.append(
                {
                    **base_metrics,
                    "accuracy": None,
                    "balanced_accuracy": None,
                    "macro_precision": None,
                    "macro_recall": None,
                    "macro_f1": None,
                    "weighted_precision": None,
                    "weighted_recall": None,
                    "weighted_f1": None,
                    "cohen_kappa": None,
                }
            )
            continue

        y_true = model_valid["expected_answer"]
        y_pred = model_valid["prediction"]

        precision_macro, recall_macro, f1_macro, _ = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        )

        precision_weighted, recall_weighted, f1_weighted, _ = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=LABELS,
                average="weighted",
                zero_division=0,
            )
        )

        metrics_rows.append(
            {
                **base_metrics,
                "accuracy": accuracy_score(y_true, y_pred),
                "balanced_accuracy": balanced_accuracy_score(
                    y_true,
                    y_pred,
                ),
                "macro_precision": precision_macro,
                "macro_recall": recall_macro,
                "macro_f1": f1_macro,
                "weighted_precision": precision_weighted,
                "weighted_recall": recall_weighted,
                "weighted_f1": f1_weighted,
                "cohen_kappa": cohen_kappa_score(y_true, y_pred),
            }
        )

        report = classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )

        for label in LABELS:
            per_class_rows.append(
                {
                    "model_alias": model_alias,
                    "class": label,
                    "precision": report[label]["precision"],
                    "recall": report[label]["recall"],
                    "f1_score": report[label]["f1-score"],
                    "support": report[label]["support"],
                }
            )

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=LABELS,
        )

        matrix_df = pd.DataFrame(
            matrix,
            index=[f"true_{label}" for label in LABELS],
            columns=[f"pred_{label}" for label in LABELS],
        )

        matrix_df.to_csv(
            f"confusion_matrix_task3_anomaly_{model_alias}.csv"
        )

    metrics_df = pd.DataFrame(metrics_rows)

    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(
            by=["macro_f1", "accuracy"],
            ascending=False,
            na_position="last",
        )

    per_class_df = pd.DataFrame(per_class_rows)

    return metrics_df, per_class_df


# ============================================================
# 12. MAIN
# ============================================================

def main():
    args = parse_args()

    if args.list:
        print("Configured model aliases:")

        for alias, config in MODELS.items():
            print(
                f"  {alias:22s} -> "
                f"{config['provider']}/{config['model']}"
            )

        return

    if args.list_live_groq:
        client = make_client("groq")
        live_models = get_live_groq_model_ids(client)

        print("Groq models available to this API key:")

        for model_id in live_models:
            print(f"  {model_id}")

        return

    if args.list_live_openrouter:
        client = make_client("openrouter")
        live_models = get_live_openrouter_model_ids(client)

        print("OpenRouter models currently listed:")

        for model_id in live_models:
            print(f"  {model_id}")

        return

    if not args.model:
        print("No --model argument was provided.")
        print("\nAvailable aliases:")

        for alias in MODELS:
            print(f"  {alias}")

        print("\nExample:")
        print(
            "  python llm_eval_task3.py "
            "--model llama_3_1_8b --limit 6 "
            "--sample-mode balanced"
        )

        sys.exit(1)

    config = MODELS[args.model]
    client = make_client(config["provider"])

    # Verify Groq availability before loading the benchmark or making
    # hundreds of requests.
    if config["provider"] == "groq":
        verify_groq_model_available(
            client=client,
            model_alias=args.model,
            model_name=config["model"],
        )

    if config["provider"] == "openrouter":
        verify_openrouter_model_available(
            client=client,
            model_alias=args.model,
            model_name=config["model"],
        )

    benchmark = load_benchmark(
        limit=args.limit,
        sample_mode=args.sample_mode,
    )

    print("Benchmark loaded.")
    print(f"Rows selected: {len(benchmark):,}")
    print(benchmark["expected_answer"].value_counts())

    if args.limit is not None:
        print(
            f"Test limit applied: {len(benchmark)} rows "
            f"using {args.sample_mode} sampling."
        )

    existing = load_existing_predictions()
    existing = keep_latest_model_rows(existing, args.model)

    previously_saved = model_saved_row_count(
        existing,
        args.model,
    )

    if args.overwrite and args.resume:
        raise ValueError(
            "Use either --overwrite or --resume, not both."
        )

    if args.overwrite:
        existing = remove_previous_model_rows(existing, args.model)

    elif args.resume:
        completed_row_ids = get_completed_row_ids(
            existing,
            args.model,
        )

        before_resume = len(benchmark)
        benchmark = benchmark[
            ~benchmark.index.isin(completed_row_ids)
        ].copy()

        skipped = before_resume - len(benchmark)

        print(
            f"\nResume mode: found {len(completed_row_ids):,} "
            f"saved row_id values for '{args.model}'."
        )
        print(
            f"Skipping {skipped:,} completed benchmark rows; "
            f"{len(benchmark):,} rows remain."
        )

        if benchmark.empty:
            print(
                "All selected benchmark rows are already completed "
                "for this model."
            )
            return

    elif previously_saved > 0:
        print(
            f"\n'{args.model}' already has "
            f"{previously_saved:,} saved rows."
        )
        print("Use --resume to continue or --overwrite to restart.")
        return

    try:
        new_results, stopped_early, stop_reason = run_model(
            model_alias=args.model, benchmark=benchmark, client=client, existing=existing,
            sleep_seconds=args.sleep, max_retries=args.max_retries,
            retry_wait=args.retry_wait, checkpoint_every=args.checkpoint_every,
        )
    except KeyboardInterrupt:
        print(
            "\nRun interrupted by the user. Previously written checkpoints "
            "remain safe. Use --resume to continue."
        )
        return

    combined = pd.concat([existing, new_results], ignore_index=True)

    if combined.empty:
        print(
            "\nNo completed predictions were produced, "
            "so no prediction or metric files were overwritten."
        )
        metrics_df = pd.DataFrame()
        per_class_df = pd.DataFrame()
    else:
        combined.to_csv(PREDICTIONS_FILE, index=False)

        print(
            f"\nSaved: {PREDICTIONS_FILE} "
            f"({len(combined):,} total rows)"
        )

        metrics_df, per_class_df = compute_metrics(combined)

        metrics_df.to_csv(METRICS_FILE, index=False)
        per_class_df.to_csv(PER_CLASS_FILE, index=False)

        print(f"Saved: {METRICS_FILE}")
        print(f"Saved: {PER_CLASS_FILE}")

    if not new_results.empty and new_results["prediction"].isin(LABELS).any():
        print(
            "Saved: "
            f"confusion_matrix_task3_anomaly_{args.model}.csv"
        )
    else:
        print(
            "No confusion matrix was created for the current model "
            "because it produced no valid A/B/C predictions."
        )


    if stopped_early:
        print(f"\nRun stopped after {len(new_results):,}/{len(benchmark):,} completed rows.")
        print(f"Reason: {stop_reason}")
        print(
            "Completed rows were saved. Re-run later with --resume "
            "to continue, or --overwrite to restart this model."
        )

    print("\nModel ranking so far:")

    if metrics_df.empty:
        print("No metrics available.")
    else:
        print(metrics_df.to_string(index=False))

    if (
        combined.empty
        or "model_alias" not in combined.columns
    ):
        done_models = set()
    else:
        done_models = set(
            combined["model_alias"].dropna().unique()
        )

    remaining = [
        alias
        for alias in MODELS
        if alias not in done_models
    ]

    if remaining:
        print(
            "\nModels not yet run: "
            + ", ".join(remaining)
        )
        print(
            "Next: python llm_eval_task3.py "
            f"--model {remaining[0]}"
        )
    else:
        print("\nAll configured models have been run.")


if __name__ == "__main__":
    main()