"""
Build Task 4 Crash Disruption Sensitivity QA.

This standalone script creates matched crash-disruption contrastive examples.
It does not call any model API and does not modify the source master table.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
CANDIDATE_PAIR_TARGET = 6000
FINAL_QA_TARGET = 600
FINAL_PER_CLASS = 200

REPO_ROOT = Path(__file__).resolve().parents[3]
INPUT_CSV = Path(os.environ.get("TASK4_INPUT_CSV", REPO_ROOT / "data" / "master_table_station_hour_2022_2024_benchmark_labeled.csv"))
OUTPUT_DIR = Path(os.environ.get("TASK4_OUTPUT_DIR", Path(__file__).resolve().parent / "generated" / "crash_disruption"))


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


def snake_case(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def make_unique_columns(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    output = []
    for col in columns:
        base = snake_case(col)
        seen[base] = seen.get(base, -1) + 1
        output.append(base if seen[base] == 0 else f"{base}_{seen[base]}")
    return output


def safe_float(value, default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=None):
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def as_bool_series(series: pd.Series, default: bool = False) -> pd.Series:
    if series is None:
        return pd.Series(default)
    true_values = {"1", "true", "t", "yes", "y"}
    false_values = {"0", "false", "f", "no", "n", ""}

    def convert(value) -> bool:
        if pd.isna(value):
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(value)
        text = str(value).strip().lower()
        if text in true_values:
            return True
        if text in false_values:
            return False
        return default

    return series.map(convert)


def bool_value(value) -> bool:
    return bool(as_bool_series(pd.Series([value])).iloc[0])


def save_json(data, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def save_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def add_aliases(df: pd.DataFrame) -> pd.DataFrame:
    if "row_traffic_label" not in df.columns and "traffic_change_label" in df.columns:
        df["row_traffic_label"] = df["traffic_change_label"]
        print("Alias created: traffic_change_label -> row_traffic_label")
    if "station_name" not in df.columns and "name" in df.columns:
        df["station_name"] = df["name"]
        print("Alias created: name -> station_name")
    if "rain_mm" not in df.columns and "precipitation_mm" in df.columns:
        df["rain_mm"] = df["precipitation_mm"]
        print("Alias created: precipitation_mm -> rain_mm")
    return df


def month_to_season(month) -> str:
    m = safe_int(month)
    if m in {12, 1, 2}:
        return "summer"
    if m in {3, 4, 5}:
        return "autumn"
    if m in {6, 7, 8}:
        return "winter"
    if m in {9, 10, 11}:
        return "spring"
    return "unknown"


def add_helpers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["date"], errors="coerce") if "date" in df.columns else pd.NaT
    if "month" not in df.columns:
        df["month"] = df["datetime"].dt.month
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df["season"] = df["month"].map(month_to_season)

    if "is_weekend" in df.columns:
        weekend = as_bool_series(df["is_weekend"])
        df["day_type"] = np.where(weekend, "weekend", "weekday")
    elif "day_of_week" in df.columns:
        dow = df["day_of_week"].astype(str).str.strip().str.lower()
        weekend_names = {"saturday", "sun", "sunday", "sat"}
        dow_num = pd.to_numeric(df["day_of_week"], errors="coerce")
        is_weekend = dow.isin(weekend_names) | dow_num.isin([5, 6, 7])
        df["day_type"] = np.where(is_weekend, "weekend", "weekday")
    else:
        df["day_type"] = "unknown"

    df["rain_mm"] = pd.to_numeric(df["rain_mm"], errors="coerce")
    df["rain_bucket"] = np.select(
        [df["rain_mm"].eq(0), df["rain_mm"].gt(0) & df["rain_mm"].le(1)],
        ["no_rain", "light_rain"],
        default="moderate_heavy_rain",
    )
    df.loc[df["rain_mm"].isna(), "rain_bucket"] = "unknown"

    temp_col = None
    for candidate in ["temperature_2m_c", "temperature", "temp_value", "temp_c"]:
        if candidate in df.columns:
            temp_col = candidate
            break
    df["temperature_value"] = pd.to_numeric(df[temp_col], errors="coerce") if temp_col else np.nan
    df["temperature_bucket"] = pd.cut(
        df["temperature_value"],
        bins=[-np.inf, 10, 20, 30, np.inf],
        labels=["cold", "mild", "warm", "hot"],
        right=False,
    ).astype("object")
    df.loc[df["temperature_value"].isna(), "temperature_bucket"] = "unknown"

    for col in ["has_event_3km", "public_holiday", "school_holiday"]:
        df[col] = as_bool_series(df[col]) if col in df.columns else False
    df["crash_count"] = pd.to_numeric(df["crash_count"], errors="coerce").fillna(0)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    return df


def load_data() -> pd.DataFrame:
    section("Load and prepare data")
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df.columns = make_unique_columns(list(df.columns))
    df = add_aliases(df)
    require_columns(
        df,
        ["station_key", "date", "hour", "volume", "rain_mm", "crash_count", "has_event_3km", "public_holiday", "school_holiday"],
    )
    df = add_helpers(df)
    print(f"Loaded rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    return df


def build_pools(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    section("Build Scenario A/B pools")
    base = (
        df["station_key"].notna()
        & df["hour"].notna()
        & df["day_type"].isin(["weekday", "weekend"])
        & df["volume"].notna()
        & df["rain_mm"].notna()
        & ~df["has_event_3km"]
        & ~df["public_holiday"]
        & ~df["school_holiday"]
    )
    normal = pd.Series(True, index=df.index)
    if "row_traffic_label" in df.columns:
        normal = df["row_traffic_label"].astype(str).str.lower().eq("normal")

    a_pool = df.loc[base & normal & df["crash_count"].eq(0)].copy()
    b_pool = df.loc[base & df["crash_count"].ge(1)].copy()
    print(f"Scenario A rows: {len(a_pool):,}")
    print(f"Scenario B rows: {len(b_pool):,}")
    return a_pool, b_pool


MATCH_LEVELS = [
    ("same_month_temp", ["station_key", "hour", "day_type", "rain_bucket", "temperature_bucket", "month"]),
    ("same_season_temp", ["station_key", "hour", "day_type", "rain_bucket", "temperature_bucket", "season"]),
    ("same_hard_keys_temp", ["station_key", "hour", "day_type", "rain_bucket", "temperature_bucket"]),
]


def group_indices(df: pd.DataFrame, keys: list[str]) -> dict[tuple, np.ndarray]:
    groups: dict[tuple, np.ndarray] = {}
    for key, group in df.groupby(keys, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        groups[key] = group.index.to_numpy()
    return groups


def key_for(row: pd.Series, keys: list[str]) -> tuple:
    return tuple(row.get(key) for key in keys)


def choose_candidate(
    b_row: pd.Series,
    a_pool: pd.DataFrame,
    grouped: dict[str, dict[tuple, np.ndarray]],
    rng: np.random.Generator,
) -> tuple[int | None, str | None]:
    for level, keys in MATCH_LEVELS:
        idxs = grouped[level].get(key_for(b_row, keys))
        if idxs is None or len(idxs) == 0:
            continue
        candidates = a_pool.loc[idxs].copy()
        if "datetime" in candidates.columns and pd.notna(b_row.get("datetime")):
            candidates["_date_distance"] = (candidates["datetime"] - b_row["datetime"]).abs()
            candidates = candidates.sort_values("_date_distance", na_position="last")
            return int(candidates.iloc[0].name), level
        return int(rng.choice(idxs)), level
    return None, None


def build_candidate_pairs(a_pool: pd.DataFrame, b_pool: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    section("Match candidate pairs")
    rng = np.random.default_rng(RANDOM_SEED)
    grouped = {level: group_indices(a_pool, keys) for level, keys in MATCH_LEVELS}
    b_order = b_pool.sample(frac=1, random_state=RANDOM_SEED)
    rows = []
    dropped_invalid_volume = 0
    total_valid_before_sampling = 0

    for _, b_row in b_order.iterrows():
        a_idx, match_level = choose_candidate(b_row, a_pool, grouped, rng)
        if a_idx is None:
            continue
        a_row = a_pool.loc[a_idx]
        volume_a = safe_float(a_row.get("volume"))
        volume_b = safe_float(b_row.get("volume"))
        if volume_a is None or volume_b is None or volume_a <= 0:
            dropped_invalid_volume += 1
            continue

        delta_pct = (volume_b - volume_a) / volume_a
        if delta_pct > 0.10:
            correct_answer, correct_option = "increase", "A"
        elif delta_pct < -0.10:
            correct_answer, correct_option = "decrease", "B"
        else:
            correct_answer, correct_option = "normal", "C"

        total_valid_before_sampling += 1
        rows.append(
            {
                "question_id": f"crash_disruption_{len(rows) + 1:06d}",
                "context_type": "crash_disruption",
                "station_key": b_row.get("station_key"),
                "station_name": text_value(b_row, ["station_name", "name"]),
                "road_name": text_value(b_row, ["road_name"]),
                "suburb": text_value(b_row, ["suburb"]),
                "lga": text_value(b_row, ["lga"]),
                "date_A": a_row.get("date"),
                "date_B": b_row.get("date"),
                "hour": safe_int(b_row.get("hour")),
                "day_type": b_row.get("day_type"),
                "month_A": safe_int(a_row.get("month")),
                "month_B": safe_int(b_row.get("month")),
                "season_A": a_row.get("season"),
                "season_B": b_row.get("season"),
                "rain_mm_A": safe_float(a_row.get("rain_mm")),
                "rain_mm_B": safe_float(b_row.get("rain_mm")),
                "rain_bucket_A": a_row.get("rain_bucket"),
                "rain_bucket_B": b_row.get("rain_bucket"),
                "rain_bucket": b_row.get("rain_bucket"),
                "temperature_A": safe_float(a_row.get("temperature_value")),
                "temperature_B": safe_float(b_row.get("temperature_value")),
                "temperature_bucket_A": a_row.get("temperature_bucket"),
                "temperature_bucket_B": b_row.get("temperature_bucket"),
                "crash_count_A": safe_float(a_row.get("crash_count"), 0),
                "crash_count_B": safe_float(b_row.get("crash_count"), 0),
                "has_event_3km_A": bool_value(a_row.get("has_event_3km")),
                "has_event_3km_B": bool_value(b_row.get("has_event_3km")),
                "public_holiday_A": bool_value(a_row.get("public_holiday")),
                "public_holiday_B": bool_value(b_row.get("public_holiday")),
                "school_holiday_A": bool_value(a_row.get("school_holiday")),
                "school_holiday_B": bool_value(b_row.get("school_holiday")),
                "volume_A": volume_a,
                "volume_B": volume_b,
                "delta_pct": delta_pct,
                "correct_answer": correct_answer,
                "correct_option": correct_option,
                "gold_answer_type": "option",
                "match_level": match_level,
                "land_use_description": land_use_description(b_row),
                "poi_description": poi_description(b_row),
            }
        )
        if len(rows) >= CANDIDATE_PAIR_TARGET:
            break

    pairs = pd.DataFrame(rows)
    stats = {
        "total_matched_candidate_pairs_before_candidate_sampling": int(total_valid_before_sampling),
        "rows_dropped_due_to_missing_volume_or_zero_volume_A": int(dropped_invalid_volume),
    }
    if len(pairs) < CANDIDATE_PAIR_TARGET:
        print(f"WARNING: only {len(pairs):,} valid candidate pairs available.")
    else:
        print(f"Saved candidate pair target reached: {len(pairs):,}")
    return pairs, stats


def text_value(row: pd.Series, candidates: list[str], default: str = "unknown") -> str:
    for col in candidates:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col]).strip()
    return default


def land_pct(value) -> str | None:
    pct = safe_float(value)
    if pct is None:
        return None
    if 0 <= pct <= 1:
        pct *= 100
    return f"{pct:.2f}%"


def land_use_description(row: pd.Series) -> str:
    cat = text_value(row, ["landuse_dom_category_500m", "landuse_top1_category_500m"], "")
    ratio_col = "landuse_dom_ratio_500m" if "landuse_dom_ratio_500m" in row.index else "landuse_top1_ratio_500m"
    pct = land_pct(row.get(ratio_col)) if ratio_col in row.index else None
    if cat and pct:
        return f"{cat} land use ({pct})"
    if cat:
        return f"{cat} land use"
    return "urban mixed land use"


def poi_description(row: pd.Series) -> str:
    labels = [
        ("food", "poi_food_count_500m"),
        ("education", "poi_education_count_500m"),
        ("healthcare", "poi_healthcare_count_500m"),
        ("public transport", "poi_public_transport_count_500m"),
        ("leisure", "poi_leisure_count_500m"),
        ("tourism", "poi_tourism_count_500m"),
        ("shops", "poi_shop_count_500m"),
    ]
    parts = []
    for label, col in labels:
        value = safe_int(row.get(col), 0) if col in row.index else 0
        if value > 0:
            parts.append(f"{value} {label}")
    return ", ".join(parts) if parts else "limited mapped POIs"


def weather_desc(row: pd.Series, suffix: str) -> str:
    rain_map = {
        "no_rain": "no-rain condition",
        "light_rain": "light-rain condition",
        "moderate_heavy_rain": "moderate/heavy-rain condition",
    }
    temp_map = {
        "cold": "cold temperature category",
        "mild": "mild temperature category",
        "warm": "warm temperature category",
        "hot": "hot temperature category",
    }
    rain_bucket = row.get("rain_bucket") or row.get(f"rain_bucket_{suffix}")
    temp_bucket = row.get(f"temperature_bucket_{suffix}")
    rain_text = rain_map.get(str(rain_bucket), "unknown rain condition")
    temp_text = temp_map.get(str(temp_bucket), "unknown temperature category")
    return f"{rain_text}, {temp_text}"


def event_desc(row: pd.Series, suffix: str) -> str:
    return "nearby event" if bool_value(row.get(f"has_event_3km_{suffix}")) else "no nearby event"


def crash_desc(row: pd.Series, suffix: str) -> str:
    count = safe_int(row.get(f"crash_count_{suffix}"), 0)
    if count <= 0:
        return "no nearby crash"
    if count == 1:
        return "1 nearby crash within 3 km"
    return f"{count} nearby crashes within 3 km"


def holiday_desc(row: pd.Series, suffix: str) -> str:
    public = bool_value(row.get(f"public_holiday_{suffix}"))
    school = bool_value(row.get(f"school_holiday_{suffix}"))
    if public and school:
        return "public and school holiday"
    if public:
        return "public holiday"
    if school:
        return "school holiday"
    return "non-holiday"


OPTIONS_TEXT = (
    "(A) Higher than Scenario A, if traffic volume is more than 10% higher.\n"
    "(B) Lower than Scenario A, if traffic volume is more than 10% lower.\n"
    "(C) Similar to Scenario A, if the difference is within +/-10%."
)
OPTION_LABEL_MAP = {"A": "increase", "B": "decrease", "C": "normal"}


def make_question(row: pd.Series) -> str:
    return f"""You are given two traffic scenarios at the same monitoring station.

Station: {row['station_name']}
Road: {row['road_name']}
Area: {row['suburb']}, {row['lga']}
Nearby urban context: {row['land_use_description']}, with nearby POIs including {row['poi_description']}.

Both scenarios occur on {row['day_type']} at {int(row['hour'])}:00.

Scenario A:
- Weather: {weather_desc(row, 'A')}
- Event: {event_desc(row, 'A')}
- Crash: {crash_desc(row, 'A')}
- Holiday: {holiday_desc(row, 'A')}

Scenario B:
- Weather: {weather_desc(row, 'B')}
- Event: {event_desc(row, 'B')}
- Crash: {crash_desc(row, 'B')}
- Holiday: {holiday_desc(row, 'B')}

Compared with Scenario A, the traffic volume in Scenario B is most likely to be:
{OPTIONS_TEXT}

Choose exactly one option from A, B, or C.

Do not explain your reasoning.

Your entire response must be exactly one line:
Final answer: <A/B/C>"""


def sample_final_qa(pairs: pd.DataFrame) -> pd.DataFrame:
    section("Sample final eval600 QA")
    rng = np.random.default_rng(RANDOM_SEED)
    sampled_parts = []
    for option in ["A", "B", "C"]:
        group = pairs[pairs["correct_option"] == option]
        n = min(FINAL_PER_CLASS, len(group))
        if n < FINAL_PER_CLASS:
            print(f"WARNING: class {option} has only {len(group):,} samples; using {n:,}.")
        if n:
            sampled_parts.append(group.sample(n=n, random_state=RANDOM_SEED))
    final = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else pd.DataFrame()
    if not final.empty:
        final = final.iloc[rng.permutation(len(final))].reset_index(drop=True)
    final["question"] = final.apply(make_question, axis=1)
    final["options"] = OPTIONS_TEXT
    final["option_label_map"] = json.dumps(OPTION_LABEL_MAP, sort_keys=True)
    return final


def sanity_check(pairs: pd.DataFrame) -> None:
    checks = {
        "crash_count_A == 0": pairs["crash_count_A"].eq(0),
        "crash_count_B >= 1": pairs["crash_count_B"].ge(1),
        "has_event_3km_A == False": ~pairs["has_event_3km_A"],
        "has_event_3km_B == False": ~pairs["has_event_3km_B"],
        "public_holiday_A == False": ~pairs["public_holiday_A"],
        "public_holiday_B == False": ~pairs["public_holiday_B"],
        "school_holiday_A == False": ~pairs["school_holiday_A"],
        "school_holiday_B == False": ~pairs["school_holiday_B"],
        "rain_bucket_A == rain_bucket_B": pairs["rain_bucket_A"].eq(pairs["rain_bucket_B"]),
        "temperature_bucket_A == temperature_bucket_B": pairs["temperature_bucket_A"].eq(pairs["temperature_bucket_B"]),
    }
    failures = {name: int((~mask).sum()) for name, mask in checks.items() if not bool(mask.all())}
    if failures:
        print("Sanity check failures:")
        print(failures)
        raise AssertionError(f"Crash disruption sanity checks failed: {failures}")


def write_outputs(
    candidate_pairs: pd.DataFrame,
    final_qa: pd.DataFrame,
    a_pool: pd.DataFrame,
    b_pool: pd.DataFrame,
    extra_stats: dict,
) -> None:
    section("Write outputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidate_path = OUTPUT_DIR / "crash_disruption_candidate_pairs_6000.csv"
    csv_path = OUTPUT_DIR / "crash_disruption_qa_eval600.csv"
    jsonl_path = OUTPUT_DIR / "crash_disruption_qa_eval600.jsonl"
    stats_path = OUTPUT_DIR / "crash_disruption_pair_stats.json"
    readme_path = OUTPUT_DIR / "crash_disruption_readme.md"

    candidate_pairs.to_csv(candidate_path, index=False)
    final_cols = [
        "question_id", "context_type", "question", "options", "option_label_map",
        "correct_answer", "correct_option", "gold_answer_type", "station_key",
        "station_name", "road_name", "suburb", "lga", "date_A", "date_B", "hour",
        "day_type", "month_A", "month_B", "season_A", "season_B", "rain_mm_A",
        "rain_mm_B", "rain_bucket", "temperature_A", "temperature_B",
        "temperature_bucket_A", "temperature_bucket_B", "crash_count_A",
        "crash_count_B", "volume_A", "volume_B", "delta_pct", "match_level",
    ]
    final_qa[final_cols].to_csv(csv_path, index=False)
    save_jsonl(final_qa[final_cols].to_dict("records"), jsonl_path)

    stats = {
        "task_name": "Crash Disruption Sensitivity Task",
        "scenario_A_rows": int(len(a_pool)),
        "scenario_B_rows": int(len(b_pool)),
        **extra_stats,
        "total_saved_candidate_pairs": int(len(candidate_pairs)),
        "candidate_label_distribution": candidate_pairs["correct_option"].value_counts().sort_index().to_dict(),
        "final_QA_label_distribution": final_qa["correct_option"].value_counts().sort_index().to_dict(),
        "final_rain_bucket_distribution": final_qa["rain_bucket"].value_counts().to_dict(),
        "final_temperature_bucket_distribution": final_qa["temperature_bucket_A"].value_counts().to_dict(),
        "final_temperature_bucket_mismatch_count": int((final_qa["temperature_bucket_A"] != final_qa["temperature_bucket_B"]).sum()),
        "final_crash_count_B_distribution": final_qa["crash_count_B"].value_counts().sort_index().to_dict(),
        "match_level_distribution": candidate_pairs["match_level"].value_counts().to_dict(),
        "output_paths": {
            "candidate_pairs_csv": str(candidate_path),
            "qa_eval600_csv": str(csv_path),
            "qa_eval600_jsonl": str(jsonl_path),
            "stats_json": str(stats_path),
            "readme": str(readme_path),
            "script": str(OUTPUT_DIR / "build_crash_disruption_sensitivity_qa.py"),
        },
    }
    save_json(stats, stats_path)
    readme_path.write_text(README_TEXT, encoding="utf-8")

    source_script = Path(__file__).resolve()
    dest_script = OUTPUT_DIR / "build_crash_disruption_sensitivity_qa.py"
    if source_script != dest_script:
        shutil.copy2(source_script, dest_script)

    print(f"Candidate pairs: {candidate_path}")
    print(f"Final QA CSV: {csv_path}")
    print(f"Final QA JSONL: {jsonl_path}")
    print(f"Stats: {stats_path}")


README_TEXT = """# Crash Disruption Sensitivity Task

## Objective
Evaluate whether models can reason about traffic-volume differences under nearby crash conditions while other contextual factors are controlled.

## Crash as a discrete disruption
Crash is treated as a discrete disruption variable. Scenario A uses no-crash normal reference rows, while Scenario B uses comparable rows with at least one nearby crash.

## Scenario rules
Scenario A: crash_count == 0, no nearby event, non-holiday, and row traffic label normal where available.
Scenario B: crash_count >= 1, no nearby event, and non-holiday.

## Rain bucket control
Scenario A and Scenario B are matched within the same rain bucket: no_rain, light_rain, or moderate_heavy_rain.

## Temperature control
Temperature bucket is a hard within-pair matching key and is not displayed in the prompt.

## Ground truth
delta_pct = (volume_B - volume_A) / volume_A.
A = higher / increase if delta_pct > 0.10.
B = lower / decrease if delta_pct < -0.10.
C = similar / normal otherwise.

## Output files
- crash_disruption_candidate_pairs_6000.csv
- crash_disruption_qa_eval600.csv
- crash_disruption_qa_eval600.jsonl
- crash_disruption_pair_stats.json
- crash_disruption_readme.md
- build_crash_disruption_sensitivity_qa.py

## Important limitations
Rainfall is only controlled through within-pair rain bucket matching. The task does not attempt to estimate the causal effect of rainfall, nor does it force equal sample sizes across rain buckets. The objective is to evaluate crash-related contextual reasoning while reducing major weather-related confounding.
"""


def main() -> None:
    section("Crash Disruption Sensitivity Task")
    df = load_data()
    a_pool, b_pool = build_pools(df)
    candidate_pairs, extra_stats = build_candidate_pairs(a_pool, b_pool)
    if candidate_pairs.empty:
        raise RuntimeError("No candidate pairs were created.")
    sanity_check(candidate_pairs)
    final_qa = sample_final_qa(candidate_pairs)
    write_outputs(candidate_pairs, final_qa, a_pool, b_pool, extra_stats)
    section("Summary")
    print(f"Scenario A rows: {len(a_pool):,}")
    print(f"Scenario B rows: {len(b_pool):,}")
    print(f"Candidate pairs: {len(candidate_pairs):,}")
    print(f"Final QA rows: {len(final_qa):,}")
    print(f"Final label distribution: {final_qa['correct_option'].value_counts().sort_index().to_dict()}")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()




