"""
Build Task 4 Planned Activity / Calendar Contrast QA.

This standalone script creates planned-activity/calendar contrast examples.
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
OUTPUT_DIR = Path(os.environ.get("TASK4_OUTPUT_DIR", Path(__file__).resolve().parent / "generated" / "planned_activity_calendar"))


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


def bool_series(series: pd.Series, default: bool = False) -> pd.Series:
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
    return bool(bool_series(pd.Series([value])).iloc[0])


def text_value(row: pd.Series, candidates: list[str], default: str = "unknown") -> str:
    for col in candidates:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col]).strip()
    return default


def clean_event_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "unknown", "null"}:
        return ""
    return text[:80]


def save_json(data, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def save_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


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


def first_existing(columns: set[str], candidates: list[str]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None


def add_aliases(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    info = {}
    if "station_name" not in df.columns and "name" in df.columns:
        df["station_name"] = df["name"]
        info["station_name_alias"] = "name"
    if "row_traffic_label" not in df.columns and "traffic_change_label" in df.columns:
        df["row_traffic_label"] = df["traffic_change_label"]
        info["row_traffic_label_alias"] = "traffic_change_label"
    if "rain_mm" not in df.columns and "precipitation_mm" in df.columns:
        df["rain_mm"] = df["precipitation_mm"]
        info["rain_mm_alias"] = "precipitation_mm"
    event_col = first_existing(
        set(df.columns),
        ["event_count", "event_count_3km", "event_cnt", "event_cnt_3km", "nearby_event_count"],
    )
    if event_col is None:
        df["event_count"] = 0
        event_col = "event_count"
    elif event_col != "event_count":
        df["event_count"] = df[event_col]
    info["event_count_column_used"] = event_col
    return df, info


def require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def add_helpers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
    if "month" not in df.columns:
        df["month"] = df["datetime"].dt.month
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df["season"] = df["month"].map(month_to_season)

    if "is_weekend" in df.columns:
        df["day_type"] = np.where(bool_series(df["is_weekend"]), "weekend", "weekday")
    elif "day_of_week" in df.columns:
        dow_num = pd.to_numeric(df["day_of_week"], errors="coerce")
        dow_text = df["day_of_week"].astype(str).str.strip().str.lower()
        is_weekend = dow_num.isin([5, 6, 7]) | dow_text.isin(["sat", "saturday", "sun", "sunday"])
        df["day_type"] = np.where(is_weekend, "weekend", "weekday")
    else:
        df["day_type"] = np.where(df["datetime"].dt.dayofweek.ge(5), "weekend", "weekday")

    for col in ["public_holiday", "school_holiday", "has_event_3km"]:
        df[col] = bool_series(df[col]) if col in df.columns else False

    df["event_count"] = pd.to_numeric(df["event_count"], errors="coerce").fillna(0).clip(lower=0)
    df["nearest_event_type"] = df["nearest_event_type"].map(clean_event_text) if "nearest_event_type" in df.columns else ""
    df["nearest_event_name"] = df["nearest_event_name"].map(clean_event_text) if "nearest_event_name" in df.columns else ""

    df["rain_mm"] = pd.to_numeric(df["rain_mm"], errors="coerce")
    df["rain_bucket"] = np.select(
        [df["rain_mm"].eq(0), df["rain_mm"].gt(0) & df["rain_mm"].le(1)],
        ["no_rain", "light_rain"],
        default="moderate_heavy_rain",
    )
    df.loc[df["rain_mm"].isna(), "rain_bucket"] = "unknown"

    temp_col = first_existing(set(df.columns), ["temperature_2m_c", "temperature", "temp_value", "temp_c"])
    df["temperature_value"] = pd.to_numeric(df[temp_col], errors="coerce") if temp_col else np.nan
    df["temperature_bucket"] = pd.cut(
        df["temperature_value"],
        bins=[-np.inf, 10, 20, 30, np.inf],
        labels=["cold", "mild", "warm", "hot"],
        right=False,
    ).astype("object")
    df.loc[df["temperature_value"].isna(), "temperature_bucket"] = "unknown"

    df["crash_count"] = pd.to_numeric(df["crash_count"], errors="coerce").fillna(0)
    df["crash_bucket"] = np.where(df["crash_count"].eq(0), "no_crash", "minor_crash_context")
    df.loc[df["crash_count"].gt(1), "crash_bucket"] = "excluded_crash_gt_1"
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    return df


def load_data() -> tuple[pd.DataFrame, dict]:
    section("Load and prepare data")
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df.columns = make_unique_columns(list(df.columns))
    df, alias_info = add_aliases(df)
    require_columns(df, ["station_key", "date", "hour", "volume", "rain_mm", "crash_count"])
    df = add_helpers(df)
    print(f"Loaded rows: {len(df):,}")
    return df, alias_info


def valid_pool(df: pd.DataFrame, crash_limit: int) -> pd.DataFrame:
    mask = (
        df["station_key"].notna()
        & df["date"].notna()
        & df["hour"].notna()
        & df["day_type"].isin(["weekday", "weekend"])
        & df["volume"].notna()
        & df["volume"].gt(0)
        & df["rain_bucket"].ne("unknown")
        & df["temperature_bucket"].ne("unknown")
        & df["crash_count"].le(crash_limit)
    )
    return df.loc[mask].copy()


MATCH_LEVELS = [
    ("same_month", ["station_key", "hour", "day_type", "rain_bucket", "temperature_bucket", "crash_bucket", "month"]),
    ("same_season", ["station_key", "hour", "day_type", "rain_bucket", "temperature_bucket", "crash_bucket", "season"]),
    ("same_hard_keys", ["station_key", "hour", "day_type", "rain_bucket", "temperature_bucket", "crash_bucket"]),
]


def activity_signature(row: pd.Series) -> tuple:
    return (
        bool_value(row.get("public_holiday")),
        bool_value(row.get("school_holiday")),
        bool_value(row.get("has_event_3km")),
        safe_int(row.get("event_count"), 0),
        clean_event_text(row.get("nearest_event_type")),
        clean_event_text(row.get("nearest_event_name")),
    )


def activity_difference(a_row: pd.Series, b_row: pd.Series) -> str:
    diffs = []
    if bool_value(a_row.get("public_holiday")) != bool_value(b_row.get("public_holiday")):
        diffs.append("public_holiday_changed")
    if bool_value(a_row.get("school_holiday")) != bool_value(b_row.get("school_holiday")):
        diffs.append("school_holiday_changed")
    if bool_value(a_row.get("has_event_3km")) != bool_value(b_row.get("has_event_3km")):
        diffs.append("event_presence_changed")
    if safe_int(a_row.get("event_count"), 0) != safe_int(b_row.get("event_count"), 0):
        diffs.append("event_count_changed")
    a_type = clean_event_text(a_row.get("nearest_event_type"))
    b_type = clean_event_text(b_row.get("nearest_event_type"))
    if (a_type or b_type) and a_type != b_type:
        diffs.append("event_type_changed")
    a_name = clean_event_text(a_row.get("nearest_event_name"))
    b_name = clean_event_text(b_row.get("nearest_event_name"))
    if (a_name or b_name) and a_name != b_name:
        diffs.append("event_name_changed")
    return "+".join(diffs)


def group_indices(df: pd.DataFrame, keys: list[str]) -> dict[tuple, np.ndarray]:
    out = {}
    for key, group in df.groupby(keys, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        out[key] = group.index.to_numpy()
    return out


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


def row_to_pair(a_row: pd.Series, b_row: pd.Series, match_level: str, pair_id: int) -> dict | None:
    diff_type = activity_difference(a_row, b_row)
    if not diff_type:
        return None
    volume_a = safe_float(a_row.get("volume"))
    volume_b = safe_float(b_row.get("volume"))
    if volume_a is None or volume_b is None or volume_a <= 0:
        return None
    delta_pct = (volume_b - volume_a) / volume_a
    if delta_pct > 0.10:
        correct_answer, correct_option = "increase", "A"
    elif delta_pct < -0.10:
        correct_answer, correct_option = "decrease", "B"
    else:
        correct_answer, correct_option = "normal", "C"
    return {
        "question_id": f"planned_activity_{pair_id:06d}",
        "context_type": "planned_activity_calendar",
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
        "rain_bucket_A": a_row.get("rain_bucket"),
        "rain_bucket_B": b_row.get("rain_bucket"),
        "temperature_bucket_A": a_row.get("temperature_bucket"),
        "temperature_bucket_B": b_row.get("temperature_bucket"),
        "crash_count_A": safe_int(a_row.get("crash_count"), 0),
        "crash_count_B": safe_int(b_row.get("crash_count"), 0),
        "crash_bucket_A": a_row.get("crash_bucket"),
        "crash_bucket_B": b_row.get("crash_bucket"),
        "public_holiday_A": bool_value(a_row.get("public_holiday")),
        "public_holiday_B": bool_value(b_row.get("public_holiday")),
        "school_holiday_A": bool_value(a_row.get("school_holiday")),
        "school_holiday_B": bool_value(b_row.get("school_holiday")),
        "has_event_3km_A": bool_value(a_row.get("has_event_3km")),
        "has_event_3km_B": bool_value(b_row.get("has_event_3km")),
        "event_count_A": safe_int(a_row.get("event_count"), 0),
        "event_count_B": safe_int(b_row.get("event_count"), 0),
        "nearest_event_type_A": clean_event_text(a_row.get("nearest_event_type")),
        "nearest_event_type_B": clean_event_text(b_row.get("nearest_event_type")),
        "nearest_event_name_A": clean_event_text(a_row.get("nearest_event_name")),
        "nearest_event_name_B": clean_event_text(b_row.get("nearest_event_name")),
        "activity_difference_type": diff_type,
        "volume_A": volume_a,
        "volume_B": volume_b,
        "delta_pct": delta_pct,
        "correct_answer": correct_answer,
        "correct_option": correct_option,
        "match_level": match_level,
        "gold_answer_type": "option",
        "land_use_description": land_use_description(b_row),
        "poi_description": poi_description(b_row),
    }


def build_pairs_from_pool(pool: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    seen_pairs = set()
    pair_id = 1
    for match_level, keys in MATCH_LEVELS:
        groups = group_indices(pool, keys)
        group_keys = list(groups.keys())
        rng.shuffle(group_keys)
        for group_key in group_keys:
            idxs = groups[group_key]
            if len(idxs) < 2:
                continue
            take = min(len(idxs), 80)
            sampled = rng.choice(idxs, size=take, replace=False) if len(idxs) > take else idxs
            group = pool.loc[sampled].copy()
            sigs = group.apply(activity_signature, axis=1)
            if sigs.nunique() < 2:
                continue
            order = group.index.to_numpy().copy()
            rng.shuffle(order)
            attempts = min(400, len(order) * 10)
            for _ in range(attempts):
                a_idx, b_idx = rng.choice(order, size=2, replace=False)
                pair_key = tuple(sorted((int(a_idx), int(b_idx))))
                if pair_key in seen_pairs:
                    continue
                a_row = pool.loc[a_idx]
                b_row = pool.loc[b_idx]
                pair = row_to_pair(a_row, b_row, match_level, pair_id)
                if pair is None:
                    continue
                seen_pairs.add(pair_key)
                rows.append(pair)
                pair_id += 1
                if len(rows) >= CANDIDATE_PAIR_TARGET:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def build_candidate_pairs(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    section("Build candidate pairs")
    pool0 = valid_pool(df, crash_limit=0)
    print(f"Valid pool with crash_count == 0: {len(pool0):,}")
    pairs = build_pairs_from_pool(pool0)
    fallback_used = False
    pool_used = pool0
    if len(pairs) < CANDIDATE_PAIR_TARGET:
        print("WARNING: crash_count == 0 pool did not reach 6000 pairs; falling back to crash_count <= 1.")
        pool1 = valid_pool(df, crash_limit=1)
        pool_used = pool1
        pairs = build_pairs_from_pool(pool1)
        fallback_used = True
    stats = {
        "pool_rows_used": int(len(pool_used)),
        "fallback_to_crash_count_le_1": fallback_used,
        "candidate_pair_count": int(len(pairs)),
    }
    print(f"Candidate pairs: {len(pairs):,}")
    return pairs, stats


RAIN_DISPLAY = {
    "no_rain": "no-rain condition",
    "light_rain": "light-rain condition",
    "moderate_heavy_rain": "moderate/heavy-rain condition",
}
TEMP_DISPLAY = {
    "cold": "cold temperature category",
    "mild": "mild temperature category",
    "warm": "warm temperature category",
    "hot": "hot temperature category",
}
OPTIONS_TEXT = (
    "(A) Higher than Scenario A, if traffic volume is more than 10% higher.\n"
    "(B) Lower than Scenario A, if traffic volume is more than 10% lower.\n"
    "(C) Similar to Scenario A, if the difference is within +/-10%."
)
OPTION_LABEL_MAP = {"A": "increase", "B": "decrease", "C": "normal"}


def weather_desc(row: pd.Series, suffix: str) -> str:
    rain = RAIN_DISPLAY.get(str(row.get(f"rain_bucket_{suffix}")), "unknown rain condition")
    temp = TEMP_DISPLAY.get(str(row.get(f"temperature_bucket_{suffix}")), "unknown temperature category")
    return f"{rain}, {temp}"


def event_desc(row: pd.Series, suffix: str) -> str:
    count = safe_int(row.get(f"event_count_{suffix}"), 0)
    name = clean_event_text(row.get(f"nearest_event_name_{suffix}"))
    etype = clean_event_text(row.get(f"nearest_event_type_{suffix}"))
    has_event = bool_value(row.get(f"has_event_3km_{suffix}")) or count > 0
    if not has_event:
        return "no nearby event"
    text = f"{count} nearby event(s) within 3 km"
    if name:
        text += f"; nearest event: {name}"
    if etype:
        text += f"; type: {etype}" if name else f"; nearest event type: {etype}"
    return text


def holiday_desc(row: pd.Series, suffix: str) -> str:
    public = bool_value(row.get(f"public_holiday_{suffix}"))
    school = bool_value(row.get(f"school_holiday_{suffix}"))
    if public and school:
        return "public holiday and school holiday"
    if public:
        return "public holiday"
    if school:
        return "school holiday"
    return "non-holiday"


def crash_desc(row: pd.Series, suffix: str) -> str:
    count = safe_int(row.get(f"crash_count_{suffix}"), 0)
    if count <= 0:
        return "no nearby crash"
    return "1 nearby crash within 3 km"


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


def sample_final_qa(pairs: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    section("Sample final QA")
    warnings = []
    rng = np.random.default_rng(RANDOM_SEED)
    sampled_parts = []
    for option in ["A", "B", "C"]:
        group = pairs[pairs["correct_option"] == option]
        n = min(FINAL_PER_CLASS, len(group))
        if n < FINAL_PER_CLASS:
            msg = f"Class {option} has only {len(group)} samples; using {n}."
            print(f"WARNING: {msg}")
            warnings.append(msg)
        if n:
            sampled_parts.append(group.sample(n=n, random_state=RANDOM_SEED))
    final = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else pd.DataFrame()
    if not final.empty:
        final = final.iloc[rng.permutation(len(final))].reset_index(drop=True)
    final["question"] = final.apply(make_question, axis=1)
    final["options"] = OPTIONS_TEXT
    final["option_label_map"] = json.dumps(OPTION_LABEL_MAP, sort_keys=True)
    return final, warnings


OUTPUT_COLUMNS = [
    "question_id", "context_type", "station_key", "station_name", "road_name",
    "suburb", "lga", "date_A", "date_B", "hour", "day_type", "month_A",
    "month_B", "season_A", "season_B", "rain_bucket_A", "rain_bucket_B",
    "temperature_bucket_A", "temperature_bucket_B", "crash_count_A",
    "crash_count_B", "crash_bucket_A", "crash_bucket_B", "public_holiday_A",
    "public_holiday_B", "school_holiday_A", "school_holiday_B",
    "has_event_3km_A", "has_event_3km_B", "event_count_A", "event_count_B",
    "nearest_event_type_A", "nearest_event_type_B", "nearest_event_name_A",
    "nearest_event_name_B", "activity_difference_type", "volume_A", "volume_B",
    "delta_pct", "correct_answer", "correct_option", "match_level", "question",
    "options", "option_label_map", "gold_answer_type",
]


def ratio_equal(series_a: pd.Series, series_b: pd.Series) -> float:
    if len(series_a) == 0:
        return 0.0
    return float(series_a.eq(series_b).mean())


def sanity_stats(candidate_pairs: pd.DataFrame, final_qa: pd.DataFrame, warnings: list[str], build_stats: dict) -> dict:
    stats = {
        **build_stats,
        "final_qa_count": int(len(final_qa)),
        "candidate_correct_answer_distribution": candidate_pairs["correct_answer"].value_counts().to_dict(),
        "candidate_correct_option_distribution": candidate_pairs["correct_option"].value_counts().sort_index().to_dict(),
        "final_correct_answer_distribution": final_qa["correct_answer"].value_counts().to_dict(),
        "final_correct_option_distribution": final_qa["correct_option"].value_counts().sort_index().to_dict(),
        "match_level_distribution": candidate_pairs["match_level"].value_counts().to_dict(),
        "activity_difference_type_distribution": candidate_pairs["activity_difference_type"].value_counts().head(50).to_dict(),
        "max_crash_count_A": safe_float(candidate_pairs["crash_count_A"].max()),
        "max_crash_count_B": safe_float(candidate_pairs["crash_count_B"].max()),
        "crash_bucket_match_ratio": ratio_equal(candidate_pairs["crash_bucket_A"], candidate_pairs["crash_bucket_B"]),
        "rain_bucket_match_ratio": ratio_equal(candidate_pairs["rain_bucket_A"], candidate_pairs["rain_bucket_B"]),
        "temperature_bucket_match_ratio": ratio_equal(candidate_pairs["temperature_bucket_A"], candidate_pairs["temperature_bucket_B"]),
        "final_has_activity_difference_ratio": float(final_qa["activity_difference_type"].astype(str).ne("").mean()) if len(final_qa) else 0.0,
        "event_count_A_distribution": final_qa["event_count_A"].value_counts().sort_index().to_dict(),
        "event_count_B_distribution": final_qa["event_count_B"].value_counts().sort_index().to_dict(),
        "public_holiday_A_distribution": final_qa["public_holiday_A"].value_counts().to_dict(),
        "public_holiday_B_distribution": final_qa["public_holiday_B"].value_counts().to_dict(),
        "school_holiday_A_distribution": final_qa["school_holiday_A"].value_counts().to_dict(),
        "school_holiday_B_distribution": final_qa["school_holiday_B"].value_counts().to_dict(),
        "nearest_event_type_A_distribution": final_qa["nearest_event_type_A"].replace("", "none").value_counts().head(30).to_dict(),
        "nearest_event_type_B_distribution": final_qa["nearest_event_type_B"].replace("", "none").value_counts().head(30).to_dict(),
        "warnings": warnings,
    }
    if stats["max_crash_count_A"] > 1 or stats["max_crash_count_B"] > 1:
        raise AssertionError("crash_count_A/B exceeds 1")
    for key in ["crash_bucket_match_ratio", "rain_bucket_match_ratio", "temperature_bucket_match_ratio", "final_has_activity_difference_ratio"]:
        if stats[key] < 1.0:
            raise AssertionError(f"Sanity check failed: {key}={stats[key]}")
    return stats


README_TEXT = """# Planned Activity / Calendar Contrast Task

This task evaluates whether models can compare traffic-volume differences under different planned activity and calendar contexts.

Scenario A is not required to be a normal reference row. The task is not a simple no-event vs event or non-holiday vs holiday binary contrast, because public holidays, school holidays, nearby events, event counts, event names, and event types do not form a reliable linear intensity scale.

Scenario B is not assumed to have a stronger activity context than Scenario A. The only requirement is that Scenario A and Scenario B differ in at least one planned activity/calendar feature.

Planned activity/calendar features include public holiday, school holiday, nearby event presence, event count, nearest_event_type, and nearest_event_name.

Volume_A, volume_B, and delta_pct are saved for diagnostics and label construction only. They are not included in the prompt.

Crash, weather, rain bucket, temperature bucket, station, hour, and day type are treated as controls. Crash count is restricted to 0 first, with fallback to <=1 only if needed, and crash buckets must match within each pair.

Limitations: event_count and nearest event descriptions are proxy measures of planned activity exposure. They do not represent strict causal intensity.
"""


def write_outputs(candidate_pairs: pd.DataFrame, final_qa: pd.DataFrame, stats: dict) -> None:
    section("Write outputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidate_path = OUTPUT_DIR / "planned_activity_candidate_pairs_6000.csv"
    csv_path = OUTPUT_DIR / "planned_activity_qa_eval600.csv"
    jsonl_path = OUTPUT_DIR / "planned_activity_qa_eval600.jsonl"
    stats_path = OUTPUT_DIR / "planned_activity_pair_stats.json"
    readme_path = OUTPUT_DIR / "planned_activity_readme.md"
    script_path = OUTPUT_DIR / "build_planned_activity_contrast_qa.py"

    candidate_out = candidate_pairs.copy()
    if "question" not in candidate_out.columns:
        candidate_out["question"] = candidate_out.apply(make_question, axis=1)
    candidate_out["options"] = OPTIONS_TEXT
    candidate_out["option_label_map"] = json.dumps(OPTION_LABEL_MAP, sort_keys=True)
    candidate_pairs = candidate_out

    candidate_pairs[OUTPUT_COLUMNS].to_csv(candidate_path, index=False)
    final_qa[OUTPUT_COLUMNS].to_csv(csv_path, index=False)
    save_jsonl(final_qa[OUTPUT_COLUMNS].to_dict("records"), jsonl_path)
    stats["output_paths"] = {
        "candidate_pairs_csv": str(candidate_path),
        "qa_eval600_csv": str(csv_path),
        "qa_eval600_jsonl": str(jsonl_path),
        "stats_json": str(stats_path),
        "readme": str(readme_path),
        "script": str(script_path),
    }
    save_json(stats, stats_path)
    readme_path.write_text(README_TEXT, encoding="utf-8")
    source = Path(__file__).resolve()
    if source != script_path:
        shutil.copy2(source, script_path)
    print(f"Candidate pairs: {candidate_path}")
    print(f"Final QA CSV: {csv_path}")
    print(f"Final QA JSONL: {jsonl_path}")
    print(f"Stats: {stats_path}")


def main() -> None:
    section("Planned Activity / Calendar Contrast Task")
    df, alias_info = load_data()
    candidate_pairs, build_stats = build_candidate_pairs(df)
    if candidate_pairs.empty:
        raise RuntimeError("No candidate pairs were created.")
    final_qa, warnings = sample_final_qa(candidate_pairs)
    build_stats["alias_info"] = alias_info
    stats = sanity_stats(candidate_pairs, final_qa, warnings, build_stats)
    write_outputs(candidate_pairs, final_qa, stats)
    section("Summary")
    print(f"Candidate pairs: {len(candidate_pairs):,}")
    print(f"Final QA rows: {len(final_qa):,}")
    print(f"Correct option distribution: {final_qa['correct_option'].value_counts().sort_index().to_dict()}")
    print(f"Match levels: {candidate_pairs['match_level'].value_counts().to_dict()}")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
