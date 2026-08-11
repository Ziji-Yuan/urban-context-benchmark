"""
Build Task 4 Rain Numerical Context Contrastive QA.

This script creates pairwise rain numerical-context contrastive examples.
It does not call any API and does not modify the original labeled data.

Run:
    python build_rainfall_qa.py --input_csv path/to/labeled_master_table.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_CSV = REPO_ROOT / "data" / "master_table_station_hour_2022_2024_benchmark_labeled.csv"
DEFAULT_INPUT_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "generated" / "rainfall"


def section(message: str) -> None:
    print(f"\n{'=' * 72}\n  {message}\n{'=' * 72}")


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
        if base not in seen:
            seen[base] = 0
            output.append(base)
        else:
            seen[base] += 1
            output.append(f"{base}_{seen[base]}")
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


def safe_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def save_json(data, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def resolve_input_csv(input_csv: Path | None) -> Path:
    if input_csv and input_csv.exists():
        return input_csv
    if DEFAULT_INPUT_CSV.exists():
        return DEFAULT_INPUT_CSV

    csv_files = sorted(DEFAULT_INPUT_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DEFAULT_INPUT_DIR}")

    def score(path: Path) -> tuple[int, int]:
        name = path.name.lower()
        s = 0
        if "master" in name:
            s += 100
        if "labeled" in name:
            s += 80
        if "station_hour" in name:
            s += 30
        if "baseline" in name:
            s -= 80
        return s, path.stat().st_size

    return max(csv_files, key=score)


def first_existing(columns: set[str], candidates: list[str]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None


def add_obvious_aliases(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "traffic_change_label": "row_traffic_label",
        "name": "station_name",
        "precipitation_mm": "rain_mm",
    }
    for source, target in aliases.items():
        if target not in df.columns and source in df.columns:
            df[target] = df[source]
            print(f"Alias created: {source} -> {target}")
    return df


def detect_temperature_column(df: pd.DataFrame) -> dict:
    columns = set(df.columns)
    direct_candidates = [
        "temperature_2m_c",
        "temperature_2m",
        "temperature_c",
        "temp_c",
        "temperature",
        "temp",
        "air_temperature",
        "apparent_temperature_c",
        "apparent_temperature",
    ]
    direct = first_existing(columns, direct_candidates)
    if direct:
        return {
            "temp_source": "direct_temperature_column",
            "temperature_column_used": direct,
            "max_temp_column": None,
            "min_temp_column": None,
        }

    max_col = first_existing(
        columns,
        ["max_temp", "maximum_temperature", "max_temperature", "temperature_max"],
    )
    min_col = first_existing(
        columns,
        ["min_temp", "minimum_temperature", "min_temperature", "temperature_min"],
    )
    if max_col and min_col:
        return {
            "temp_source": "average_of_max_min",
            "temperature_column_used": f"{max_col},{min_col}",
            "max_temp_column": max_col,
            "min_temp_column": min_col,
        }

    return {
        "temp_source": "not_available",
        "temperature_column_used": None,
        "max_temp_column": None,
        "min_temp_column": None,
    }


def add_temperature_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    info = detect_temperature_column(df)
    if info["temp_source"] == "direct_temperature_column":
        df["temp_value"] = pd.to_numeric(df[info["temperature_column_used"]], errors="coerce")
    elif info["temp_source"] == "average_of_max_min":
        max_temp = pd.to_numeric(df[info["max_temp_column"]], errors="coerce")
        min_temp = pd.to_numeric(df[info["min_temp_column"]], errors="coerce")
        df["temp_value"] = (max_temp + min_temp) / 2
    else:
        df["temp_value"] = np.nan

    df["temp_bucket"] = pd.cut(
        df["temp_value"],
        bins=[-np.inf, 10, 20, 30, np.inf],
        labels=["cold", "mild", "warm", "hot"],
        right=False,
    ).astype("object")
    df.loc[df["temp_value"].isna(), "temp_bucket"] = "unknown"
    return df, info


def month_to_season(month) -> str | None:
    m = safe_int(month)
    if m is None or m < 1 or m > 12:
        return None
    if m in {12, 1, 2}:
        return "summer"
    if m in {3, 4, 5}:
        return "autumn"
    if m in {6, 7, 8}:
        return "winter"
    return "spring"


def load_and_standardize(input_csv: Path) -> tuple[pd.DataFrame, dict]:
    section("Load and inspect labeled data")
    df = pd.read_csv(input_csv, low_memory=False)
    df.columns = make_unique_columns(list(df.columns))
    df = add_obvious_aliases(df)
    print(f"Rows loaded: {len(df):,}")
    print(f"Columns ({len(df.columns)}): {', '.join(df.columns)}")

    required = ["station_key", "hour", "volume", "rain_mm"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "date" in df.columns:
        date_part = pd.to_datetime(df["date"], errors="coerce")
        hour_part = pd.to_numeric(df["hour"], errors="coerce").fillna(0)
        df["datetime"] = date_part + pd.to_timedelta(hour_part, unit="h")

    if "date" not in df.columns and "datetime" in df.columns:
        df["date"] = df["datetime"].dt.date.astype(str)

    if "month" not in df.columns and "datetime" in df.columns:
        df["month"] = df["datetime"].dt.month

    if "season" not in df.columns and "month" in df.columns:
        df["season"] = df["month"].map(month_to_season)

    if "day_type" not in df.columns:
        if "is_weekend" in df.columns:
            df["day_type"] = df["is_weekend"].map(lambda x: "weekend" if safe_bool(x) else "weekday")
        elif "day_of_week" in df.columns:
            dow = pd.to_numeric(df["day_of_week"], errors="coerce")
            df["day_type"] = np.where(dow >= 5, "weekend", "weekday")
        elif "datetime" in df.columns:
            df["day_type"] = np.where(df["datetime"].dt.dayofweek >= 5, "weekend", "weekday")
        else:
            df["day_type"] = "unknown"

    numeric_cols = [
        "hour",
        "volume",
        "expected_volume",
        "rain_mm",
        "crash_count",
        "event_count_3km",
        "month",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["has_event_3km", "public_holiday", "school_holiday"]:
        if col in df.columns:
            df[col] = df[col].map(safe_bool)

    if "row_traffic_label" in df.columns:
        df["row_traffic_label"] = df["row_traffic_label"].astype(str).str.strip().str.lower()
    df["day_type"] = df["day_type"].astype(str).str.strip().str.lower()
    if "season" in df.columns:
        df["season"] = df["season"].astype(str).str.strip().str.lower()

    df, temp_info = add_temperature_features(df)
    df = df.reset_index(drop=False).rename(columns={"index": "source_index"})
    return df, temp_info


def no_event_mask(df: pd.DataFrame) -> pd.Series:
    if "has_event_3km" in df.columns:
        return df["has_event_3km"].eq(False)
    if "event_count_3km" in df.columns:
        return df["event_count_3km"].fillna(0).le(0)
    return pd.Series(True, index=df.index)


def no_major_crash_mask(df: pd.DataFrame) -> pd.Series:
    if "crash_count" in df.columns:
        return df["crash_count"].fillna(999).le(1)
    return pd.Series(True, index=df.index)


def non_holiday_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if "public_holiday" in df.columns:
        mask &= df["public_holiday"].eq(False)
    if "school_holiday" in df.columns:
        mask &= df["school_holiday"].eq(False)
    return mask


def normal_reference_mask(df: pd.DataFrame) -> pd.Series:
    if "row_traffic_label" in df.columns:
        return df["row_traffic_label"].eq("normal")
    return pd.Series(True, index=df.index)


def create_candidate_pools(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    section("Create Scenario A and Scenario B candidate pools")
    clean_context = no_event_mask(df) & no_major_crash_mask(df) & non_holiday_mask(df)
    valid_core = (
        df["station_key"].notna()
        & df["hour"].notna()
        & df["volume"].notna()
        & df["volume"].gt(0)
        & df["rain_mm"].notna()
    )
    base_filter = clean_context & valid_core
    a_pool = df.loc[base_filter].copy()
    b_pool = df.loc[base_filter].copy()
    print(f"Scenario A candidates: {len(a_pool):,}")
    print(f"Scenario B candidates: {len(b_pool):,}")
    print("Scenario A and B use the same base filters; row_traffic_label is not used as an A-only filter.")
    return a_pool, b_pool


def group_indices(df: pd.DataFrame, keys: list[str]) -> dict[tuple, list[int]]:
    if any(key not in df.columns for key in keys):
        return {}
    grouped: dict[tuple, list[int]] = {}
    for key, group in df.groupby(keys, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        grouped[key] = group.index.tolist()
    return grouped


def row_key(row: pd.Series, keys: list[str]) -> tuple:
    return tuple(row.get(key) for key in keys)


def choose_best_candidate(b_row: pd.Series, a_pool: pd.DataFrame, idxs: list[int]) -> int | None:
    if not idxs:
        return None
    candidates = a_pool.loc[idxs].copy()
    candidates = candidates[candidates["source_index"] != b_row.get("source_index")]
    if candidates.empty:
        return None

    if "expected_volume" in candidates.columns and pd.notna(b_row.get("expected_volume")):
        denom = candidates["expected_volume"].replace(0, np.nan)
        candidates["_expected_diff"] = (
            candidates["expected_volume"] - b_row["expected_volume"]
        ).abs() / denom
    else:
        candidates["_expected_diff"] = np.nan

    if "temp_value" in candidates.columns and pd.notna(b_row.get("temp_value")):
        candidates["_temp_abs_diff"] = (candidates["temp_value"] - b_row["temp_value"]).abs()
    else:
        candidates["_temp_abs_diff"] = np.nan

    if "datetime" in candidates.columns and pd.notna(b_row.get("datetime")):
        candidates["_date_distance"] = (candidates["datetime"] - b_row["datetime"]).abs()
    else:
        candidates["_date_distance"] = pd.NaT

    candidates = candidates.sort_values(
        ["_expected_diff", "_temp_abs_diff", "_date_distance"],
        ascending=[True, True, True],
        na_position="last",
    )
    return int(candidates.iloc[0].name)


def add_match_value(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "expected_volume" in out.columns:
        out["_match_value"] = pd.to_numeric(out["expected_volume"], errors="coerce")
    else:
        out["_match_value"] = np.nan
    out["_match_value"] = out["_match_value"].fillna(pd.to_numeric(out["volume"], errors="coerce"))
    out["_match_value"] = out["_match_value"].fillna(0)
    return out


def date_numeric(series: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(series, errors="coerce")
    values = dt.astype("int64").to_numpy(dtype="float64")
    values[values < 0] = np.nan
    return values


def normalized_date_values(series: pd.Series | None, length: int) -> np.ndarray:
    if series is None:
        return np.array([None] * length, dtype="object")
    return pd.to_datetime(series, errors="coerce").dt.date.astype("object").to_numpy()


def match_one_level(
    a_pool: pd.DataFrame,
    b_pool: pd.DataFrame,
    keys: list[str],
    level: str,
    remaining_needed: int,
    seed: int,
) -> pd.DataFrame:
    if any(key not in a_pool.columns or key not in b_pool.columns for key in keys):
        print(f"  {level}: skipped because one or more keys are missing: {keys}")
        return pd.DataFrame(columns=["b_idx", "a_idx", "match_level"])

    if remaining_needed <= 0:
        return pd.DataFrame(columns=["b_idx", "a_idx", "match_level"])

    rng = np.random.default_rng(seed)
    a_work = add_match_value(a_pool)
    b_work = add_match_value(b_pool)
    a_groups = {key: group for key, group in a_work.groupby(keys, dropna=False, sort=False)}
    b_groups = {key: group for key, group in b_work.groupby(keys, dropna=False, sort=False)}
    common_groups = 0
    matched_parts = []
    matched_count = 0
    attempted_groups = 0
    attempted_b_rows = 0
    rejected_same_datetime = 0
    rejected_self = 0
    rejected_same_rain = 0
    rejected_small_rain_diff = 0

    print(
        f"  {level}: A groups={len(a_groups):,}, B groups={len(b_groups):,}, "
        f"B rows available={len(b_pool):,}, remaining_needed={remaining_needed:,}"
    )

    common_keys = [key for key in b_groups.keys() if key in a_groups]
    rng.shuffle(common_keys)
    print(f"  {level}: common groups available={len(common_keys):,}")

    for group_key in common_keys:
        if matched_count >= remaining_needed:
            break
        b_group = b_groups[group_key]
        a_group = a_groups.get(group_key)
        if a_group is None or a_group.empty:
            continue
        common_groups += 1
        attempted_groups += 1

        group_remaining = remaining_needed - matched_count
        b_take = min(len(b_group), max(group_remaining * 3, group_remaining))
        if len(b_group) > b_take:
            b_group = b_group.sample(n=b_take, random_state=int(rng.integers(0, 2**31 - 1)))
        attempted_b_rows += len(b_group)

        a_sorted = a_group.sort_values("_match_value").copy()
        a_values = a_sorted["_match_value"].to_numpy(dtype="float64")
        b_values = b_group["_match_value"].to_numpy(dtype="float64")
        insert_pos = np.searchsorted(a_values, b_values, side="left")

        candidate_frames = []
        for offset in [-2, -1, 0, 1, 2]:
            pos = insert_pos + offset
            valid = (pos >= 0) & (pos < len(a_sorted))
            if not valid.any():
                continue
            b_candidates = b_group.loc[valid].copy()
            a_candidates = a_sorted.iloc[pos[valid]].copy().reset_index(drop=False)
            b_candidates = b_candidates.reset_index(drop=False).rename(columns={"index": "b_idx"})
            candidate = pd.DataFrame(
                {
                    "b_idx": b_candidates["b_idx"].to_numpy(),
                    "a_idx": a_candidates["index"].to_numpy(),
                    "source_index_B": b_candidates["source_index"].to_numpy(),
                    "source_index_A": a_candidates["source_index"].to_numpy(),
                    "date_B": normalized_date_values(b_candidates.get("date"), len(b_candidates)),
                    "date_A": normalized_date_values(a_candidates.get("date"), len(a_candidates)),
                    "hour_B": pd.to_numeric(b_candidates.get("hour"), errors="coerce").to_numpy(),
                    "hour_A": pd.to_numeric(a_candidates.get("hour"), errors="coerce").to_numpy(),
                    "station_key_B": b_candidates.get("station_key").to_numpy(),
                    "station_key_A": a_candidates.get("station_key").to_numpy(),
                    "rain_mm_B": pd.to_numeric(b_candidates.get("rain_mm"), errors="coerce").to_numpy(dtype="float64"),
                    "rain_mm_A": pd.to_numeric(a_candidates.get("rain_mm"), errors="coerce").to_numpy(dtype="float64"),
                    "_expected_diff": np.abs(
                        b_candidates["_match_value"].to_numpy(dtype="float64")
                        - a_candidates["_match_value"].to_numpy(dtype="float64")
                    ),
                    "_temp_abs_diff": np.abs(
                        pd.to_numeric(b_candidates.get("temp_value"), errors="coerce").to_numpy(dtype="float64")
                        - pd.to_numeric(a_candidates.get("temp_value"), errors="coerce").to_numpy(dtype="float64")
                    ),
                    "_date_distance": np.abs(
                        date_numeric(b_candidates.get("datetime"))
                        - date_numeric(a_candidates.get("datetime"))
                    ),
                }
            )
            if "record_id" in b_candidates.columns and "record_id" in a_candidates.columns:
                candidate["record_id_B"] = b_candidates["record_id"].to_numpy()
                candidate["record_id_A"] = a_candidates["record_id"].to_numpy()
            candidate_frames.append(candidate)

        if not candidate_frames:
            continue

        candidates = pd.concat(candidate_frames, ignore_index=True)
        same_datetime = candidates["date_A"].eq(candidates["date_B"]) & candidates["hour_A"].eq(candidates["hour_B"])
        rejected_same_datetime += int(same_datetime.sum())
        if "record_id_A" in candidates.columns and "record_id_B" in candidates.columns:
            self_pair = candidates["record_id_A"].eq(candidates["record_id_B"])
        else:
            self_pair = (
                candidates["station_key_A"].eq(candidates["station_key_B"])
                & candidates["date_A"].eq(candidates["date_B"])
                & candidates["hour_A"].eq(candidates["hour_B"])
            )
        self_pair |= candidates["source_index_A"].eq(candidates["source_index_B"])
        rejected_self += int(self_pair.sum())
        rain_diff = candidates["rain_mm_B"].sub(candidates["rain_mm_A"])
        same_rain = candidates["rain_mm_A"].eq(candidates["rain_mm_B"])
        small_rain_diff = rain_diff.abs().lt(1)
        rejected_same_rain += int(same_rain.sum())
        rejected_small_rain_diff += int(small_rain_diff.sum())
        candidates = candidates[~same_datetime & ~self_pair & ~same_rain & ~small_rain_diff]
        if candidates.empty:
            continue
        candidates = candidates.drop_duplicates(["b_idx", "a_idx"])
        candidates = candidates.sort_values(
            ["b_idx", "_expected_diff", "_temp_abs_diff", "_date_distance"],
            ascending=[True, True, True, True],
            na_position="last",
        )
        best = candidates.drop_duplicates("b_idx", keep="first")
        if len(best) > group_remaining:
            best = best.sample(
                n=group_remaining,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
        matched_parts.append(best[["b_idx", "a_idx"]])
        matched_count += len(best)

    if not matched_parts:
        print(
            f"  {level}: attempted groups={attempted_groups:,}, "
            f"attempted B rows={attempted_b_rows:,}, matched rows=0"
        )
        out = pd.DataFrame(columns=["b_idx", "a_idx", "match_level"])
        out.attrs["attempted_groups"] = attempted_groups
        out.attrs["attempted_b_rows"] = attempted_b_rows
        out.attrs["rejected_same_datetime_pairs"] = rejected_same_datetime
        out.attrs["rejected_self_pairs"] = rejected_self
        out.attrs["rejected_same_rain_pairs"] = rejected_same_rain
        out.attrs["rejected_small_rain_diff_pairs"] = rejected_small_rain_diff
        return out

    matched = pd.concat(matched_parts, ignore_index=True).drop_duplicates("b_idx", keep="first")
    if len(matched) > remaining_needed:
        matched = matched.sample(n=remaining_needed, random_state=seed)
    matched["match_level"] = level
    print(
        f"  {level}: attempted groups={attempted_groups:,}, "
        f"attempted B rows={attempted_b_rows:,}, matched rows={len(matched):,}"
    )
    matched.attrs["attempted_groups"] = attempted_groups
    matched.attrs["attempted_b_rows"] = attempted_b_rows
    matched.attrs["rejected_same_datetime_pairs"] = rejected_same_datetime
    matched.attrs["rejected_self_pairs"] = rejected_self
    matched.attrs["rejected_same_rain_pairs"] = rejected_same_rain
    matched.attrs["rejected_small_rain_diff_pairs"] = rejected_small_rain_diff
    return matched


def match_pairs(
    a_pool: pd.DataFrame,
    b_pool: pd.DataFrame,
    qa_pair_cap: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    section("Match A/B pairs")
    match_specs = [
        ("same_month_temp", ["station_key", "hour", "day_type", "month", "temp_bucket"]),
        ("same_season_temp", ["station_key", "hour", "day_type", "season", "temp_bucket"]),
        ("same_month_no_temp", ["station_key", "hour", "day_type", "month"]),
        ("same_season_no_temp", ["station_key", "hour", "day_type", "season"]),
    ]
    skipped = {"no_candidate": 0, "self_only": 0, "invalid_volume": 0}
    stats = {
        "b_rows_attempted": 0,
        "groups_attempted": 0,
        "early_stopping_triggered": False,
        "early_stop_match_level": None,
        "qa_pair_cap": int(qa_pair_cap),
        "rejected_same_datetime_pairs": 0,
        "rejected_self_pairs": 0,
        "rejected_same_rain_pairs": 0,
        "rejected_small_rain_diff_pairs": 0,
    }

    matched_levels = []
    fallback_counts: dict[str, int] = {}
    for level, keys in match_specs:
        current_count = sum(len(x) for x in matched_levels)
        if current_count >= qa_pair_cap:
            break
        remaining_needed = qa_pair_cap - current_count
        matched_b_so_far = (
            pd.concat(matched_levels, ignore_index=True)["b_idx"].tolist()
            if matched_levels
            else []
        )
        remaining_b = b_pool.drop(index=matched_b_so_far, errors="ignore")
        if remaining_b.empty:
            break
        matched = match_one_level(
            a_pool,
            remaining_b,
            keys,
            level,
            remaining_needed=remaining_needed,
            seed=seed + len(fallback_counts) * 101,
        )
        stats["b_rows_attempted"] += int(matched.attrs.get("attempted_b_rows", 0))
        stats["groups_attempted"] += int(matched.attrs.get("attempted_groups", 0))
        stats["rejected_same_datetime_pairs"] += int(matched.attrs.get("rejected_same_datetime_pairs", 0))
        stats["rejected_self_pairs"] += int(matched.attrs.get("rejected_self_pairs", 0))
        stats["rejected_same_rain_pairs"] += int(matched.attrs.get("rejected_same_rain_pairs", 0))
        stats["rejected_small_rain_diff_pairs"] += int(matched.attrs.get("rejected_small_rain_diff_pairs", 0))
        fallback_counts[level] = int(len(matched))
        if matched.empty:
            continue
        matched_levels.append(matched)
        current_count = sum(len(x) for x in matched_levels)
        print(f"  {level}: cumulative matched rows={current_count:,}/{qa_pair_cap:,}")
        if current_count >= qa_pair_cap:
            stats["early_stopping_triggered"] = True
            stats["early_stop_match_level"] = level
            print(f"  Early stopping triggered at {level} with {current_count:,} pairs.")
            break

    if matched_levels:
        matched_all = pd.concat(matched_levels, ignore_index=True)
    else:
        matched_all = pd.DataFrame(columns=["b_idx", "a_idx", "match_level"])

    print(f"Fallback match counts: {fallback_counts}")

    if matched_all.empty:
        pairs = pd.DataFrame()
        print("Matched pairs: 0")
        print(f"Skipped: {skipped}")
        skipped.update(stats)
        return pairs, skipped

    a_rows = a_pool.loc[matched_all["a_idx"].to_numpy()].reset_index(drop=True)
    b_rows = b_pool.loc[matched_all["b_idx"].to_numpy()].reset_index(drop=True)
    match_levels = matched_all["match_level"].reset_index(drop=True)

    volume_a = pd.to_numeric(a_rows["volume"], errors="coerce")
    volume_b = pd.to_numeric(b_rows["volume"], errors="coerce")
    valid = volume_a.notna() & volume_b.notna() & volume_a.gt(0)
    skipped["invalid_volume"] = int((~valid).sum())
    if not valid.all():
        a_rows = a_rows.loc[valid].reset_index(drop=True)
        b_rows = b_rows.loc[valid].reset_index(drop=True)
        match_levels = match_levels.loc[valid].reset_index(drop=True)
        volume_a = volume_a.loc[valid].reset_index(drop=True)
        volume_b = volume_b.loc[valid].reset_index(drop=True)

    delta_pct = (volume_b - volume_a) / volume_a
    answer = np.where(delta_pct > 0.10, "A", np.where(delta_pct < -0.10, "B", "C"))
    answer_label = pd.Series(answer).map({"A": "increase", "B": "decrease", "C": "normal"})
    rain_a = pd.to_numeric(a_rows["rain_mm"], errors="coerce").fillna(0.0)
    rain_b = pd.to_numeric(b_rows["rain_mm"], errors="coerce").fillna(0.0)

    pairs = pd.DataFrame(
        {
            "question_id": [f"rain_numeric_{i + 1:06d}" for i in range(len(a_rows))],
            "station_key": b_rows["station_key"].to_numpy(),
            "station_name": b_rows["station_name"].to_numpy()
            if "station_name" in b_rows.columns
            else b_rows.get("name", pd.Series(["unknown"] * len(b_rows))).to_numpy(),
            "road_name": b_rows["road_name"].to_numpy()
            if "road_name" in b_rows.columns
            else "unknown",
            "suburb": b_rows["suburb"].to_numpy() if "suburb" in b_rows.columns else "unknown",
            "lga": b_rows["lga"].to_numpy() if "lga" in b_rows.columns else "unknown",
            "date_A": a_rows["date"].to_numpy() if "date" in a_rows.columns else None,
            "date_B": b_rows["date"].to_numpy() if "date" in b_rows.columns else None,
            "datetime_A": a_rows["datetime"].to_numpy() if "datetime" in a_rows.columns else None,
            "datetime_B": b_rows["datetime"].to_numpy() if "datetime" in b_rows.columns else None,
            "hour_A": a_rows["hour"].map(safe_int).to_numpy(),
            "hour_B": b_rows["hour"].map(safe_int).to_numpy(),
            "hour": b_rows["hour"].map(safe_int).to_numpy(),
            "day_type": b_rows["day_type"].to_numpy(),
            "volume_A": volume_a.to_numpy(),
            "volume_B": volume_b.to_numpy(),
            "delta_pct": delta_pct.to_numpy(),
            "answer": answer,
            "answer_label": answer_label.to_numpy(),
            "rain_mm_A": rain_a.to_numpy(),
            "rain_mm_B": rain_b.to_numpy(),
            "rain_delta": (rain_b - rain_a).to_numpy(),
            "abs_rain_delta": (rain_b - rain_a).abs().to_numpy(),
            "temp_A": pd.to_numeric(a_rows.get("temp_value"), errors="coerce").to_numpy()
            if "temp_value" in a_rows.columns
            else np.nan,
            "temp_B": pd.to_numeric(b_rows.get("temp_value"), errors="coerce").to_numpy()
            if "temp_value" in b_rows.columns
            else np.nan,
            "temp_bucket_A": a_rows["temp_bucket"].to_numpy()
            if "temp_bucket" in a_rows.columns
            else "unknown",
            "temp_bucket_B": b_rows["temp_bucket"].to_numpy()
            if "temp_bucket" in b_rows.columns
            else "unknown",
            "match_level": match_levels.to_numpy(),
        }
    )
    pairs["rain_relation"] = [
        rain_relation(a, b) for a, b in zip(pairs["rain_mm_A"], pairs["rain_mm_B"])
    ]
    if "record_id" in a_rows.columns and "record_id" in b_rows.columns:
        pairs["record_id_A"] = a_rows["record_id"].to_numpy()
        pairs["record_id_B"] = b_rows["record_id"].to_numpy()
    pairs["land_use_description"] = b_rows.apply(build_land_use_description, axis=1)
    pairs["poi_summary"] = b_rows.apply(build_poi_summary, axis=1)

    same_datetime_final = pairs["date_A"].eq(pairs["date_B"]) & pairs["hour_A"].eq(pairs["hour_B"])
    if same_datetime_final.any():
        print("Offending same-date/same-hour pairs:")
        print(pairs.loc[same_datetime_final, ["station_key", "date_A", "date_B", "hour_A", "hour_B"]].head(10))
    assert not same_datetime_final.any()
    assert not (pairs["rain_mm_A"] == pairs["rain_mm_B"]).any()
    assert (pairs["rain_mm_B"].sub(pairs["rain_mm_A"]).abs() >= 1).all()
    stats["final_same_datetime_pair_count"] = int(same_datetime_final.sum())
    if "record_id_A" in pairs.columns and "record_id_B" in pairs.columns:
        stats["final_self_pair_count"] = int(pairs["record_id_A"].eq(pairs["record_id_B"]).sum())
    else:
        stats["final_self_pair_count"] = int(same_datetime_final.sum())
    stats["final_same_rain_pair_count"] = int(pairs["rain_mm_A"].eq(pairs["rain_mm_B"]).sum())
    stats["final_small_rain_diff_pair_count"] = int(pairs["rain_mm_B"].sub(pairs["rain_mm_A"]).abs().lt(1).sum())

    print(f"Matched pairs: {len(pairs):,}")
    stats["matched_pairs_collected"] = int(len(pairs))
    print(f"Skipped: {skipped}")
    print(f"Matching stats: {stats}")
    if "match_level" in pairs.columns:
        print(f"Match level counts: {pairs['match_level'].value_counts().to_dict()}")
    if "answer" in pairs.columns:
        print(f"Answer distribution: {pairs['answer'].value_counts().sort_index().to_dict()}")
    skipped.update(stats)
    return pairs, skipped


def text_value(row: pd.Series, candidates: list[str], default: str = "unknown") -> str:
    for col in candidates:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col]).strip()
    return default


def format_landuse_pct(value) -> str | None:
    pct = safe_float(value)
    if pct is None:
        return None
    if 0 <= pct <= 1:
        pct *= 100
    if float(pct).is_integer():
        return f"{pct:.0f}%"
    return f"{pct:.2f}".rstrip("0").rstrip(".") + "%"


def build_land_use_description(row: pd.Series) -> str:
    category_cols = [
        "landuse_dom_category_500m",
        "landuse_top1_category_500m",
        "land_use_description",
    ]
    ratio_cols = [
        "landuse_dom_ratio_500m",
        "landuse_top1_ratio_500m",
        "landuse_percentage_500m",
    ]
    category = text_value(row, category_cols, "")
    ratio_col = first_existing(set(row.index), ratio_cols)
    pct = format_landuse_pct(row.get(ratio_col)) if ratio_col else None
    if category and pct:
        return f"{category} land use ({pct})"
    if category:
        return f"{category} land use"
    return "urban mixed land use"


def build_poi_summary(row: pd.Series) -> str:
    poi_cols = [
        ("food", "poi_food_count_500m"),
        ("education", "poi_education_count_500m"),
        ("healthcare", "poi_healthcare_count_500m"),
        ("public transport", "poi_public_transport_count_500m"),
        ("leisure", "poi_leisure_count_500m"),
        ("tourism", "poi_tourism_count_500m"),
        ("shops", "poi_shop_count_500m"),
    ]
    parts = []
    for label, col in poi_cols:
        if col in row.index:
            count = safe_int(row.get(col), 0)
            if count and count > 0:
                parts.append(f"{count} {label}")
    building_count = safe_int(row.get("building_count_500m"), 0)
    if building_count and building_count > 0:
        parts.append(f"{building_count} buildings")
    return ", ".join(parts[:6]) if parts else "limited mapped POIs"


def rain_relation(rain_a: float, rain_b: float) -> str:
    eps = 0.1
    a_rain = rain_a > eps
    b_rain = rain_b > eps
    if not a_rain and not b_rain:
        return "both_no_rain"
    if not a_rain and b_rain:
        return "A_no_rain_B_rain"
    if a_rain and not b_rain:
        return "A_rain_B_no_rain"
    if rain_b > rain_a + eps:
        return "both_rain_B_higher"
    if rain_b < rain_a - eps:
        return "both_rain_B_lower"
    return "both_rain_similar"


def make_question(row: pd.Series) -> str:
    area = (
        f"{row['suburb']}, {row['lga']}"
        if row.get("suburb") != "unknown" and row.get("lga") != "unknown"
        else row.get("suburb") or row.get("lga") or "unknown area"
    )
    return f"""You are given two comparable traffic scenarios at the same monitoring station.

Station: {row['station_name']}
Road: {row['road_name']}
Area: {area}
Nearby urban context: {row['land_use_description']}, with nearby POIs including {row['poi_summary']}.

Both scenarios occur at the same station and hour, with major confounding factors controlled where possible.

Scenario A:
- Time: {row['day_type']}, {int(row['hour'])}:00
- Rainfall: {row['rain_mm_A']:.1f} mm
- Temperature category: {row['temp_bucket_A']}
- Event: no nearby event
- Crash: no major nearby crash
- Holiday: non-holiday


Scenario B:
- Time: {row['day_type']}, {int(row['hour'])}:00
- Rainfall: {row['rain_mm_B']:.1f} mm
- Temperature category: {row['temp_bucket_B']}
- Event: no nearby event
- Crash: no major nearby crash
- Holiday: non-holiday

Compared with Scenario A, the traffic volume in Scenario B is most likely to be:
(A) Higher than Scenario A, if traffic volume is more than 10% higher.
(B) Lower than Scenario A, if traffic volume is more than 10% lower.
(C) Similar to Scenario A, if the difference is within +/-10%.

Choose exactly one option from A, B, or C.

Do not explain your reasoning.

Your entire response must be exactly one line:
Final answer: <A/B/C>"""


def build_qa(pairs: pd.DataFrame) -> pd.DataFrame:
    qa = pairs.copy()
    qa["question"] = qa.apply(make_question, axis=1)
    qa["options.A"] = "Higher than Scenario A, if traffic volume is more than 10% higher."
    qa["options.B"] = "Lower than Scenario A, if traffic volume is more than 10% lower."
    qa["options.C"] = "Similar to Scenario A, if the difference is within +/-10%."
    qa["correct_answer"] = qa["answer"]
    qa["correct_label"] = qa["answer_label"]
    return qa


def sample_eval_set(qa: pd.DataFrame, eval_size: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    per_class_target = eval_size // 3
    sampled_parts = []
    used_indices: set[int] = set()
    for answer in ["A", "B", "C"]:
        group = qa[qa["answer"] == answer]
        n = min(per_class_target, len(group))
        if n > 0:
            picked = group.sample(n=n, random_state=seed)
            sampled_parts.append(picked)
            used_indices.update(picked.index.tolist())

    sampled = pd.concat(sampled_parts) if sampled_parts else pd.DataFrame(columns=qa.columns)
    remaining_slots = max(0, eval_size - len(sampled))
    if remaining_slots > 0:
        remaining = qa.drop(index=list(used_indices), errors="ignore")
        if not remaining.empty:
            filler = remaining.sample(
                n=min(remaining_slots, len(remaining)),
                random_state=seed + 1,
            )
            sampled = pd.concat([sampled, filler])

    if sampled.empty:
        return sampled
    order = rng.permutation(len(sampled))
    return sampled.iloc[order].reset_index(drop=True)


def numeric_summary(series: pd.Series) -> dict:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def build_diagnostics(
    df: pd.DataFrame,
    a_pool: pd.DataFrame,
    b_pool: pd.DataFrame,
    pairs: pd.DataFrame,
    qa: pd.DataFrame,
    eval_qa: pd.DataFrame,
    skipped: dict,
    temp_info: dict,
    args: argparse.Namespace,
) -> dict:
    temp_match_rate = None
    if len(pairs) > 0:
        temp_match_rate = float((pairs["temp_bucket_A"] == pairs["temp_bucket_B"]).mean())
    abs_rain_diff = pairs.get("abs_rain_delta", pd.Series(dtype=float))
    abs_rain_le_1 = int(abs_rain_diff.le(1).sum()) if len(abs_rain_diff) else 0
    abs_rain_gt_1 = int(abs_rain_diff.gt(1).sum()) if len(abs_rain_diff) else 0
    abs_rain_gt_5 = int(abs_rain_diff.gt(5).sum()) if len(abs_rain_diff) else 0
    pair_count = int(len(pairs))

    def pct(count: int) -> float | None:
        return float(count / pair_count) if pair_count else None

    return {
        "task_name": "Rain Numerical Context Contrastive Task",
        "input_csv": str(args.input_csv),
        "output_dir": str(args.output_dir),
        "threshold": args.threshold,
        "seed": args.seed,
        "qa_pair_cap": args.qa_pair_cap,
        "total_rows_loaded": int(len(df)),
        "A_pool_count": int(len(a_pool)),
        "B_pool_count": int(len(b_pool)),
        "scenario_A_pool_size": int(len(a_pool)),
        "scenario_B_pool_size": int(len(b_pool)),
        "same_base_filters_for_A_and_B": True,
        "base_filter_definition": [
            "valid station_key",
            "valid hour",
            "valid volume",
            "volume > 0",
            "valid rain_mm",
            "has_event_3km == False if available",
            "crash_count <= 1 if available",
            "public_holiday == False if available",
            "school_holiday == False if available",
            "row_traffic_label is not used as an A-only filter",
        ],
        "matched_pair_count_before_QA_cap": pair_count,
        "matched_pairs": pair_count,
        "QA_full_cap_size": int(len(qa)),
        "qa_rows_after_cap": int(len(qa)),
        "B_rows_actually_attempted": int(skipped.get("b_rows_attempted", 0)),
        "groups_actually_attempted": int(skipped.get("groups_attempted", 0)),
        "early_stopping_triggered": bool(skipped.get("early_stopping_triggered", False)),
        "early_stop_match_level": skipped.get("early_stop_match_level"),
        "matched_pairs_collected": int(skipped.get("matched_pairs_collected", pair_count)),
        "rejected_same_datetime_pairs": int(skipped.get("rejected_same_datetime_pairs", 0)),
        "rejected_self_pairs": int(skipped.get("rejected_self_pairs", 0)),
        "rejected_same_rain_pairs": int(skipped.get("rejected_same_rain_pairs", 0)),
        "rejected_small_rain_diff_pairs": int(skipped.get("rejected_small_rain_diff_pairs", 0)),
        "final_same_datetime_pair_count": int(skipped.get("final_same_datetime_pair_count", 0)),
        "final_self_pair_count": int(skipped.get("final_self_pair_count", 0)),
        "final_same_rain_pair_count": int(skipped.get("final_same_rain_pair_count", 0)),
        "final_small_rain_diff_pair_count": int(skipped.get("final_small_rain_diff_pair_count", 0)),
        "eval_size_requested": int(args.eval_size),
        "eval_size_actual": int(len(eval_qa)),
        "matched_pair_answer_distribution_before_sampling": pairs["answer"].value_counts().sort_index().to_dict(),
        "full_answer_distribution": qa["answer"].value_counts().sort_index().to_dict(),
        "QA_full_answer_distribution_after_cap": qa["answer"].value_counts().sort_index().to_dict(),
        "eval_answer_distribution": eval_qa["answer"].value_counts().sort_index().to_dict(),
        "balanced_subset_size": int(len(eval_qa)),
        "balanced_subset_answer_distribution": eval_qa["answer"].value_counts().sort_index().to_dict(),
        "rain_relation_distribution": pairs["rain_relation"].value_counts().to_dict(),
        "rain_mm_A_summary": numeric_summary(pairs["rain_mm_A"]),
        "rain_mm_B_summary": numeric_summary(pairs["rain_mm_B"]),
        "rain_diff_summary": numeric_summary(pairs["rain_delta"]),
        "rain_delta_summary": numeric_summary(pairs["rain_delta"]),
        "abs_rain_delta_summary": numeric_summary(pairs["abs_rain_delta"]),
        "pairs_abs_rain_diff_le_1_count": abs_rain_le_1,
        "pairs_abs_rain_diff_le_1_pct": pct(abs_rain_le_1),
        "pairs_abs_rain_diff_gt_1_count": abs_rain_gt_1,
        "pairs_abs_rain_diff_gt_1_pct": pct(abs_rain_gt_1),
        "pairs_abs_rain_diff_gt_5_count": abs_rain_gt_5,
        "pairs_abs_rain_diff_gt_5_pct": pct(abs_rain_gt_5),
        "temp_bucket_match_rate": temp_match_rate,
        "match_level_counts": pairs["match_level"].value_counts().to_dict(),
        "skipped_rows": skipped,
        "temperature_info": temp_info,
    }


def write_outputs(
    pairs: pd.DataFrame,
    qa: pd.DataFrame,
    eval_qa: pd.DataFrame,
    diagnostics: dict,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output_dir / "rain_numeric_context_pair_pool.csv", index=False)
    qa.to_csv(output_dir / "rain_numeric_context_qa_full.csv", index=False)
    save_json(qa.to_dict("records"), output_dir / "rain_numeric_context_qa_full.json")
    eval_qa.to_csv(output_dir / "rain_numeric_context_qa_eval600.csv", index=False)
    save_json(eval_qa.to_dict("records"), output_dir / "rain_numeric_context_qa_eval600.json")
    save_json(diagnostics, output_dir / "rain_numeric_context_diagnostics.json")
    preview = "\n\n" + ("-" * 72) + "\n\n"
    (output_dir / "rain_numeric_context_prompt_preview.txt").write_text(
        preview.join(qa["question"].head(5).tolist()),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rain numerical-context contrastive QA.")
    parser.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eval_size", type=int, default=600)
    parser.add_argument("--qa_pair_cap", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    input_csv = resolve_input_csv(args.input_csv)
    args.input_csv = input_csv
    df, temp_info = load_and_standardize(input_csv)
    a_pool, b_pool = create_candidate_pools(df)
    pairs, skipped = match_pairs(a_pool, b_pool, args.qa_pair_cap, args.seed)

    if not pairs.empty:
        pairs["answer"] = np.where(
            pairs["delta_pct"] > args.threshold,
            "A",
            np.where(pairs["delta_pct"] < -args.threshold, "B", "C"),
        )
        pairs["answer_label"] = pairs["answer"].map(
            {"A": "increase", "B": "decrease", "C": "normal"}
        )

    args.matched_pair_pool_size = len(pairs)
    qa = build_qa(pairs)
    eval_qa = sample_eval_set(qa, args.eval_size, args.seed)
    diagnostics = build_diagnostics(df, a_pool, b_pool, pairs, qa, eval_qa, skipped, temp_info, args)
    write_outputs(pairs, qa, eval_qa, diagnostics, args.output_dir)

    section("Done")
    print(f"Output folder: {args.output_dir}")
    print(f"Full QA rows: {len(qa):,}")
    print(f"Eval QA rows: {len(eval_qa):,}")
    print(f"Eval distribution: {eval_qa['answer'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()


