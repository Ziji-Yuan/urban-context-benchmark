#!/usr/bin/env python3
"""
generate_qa_pairs.py — Task 1: Context-Conditioned Traffic Prediction

Generate ~3,000 structured QA pairs from labeled NSW traffic data.
Two-pass chunked reading + stratified sampling + NL description generation.

Usage:
    python generate_qa_pairs.py

Output:
    labeled_data/task1_qa_pairs.jsonl   (~3,000 QA pairs)
    labeled_data/task1_qa_stats.json    (sampling statistics + verification)
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# ============================================================================
# Configuration
# ============================================================================

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "labeled_data" / "master_table_station_hour_2022_2024_benchmark_labeled.csv"
OUTPUT_JSONL = ROOT / "labeled_data" / "task1_qa_pairs.jsonl"
OUTPUT_STATS = ROOT / "labeled_data" / "task1_qa_stats.json"

CHUNKSIZE = 300_000
RANDOM_SEED = 42

# Sampling targets
TARGET_TOTAL = 3000
MIN_PER_LABEL = 850
MIN_PER_CONDITION = 300
YEAR_TOLERANCE = 0.10  # each year within ±10% of 1000

# Label → option mapping (±10% volume change threshold)
LABEL_TO_OPTION = {
    "decrease": "A",
    "normal": "B",
    "increase": "C",
}

LABEL_ORDER = ["decrease", "normal", "increase"]

# day_of_week is integer 0-6 (0=Monday)
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Columns needed for Pass 1 (lightweight metadata)
PASS1_COLS = [
    "station_key", "date", "hour",
    "traffic_change_label", "year",
    "rain_mm", "is_rain_hour", "has_event_3km", "crash_count",
    "road_name", "lga",
    "day_of_week", "is_weekend",
    "expected_volume", "volume", "traffic_change_pct",
    "school_holiday", "public_holiday",
]

# Columns needed for Pass 2 (full context for selected rows)
PASS2_COLS = [
    # Station identity
    "station_key", "name", "road_name", "lga", "suburb",
    "wgs84_latitude", "wgs84_longitude",
    # Time
    "date", "hour", "day_of_week", "is_weekend", "year", "month",
    # Volume / labels
    "volume", "expected_volume", "traffic_change_pct",
    "traffic_change_label", "has_significant_change",
    # Weather
    "rain_mm", "rain_intensity", "is_rain_hour",
    "temperature_2m_c", "apparent_temperature_c",
    "relative_humidity_2m_pct", "humidity_category",
    "windspeed_10m_kmh", "wind_speed_category", "is_strong_wind_hour",
    "cloud_cover_pct", "cloud_category",
    "weather_context_label",
    # Events
    "event_count_3km", "has_event_3km", "nearest_event_dist_km",
    "nearest_event_type", "nearest_event_location",
    # Crashes
    "crash_count", "has_crash", "crash_fatal_count",
    # Calendar
    "public_holiday", "school_holiday",
    # POI
    "poi_food_count_500m", "poi_education_count_500m",
    "poi_healthcare_count_500m", "poi_public_transport_count_500m",
    "poi_leisure_count_500m", "poi_tourism_count_500m",
    "poi_shop_count_500m", "building_count_500m",
    # Land use
    "landuse_top1_category_500m", "landuse_top1_ratio_500m",
    "landuse_top2_category_500m", "landuse_top2_ratio_500m",
]

# ============================================================================
# Section 1: Natural Language Description Functions
# ============================================================================

def _describe_rain(row: dict) -> str:
    """Return natural-language rain description."""
    is_rain = bool(row.get("is_rain_hour", False))
    if not is_rain:
        return "no rain, dry conditions"

    intensity = str(row.get("rain_intensity", ""))
    rain_mm_val = float(row.get("rain_mm", 0))

    if intensity == "light_rain":
        return f"light rain ({rain_mm_val:.1f} mm)"
    elif intensity == "moderate_rain":
        return f"moderate rain ({rain_mm_val:.1f} mm)"
    elif intensity == "heavy_rain":
        return f"heavy rain ({rain_mm_val:.1f} mm)"
    elif intensity == "very_heavy_rain":
        return f"very heavy rain ({rain_mm_val:.1f} mm)"
    else:
        return f"rain ({rain_mm_val:.1f} mm)"


def _describe_visibility(cloud_cover_pct: float, is_rain: bool) -> str:
    """Proxy visibility from cloud cover percentage (no visibility column exists)."""
    if pd.isna(cloud_cover_pct):
        cloud_cover_pct = 0.0
    cc = float(cloud_cover_pct)

    if cc <= 10:
        vis = ">10 km"
    elif cc <= 40:
        vis = "8–10 km"
    elif cc <= 70:
        vis = "5–8 km"
    elif cc <= 90:
        vis = "2–5 km"
    else:
        vis = "<2 km"

    if is_rain:
        vis += " (reduced by precipitation)"
    return vis


def _describe_event(row: dict) -> str:
    """Return event description text or empty string."""
    if not bool(row.get("has_event_3km", False)):
        return ""

    count = int(row.get("event_count_3km", 0))
    etype = str(row.get("nearest_event_type", ""))
    eloc = str(row.get("nearest_event_location", ""))
    dist = row.get("nearest_event_dist_km")
    if pd.notna(dist) and float(dist) > 0:
        dist_str = f"{float(dist):.1f} km away"
    else:
        dist_str = "nearby"

    if count == 1:
        return (
            f"There is 1 nearby event ({etype}) at {eloc} "
            f"({dist_str})."
        )
    else:
        article = "an" if etype and etype[0].lower() in "aeiou" else "a"
        return (
            f"There are {count} nearby events within 3 km, "
            f"the nearest being {article} {etype} event at "
            f"{eloc} ({dist_str})."
        )


def _describe_crash(row: dict) -> str:
    """Return crash description text or empty string."""
    count = int(row.get("crash_count", 0))
    if count == 0:
        return ""

    parts = []
    if count == 1:
        parts.append("There is 1 reported crash nearby")
    else:
        parts.append(f"There are {count} reported crashes nearby")
    return ", ".join(parts) + "."


def _describe_holiday(row: dict) -> str:
    """Return holiday description text or empty string."""
    if bool(row.get("public_holiday", False)):
        return "It is a public holiday."
    elif bool(row.get("school_holiday", False)):
        return "It is during school holidays."
    return ""


def _describe_land_use(row: dict) -> str:
    """Return land use description string."""
    top1 = str(row.get("landuse_top1_category_500m", "")).replace("_", " ")
    ratio1 = float(row.get("landuse_top1_ratio_500m", 0))
    top2 = str(row.get("landuse_top2_category_500m", "")).replace("_", " ")
    ratio2_raw = row.get("landuse_top2_ratio_500m", 0.0)
    ratio2 = float(ratio2_raw) if pd.notna(ratio2_raw) else 0.0

    if pd.notna(ratio2) and ratio2 >= 5:
        return f"primarily {top1} ({ratio1:.0f}%), with some {top2} ({ratio2:.0f}%)"
    return f"predominantly {top1} ({ratio1:.0f}%)"


POI_DISPLAY_NAMES = {
    "poi_food_count_500m": "food/dining venues",
    "poi_education_count_500m": "education facilities",
    "poi_healthcare_count_500m": "healthcare facilities",
    "poi_public_transport_count_500m": "public transport stops",
    "poi_leisure_count_500m": "leisure venues",
    "poi_tourism_count_500m": "tourism spots",
    "poi_shop_count_500m": "shops",
}


def _describe_poi(row: dict) -> str:
    """Return top 3 POI categories as a natural language list."""
    poi_items = []
    for col, display in POI_DISPLAY_NAMES.items():
        count = int(row.get(col, 0))
        if count > 0:
            poi_items.append((count, display))

    poi_items.sort(key=lambda x: x[0], reverse=True)
    top3 = poi_items[:3]

    if not top3:
        return "few nearby facilities"

    parts = [f"{cnt} {name}" for cnt, name in top3]
    if len(parts) == 1:
        return parts[0]
    elif len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    else:
        return f"{parts[0]}, {parts[1]}, and {parts[2]}"


def _build_all_descriptions(row: dict) -> dict:
    """Build all NL descriptions for a single row at once."""
    is_rain = bool(row.get("is_rain_hour", False))
    cloud = float(row.get("cloud_cover_pct", 0))
    return {
        "rain_description": _describe_rain(row),
        "visibility": _describe_visibility(cloud, is_rain),
        "event_text": _describe_event(row),
        "crash_text": _describe_crash(row),
        "holiday_text": _describe_holiday(row),
        "land_use_description": _describe_land_use(row),
        "top_poi_categories": _describe_poi(row),
    }


# ============================================================================
# Section 2: Two-Pass Sampling
# ============================================================================

def _make_key(row: dict) -> str:
    """Build composite key for a row: station_key|date|hour."""
    return f"{row['station_key']}|{row['date']}|{row['hour']}"


def pass1_collect_metadata(input_path: Path) -> pd.DataFrame:
    """Pass 1: chunk-read only 18 lightweight columns, return sampling frame."""
    print("Pass 1: Collecting metadata...")
    t0 = time.time()

    frames = []
    total_rows = 0
    labeled_rows = 0
    nan_rows = 0
    chunk_idx = 0

    for chunk in pd.read_csv(input_path, usecols=PASS1_COLS, chunksize=CHUNKSIZE,
                             low_memory=False):
        chunk_idx += 1
        total_rows += len(chunk)

        # Filter to labeled rows only
        chunk = chunk[chunk["traffic_change_label"].notna()].copy()
        labeled_rows += len(chunk)

        if len(chunk) == 0:
            continue

        # Derive condition flags
        chunk["is_rain"] = chunk["is_rain_hour"].astype(bool)
        chunk["has_event"] = chunk["has_event_3km"].astype(bool)
        chunk["has_crash_gt1"] = chunk["crash_count"] > 1

        # Build composite key
        chunk["_key"] = (
            chunk["station_key"].astype(str) + "|"
            + chunk["date"].astype(str) + "|"
            + chunk["hour"].astype(str)
        )

        frames.append(chunk)

        if chunk_idx % 5 == 0:
            elapsed = time.time() - t0
            print(f"  Chunk {chunk_idx}: ~{total_rows:,} total rows, "
                  f"~{labeled_rows:,} labeled, {elapsed:.0f}s")

    df = pd.concat(frames, ignore_index=True)
    nan_rows = total_rows - labeled_rows
    elapsed = time.time() - t0
    print(f"  Done: {total_rows:,} total, {labeled_rows:,} labeled, "
          f"{nan_rows:,} NaN, {elapsed:.0f}s")

    return df


def compute_sampling_targets(meta: pd.DataFrame) -> dict:
    """Compute stratified sampling targets from metadata DataFrame."""
    print("\nComputing sampling targets...")

    # Population counts
    pop_label = meta["traffic_change_label"].value_counts().to_dict()
    pop_year = meta["year"].value_counts().to_dict()
    pop_rain = int(meta["is_rain"].sum())
    pop_event = int(meta["has_event"].sum())
    pop_crash = int(meta["has_crash_gt1"].sum())

    # Tier 1: Label minima with proportional year split
    label_targets = {"decrease": 1000, "normal": 1000, "increase": 1000}
    targets_per_label_year = {}

    for label, total in label_targets.items():
        pool = meta[meta["traffic_change_label"] == label]
        year_counts = pool["year"].value_counts().to_dict()
        years_present = sorted(year_counts.keys())
        n_years = len(years_present) if years_present else 1
        # Equal split across years to enforce balance
        base = max(total // n_years, 120)
        remainder = total - base * n_years
        for i, yr in enumerate(years_present):
            target = base + (1 if i < remainder else 0)
            targets_per_label_year[(label, int(yr))] = min(target, year_counts[yr])

    # Tier 1 total
    tier1_total = sum(targets_per_label_year.values())
    print(f"  Tier 1 (label minima): {tier1_total} samples across "
          f"{len(targets_per_label_year)} (label, year) cells")

    # Check condition coverage in Tier 1 naturally
    # (We can't know exactly without sampling, but estimate from proportions)
    est_rain = int(tier1_total * pop_rain / len(meta))
    est_event = int(tier1_total * pop_event / len(meta))
    est_crash = int(tier1_total * pop_crash / len(meta))
    print(f"  Estimated condition coverage: rain={est_rain}, "
          f"event={est_event}, crash={est_crash}")

    event_supplement = 0
    if est_event < MIN_PER_CONDITION:
        event_supplement = MIN_PER_CONDITION - est_event + 50  # buffer
        print(f"  Tier 2 (event supplement): +{event_supplement}")

    print(f"  Target total: ~{tier1_total + event_supplement}")

    return {
        "label_targets": label_targets,
        "targets_per_label_year": targets_per_label_year,
        "event_supplement": event_supplement,
        "pop_label": pop_label,
        "pop_year": pop_year,
        "pop_rain": pop_rain,
        "pop_event": pop_event,
        "pop_crash": pop_crash,
        "pop_total": len(meta),
    }


def stratified_sample(meta: pd.DataFrame, targets: dict) -> pd.DataFrame:
    """Execute stratified sampling on metadata DataFrame."""
    t0 = time.time()
    print("\nSampling from metadata...")

    rng = np.random.default_rng(RANDOM_SEED)
    samples = []
    used_keys = set()

    targets_per_label_year = targets["targets_per_label_year"]

    # Phase 1: Per (label, year) cell
    for (label, year), target_n in sorted(targets_per_label_year.items()):
        pool = meta[(meta["traffic_change_label"] == label) & (meta["year"] == year)]
        n = min(target_n, len(pool))
        if n > 0:
            chosen = pool.sample(n=n, random_state=RANDOM_SEED)
            samples.append(chosen)
            used_keys.update(chosen["_key"].values)

    sample_df = pd.concat(samples, ignore_index=True)

    # Phase 2: Event supplement
    event_supp = targets["event_supplement"]
    if event_supp > 0:
        event_pool = meta[
            meta["has_event"]
            & (~meta["_key"].isin(used_keys))
        ]
        n_evt = min(event_supp, len(event_pool))
        if n_evt > 0:
            evt_sample = event_pool.sample(n=n_evt, random_state=RANDOM_SEED)
            sample_df = pd.concat([sample_df, evt_sample], ignore_index=True)
            used_keys.update(evt_sample["_key"].values)

    # Phase 3: Year balance check
    year_counts = sample_df["year"].value_counts().to_dict()
    year_low = int(TARGET_TOTAL * (1 / 3) * (1 - YEAR_TOLERANCE))  # ~900
    for yr in [2022, 2023, 2024]:
        cur = year_counts.get(yr, 0)
        if cur < year_low:
            need = year_low - cur
            yr_pool = meta[
                (meta["year"] == yr)
                & (~meta["_key"].isin(used_keys))
            ]
            n_yr = min(need, len(yr_pool))
            if n_yr > 0:
                yr_sample = yr_pool.sample(n=n_yr, random_state=RANDOM_SEED)
                sample_df = pd.concat([sample_df, yr_sample], ignore_index=True)
                used_keys.update(yr_sample["_key"].values)

    # Phase 4: Road × LGA coverage
    pop_combos = set(zip(meta["road_name"], meta["lga"]))
    sample_combos = set(zip(sample_df["road_name"], sample_df["lga"]))
    coverage = len(sample_combos) / len(pop_combos) if pop_combos else 0
    if coverage < 0.50 and len(sample_df) < 3200:
        uncovered = pop_combos - sample_combos
        uc_meta = meta[
            meta.apply(lambda r: (r["road_name"], r["lga"]) in uncovered, axis=1)
            & (~meta["_key"].isin(used_keys))
        ]
        n_uc = min(100, len(uc_meta))
        if n_uc > 0:
            uc_sample = uc_meta.sample(n=n_uc, random_state=RANDOM_SEED)
            sample_df = pd.concat([sample_df, uc_sample], ignore_index=True)
            used_keys.update(uc_sample["_key"].values)

    # Phase 5: Trim if over target
    if len(sample_df) > 3200:
        # Prefer removing from normal/slight labels to reach ~3000
        excess = len(sample_df) - TARGET_TOTAL
        removable = sample_df[
            sample_df["traffic_change_label"].isin(["normal"])
        ]
        if len(removable) >= excess:
            drop_idx = removable.sample(n=excess, random_state=RANDOM_SEED).index
            sample_df = sample_df.drop(drop_idx)

    elapsed = time.time() - t0
    print(f"  Selected {len(sample_df):,} candidates from {len(meta):,} labeled rows "
          f"({100*len(sample_df)/len(meta):.2f}%), {elapsed:.0f}s")

    return sample_df


def pass2_read_selected(input_path: Path, selected_df: pd.DataFrame) -> list:
    """Pass 2: chunk-read full columns, extract only selected rows."""
    print("\nPass 2: Reading selected rows...")
    t0 = time.time()

    # Build lookup set
    selected_keys = set(
        zip(
            selected_df["station_key"].values,
            selected_df["date"].astype(str).values,
            selected_df["hour"].astype(int).values,
        )
    )
    print(f"  Looking up {len(selected_keys)} keys")

    extracted = []
    scanned = 0

    for chunk in pd.read_csv(input_path, usecols=PASS2_COLS, chunksize=CHUNKSIZE,
                             low_memory=False):
        scanned += len(chunk)

        # Build keys for this chunk
        chunk_keys = list(zip(
            chunk["station_key"].values,
            chunk["date"].astype(str).values,
            chunk["hour"].astype(int).values,
        ))

        # Find matches
        mask = [k in selected_keys for k in chunk_keys]
        matched = chunk[mask]

        if len(matched) > 0:
            extracted.extend(matched.to_dict(orient="records"))

        if len(extracted) >= len(selected_keys):
            break

    elapsed = time.time() - t0
    print(f"  Extracted {len(extracted)} / {len(selected_keys)} rows, "
          f"scanned {scanned:,}, {elapsed:.0f}s")

    return extracted


# ============================================================================
# Section 3: QA Pair Construction
# ============================================================================

def _safe_float(val) -> float:
    """Convert value to float, returning 0.0 for NaN."""
    try:
        v = float(val)
        return 0.0 if np.isnan(v) or np.isinf(v) else v
    except (ValueError, TypeError):
        return 0.0


def _safe_int(val) -> int:
    """Convert value to int, returning 0 for NaN."""
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0


def _safe_str(val) -> str:
    """Convert value to string, returning '' for NaN/None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return str(val)


def _day_name(day_int) -> str:
    """Convert day_of_week integer (0-6) to name."""
    try:
        idx = int(day_int)
        if 0 <= idx <= 6:
            return DAY_NAMES[idx]
    except (ValueError, TypeError):
        pass
    return str(day_int)


def _build_sample_data(row: dict, descriptions: dict) -> dict:
    """Build the sample_data dict for one QA pair."""
    is_rain = bool(row.get("is_rain_hour", False))

    def g(key, default=None):
        """Get value with NaN→default handling."""
        val = row.get(key)
        if val is None:
            return default
        if isinstance(val, float) and np.isnan(val):
            return default
        return val

    return {
        # Station identity
        "station_key": _safe_int(row.get("station_key")),
        "station_name": _safe_str(row.get("name")),
        "road_type": _safe_str(row.get("road_name")),
        "lga": _safe_str(row.get("lga")),
        "suburb": _safe_str(row.get("suburb")),
        "latitude": _safe_float(row.get("wgs84_latitude")),
        "longitude": _safe_float(row.get("wgs84_longitude")),
        # Temporal
        "date": _safe_str(row.get("date")),
        "hour": _safe_int(row.get("hour")),
        "day_of_week": _day_name(g("day_of_week", 0)),
        "is_weekend": bool(g("is_weekend", False)),
        "year": _safe_int(row.get("year")),
        "month": _safe_int(row.get("month")),
        # Volume / labels
        "volume": _safe_float(row.get("volume")),
        "expected_volume": _safe_float(row.get("expected_volume")),
        "traffic_change_pct": _safe_float(row.get("traffic_change_pct")),
        "traffic_change_label": _safe_str(row.get("traffic_change_label")),
        "has_significant_change": bool(g("has_significant_change", False)),
        # Weather
        "rain_mm": _safe_float(row.get("rain_mm")),
        "rain_intensity": _safe_str(row.get("rain_intensity")),
        "is_rain_hour": is_rain,
        "temperature_c": _safe_float(row.get("temperature_2m_c")),
        "apparent_temperature_c": _safe_float(row.get("apparent_temperature_c")),
        "relative_humidity_2m_pct": _safe_float(row.get("relative_humidity_2m_pct")),
        "humidity_category": _safe_str(row.get("humidity_category")),
        "windspeed_10m_kmh": _safe_float(row.get("windspeed_10m_kmh")),
        "wind_speed_category": _safe_str(row.get("wind_speed_category")),
        "is_strong_wind_hour": bool(g("is_strong_wind_hour", False)),
        "cloud_cover_pct": _safe_float(row.get("cloud_cover_pct")),
        "cloud_category": _safe_str(row.get("cloud_category")),
        "weather_context_label": _safe_str(row.get("weather_context_label")),
        # Events
        "event_count_3km": _safe_int(row.get("event_count_3km")),
        "has_event_3km": bool(g("has_event_3km", False)),
        "nearest_event_dist_km": (None if pd.isna(g("nearest_event_dist_km"))
                                  else _safe_float(g("nearest_event_dist_km"))),
        "nearest_event_type": _safe_str(row.get("nearest_event_type")),
        "nearest_event_location": _safe_str(row.get("nearest_event_location")),
        # Crashes
        "crash_count": _safe_int(row.get("crash_count")),
        "has_crash": bool(g("has_crash", False)),
        "crash_fatal_count": _safe_int(row.get("crash_fatal_count")),
        # Calendar
        "public_holiday": bool(g("public_holiday", False)),
        "school_holiday": bool(g("school_holiday", False)),
        # POI
        "poi_food_count_500m": _safe_int(row.get("poi_food_count_500m")),
        "poi_education_count_500m": _safe_int(row.get("poi_education_count_500m")),
        "poi_healthcare_count_500m": _safe_int(row.get("poi_healthcare_count_500m")),
        "poi_public_transport_count_500m": _safe_int(row.get("poi_public_transport_count_500m")),
        "poi_leisure_count_500m": _safe_int(row.get("poi_leisure_count_500m")),
        "poi_tourism_count_500m": _safe_int(row.get("poi_tourism_count_500m")),
        "poi_shop_count_500m": _safe_int(row.get("poi_shop_count_500m")),
        "building_count_500m": _safe_int(row.get("building_count_500m")),
        # Land use
        "landuse_top1_category": _safe_str(row.get("landuse_top1_category_500m")),
        "landuse_top1_ratio": _safe_float(row.get("landuse_top1_ratio_500m")),
        "landuse_top2_category": _safe_str(row.get("landuse_top2_category_500m")),
        "landuse_top2_ratio": _safe_float(row.get("landuse_top2_ratio_500m")),
        # Generated NL descriptions
        "rain_description": descriptions["rain_description"],
        "visibility": descriptions["visibility"],
        "land_use_description": descriptions["land_use_description"],
        "top_poi_categories": descriptions["top_poi_categories"],
        "event_text": descriptions["event_text"],
        "crash_text": descriptions["crash_text"],
        "holiday_text": descriptions["holiday_text"],
    }


def _build_qa_pair(row: dict, qa_index: int) -> dict:
    """Build a single QA pair dict for Task 1."""
    descriptions = _build_all_descriptions(row)
    sample_data = _build_sample_data(row, descriptions)

    label = sample_data["traffic_change_label"]
    option = LABEL_TO_OPTION.get(label, "B")

    return {
        "qa_id": f"T1_{qa_index:05d}",
        "task": 1,
        "sample_data": sample_data,
        "task_config": {
            "task": 1,
            "prompt_level": "L4",
            "answer_option": option,
            "answer_label": label,
            "requires_cot": False,
        },
    }


def _validate_qa_pair(qa: dict) -> tuple:
    """Validate a QA pair. Returns (is_valid, error_message)."""
    d = qa["sample_data"]
    tc = qa["task_config"]

    # Required fields
    required = ["station_key", "station_name", "date", "hour",
                "traffic_change_label", "expected_volume", "volume"]
    for f in required:
        if f not in d:
            return False, f"missing required field: {f}"

    # Valid answer option
    if tc["answer_option"] not in {"A", "B", "C"}:
        return False, f"invalid answer option: {tc['answer_option']}"

    # Label-option mapping correct
    expected_opt = LABEL_TO_OPTION.get(tc["answer_label"], "B")
    if tc["answer_option"] != expected_opt:
        return False, (f"label-option mismatch: {tc['answer_label']} "
                       f"→ {tc['answer_option']} (expected {expected_opt})")

    # No NaN in critical fields
    critical = ["expected_volume", "temperature_c", "rain_mm", "cloud_cover_pct"]
    for f in critical:
        val = d.get(f)
        if val is not None and isinstance(val, float) and np.isnan(val):
            return False, f"NaN in critical field: {f}"

    return True, ""


# ============================================================================
# Section 4: Statistics & Verification
# ============================================================================

def _compute_qa_stats(qa_pairs: list, meta: pd.DataFrame,
                      targets: dict, skipped: int) -> dict:
    """Compute comprehensive QA statistics."""
    labels = [q["task_config"]["answer_label"] for q in qa_pairs]
    options = [q["task_config"]["answer_option"] for q in qa_pairs]

    # Label distribution
    label_dist = {}
    for lbl in LABEL_ORDER:
        cnt = labels.count(lbl)
        pop = targets["pop_label"].get(lbl, 0)
        pop_pct = 100 * pop / targets["pop_total"] if targets["pop_total"] else 0
        label_dist[lbl] = {
            "count": cnt,
            "pct": round(100 * cnt / len(qa_pairs), 1),
            "population_pct": round(pop_pct, 1),
        }

    # Condition coverage
    sd = [q["sample_data"] for q in qa_pairs]
    cond_rain = sum(1 for d in sd if d["is_rain_hour"])
    cond_event = sum(1 for d in sd if d["has_event_3km"])
    cond_crash = sum(1 for d in sd if d["crash_count"] > 1)
    cond_holiday = sum(1 for d in sd if d["public_holiday"])
    cond_school = sum(1 for d in sd if d["school_holiday"])

    condition_coverage = {
        "rain": {"count": cond_rain, "pct": round(100 * cond_rain / len(qa_pairs), 1),
                 "population_pct": round(100 * targets["pop_rain"] / targets["pop_total"], 1)},
        "event": {"count": cond_event, "pct": round(100 * cond_event / len(qa_pairs), 1),
                  "population_pct": round(100 * targets["pop_event"] / targets["pop_total"], 1),
                  "note": "oversampled to meet >=300 minimum"},
        "crash_gt1": {"count": cond_crash, "pct": round(100 * cond_crash / len(qa_pairs), 1),
                      "population_pct": round(100 * targets["pop_crash"] / targets["pop_total"], 1)},
        "holiday": {"count": cond_holiday,
                    "note": "no labeled rows have public_holiday=True"},
        "school_holiday": {"count": cond_school,
                           "pct": round(100 * cond_school / len(qa_pairs), 1)},
    }

    # Year distribution
    years = [d["year"] for d in sd]
    year_dist = {}
    for yr in sorted(set(years)):
        yr_cnt = years.count(yr)
        year_dist[str(yr)] = {"count": yr_cnt, "pct": round(100 * yr_cnt / len(qa_pairs), 1)}

    # Spatial coverage
    road_types = sorted(set(d["road_type"] for d in sd))
    lgas = sorted(set(d["lga"] for d in sd))
    stations = sorted(set(d["station_key"] for d in sd))

    pop_combos = set(zip(meta["road_name"], meta["lga"]))
    sample_combos = set((d["road_type"], d["lga"]) for d in sd)
    combo_cov = round(100 * len(sample_combos) / len(pop_combos), 1) if pop_combos else 0

    # Option distribution
    option_dist = {}
    for opt in ["A", "B", "C"]:
        option_dist[opt] = options.count(opt)

    # Level field map for runner
    level_field_map = {
        "L0": ["station_name", "road_type", "lga", "day_of_week", "date", "hour",
               "expected_volume"],
        "L1": ["rain_description", "temperature_c", "apparent_temperature_c",
               "relative_humidity_2m_pct", "visibility", "windspeed_10m_kmh",
               "cloud_category"],
        "L2": ["holiday_text", "is_weekend", "school_holiday"],
        "L3": ["event_text"],
        "L4": ["crash_text", "land_use_description", "top_poi_categories",
               "building_count_500m"],
    }

    return {
        "task": 1,
        "generation_info": {
            "script": "generate_qa_pairs.py",
            "generated_at": datetime.now().isoformat(),
            "input_csv": str(INPUT_CSV),
            "input_total_rows": targets["pop_total"] + 55102,
            "input_labeled_rows": targets["pop_total"],
            "input_nan_rows": 55102,
        },
        "sampling_targets": {
            "target_total": TARGET_TOTAL,
            "min_per_label": MIN_PER_LABEL,
            "min_per_condition": MIN_PER_CONDITION,
            "year_balance_tolerance_pct": int(YEAR_TOLERANCE * 100),
        },
        "sample_statistics": {
            "total_qa_pairs": len(qa_pairs),
            "total_skipped": skipped,
            "label_distribution": label_dist,
            "condition_coverage": condition_coverage,
            "year_distribution": year_dist,
            "spatial_coverage": {
                "unique_road_types": len(road_types),
                "unique_lgas": len(lgas),
                "unique_road_lga_combinations": len(pop_combos),
                "sample_road_lga_combinations": len(sample_combos),
                "coverage_pct": combo_cov,
                "unique_stations": len(stations),
            },
            "answer_option_distribution": option_dist,
            "level_field_map": level_field_map,
        },
    }


def _verify(qa_pairs: list, stats: dict) -> list:
    """Run all verification checks. Returns list of (label, passed, detail)."""
    checks = []
    sd = [q["sample_data"] for q in qa_pairs]

    # 1. Total count
    ok = 2800 <= len(qa_pairs) <= 3300
    checks.append(("Total count ~3000", ok, f"actual: {len(qa_pairs)}"))

    # 2. Unique QA IDs
    ids = [q["qa_id"] for q in qa_pairs]
    ok = len(ids) == len(set(ids))
    checks.append(("All QA IDs unique", ok, ""))

    # 3. No duplicate sample keys
    keys = [(d["station_key"], d["date"], d["hour"]) for d in sd]
    ok = len(keys) == len(set(keys))
    checks.append(("No duplicate sample keys", ok, ""))

    # 4. Valid answer options
    ok = all(q["task_config"]["answer_option"] in {"A", "B", "C", "D", "E"}
             for q in qa_pairs)
    checks.append(("All answer options valid", ok, ""))

    # 5. Label-option mapping correct
    ok = all(
        LABEL_TO_OPTION.get(q["task_config"]["answer_label"], "C")
        == q["task_config"]["answer_option"]
        for q in qa_pairs
    )
    checks.append(("Label-option mapping correct", ok, ""))

    # 6. Required fields present
    required = ["station_key", "station_name", "date", "hour",
                "traffic_change_label", "expected_volume", "volume"]
    ok = all(all(f in d for f in required) for d in sd)
    checks.append(("Required fields present", ok, ""))

    # 7. No NaN in critical fields
    critical = ["expected_volume", "temperature_c", "rain_mm", "cloud_cover_pct"]
    ok = all(
        not any(isinstance(d.get(f), float) and np.isnan(d.get(f))
                for f in critical)
        for d in sd
    )
    checks.append(("No NaN in critical fields", ok, ""))

    # Distribution checks
    label_dist = stats["sample_statistics"]["label_distribution"]
    for lbl in LABEL_ORDER:
        cnt = label_dist[lbl]["count"]
        ok = cnt >= MIN_PER_LABEL
        checks.append((f"label '{lbl}' >= {MIN_PER_LABEL}", ok,
                       f"actual: {cnt}"))

    cond = stats["sample_statistics"]["condition_coverage"]
    for ckey in ["rain", "event", "crash_gt1"]:
        cnt = cond[ckey]["count"]
        ok = cnt >= MIN_PER_CONDITION
        checks.append((f"condition '{ckey}' >= {MIN_PER_CONDITION}", ok,
                       f"actual: {cnt}"))

    yr_dist = stats["sample_statistics"]["year_distribution"]
    for yr, info in yr_dist.items():
        cnt = info["count"]
        ok = abs(cnt - 1000) <= 150
        checks.append((f"year {yr} near 1000", ok, f"actual: {cnt}"))

    combo_cov = stats["sample_statistics"]["spatial_coverage"]["coverage_pct"]
    ok = combo_cov >= 50
    checks.append(("Road x LGA coverage >= 50%", ok, f"actual: {combo_cov}%"))

    return checks


# ============================================================================
# Section 5: Main Pipeline
# ============================================================================

def _print_prompt_sample(qa_pair: dict):
    """Print a rendered L4 prompt for human spot-check."""
    d = qa_pair["sample_data"]
    parts = [
        f'A traffic monitoring station "{d["station_name"]}" on {d["road_type"]} '
        f'in {d["lga"]} area.',
    ]
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

    rd = d.get("rain_description", "")
    tc = d.get("temperature_c", 0)
    rh = d.get("relative_humidity_2m_pct", 0)
    vis = d.get("visibility", "")
    parts.append(f"Weather: {rd}, temperature {tc:.0f}°C, "
                 f"humidity {rh:.0f}%, visibility {vis}.")

    for key in ["event_text", "crash_text", "holiday_text"]:
        txt = d.get(key, "")
        if txt:
            parts.append(txt)

    parts.append(
        f"\nThe typical traffic volume for this station at this time is approximately "
        f'{d["expected_volume"]:.0f} vehicles per hour.'
    )

    parts.append("""
Based on all the conditions above, please predict the traffic volume change
compared to what is typical for this station at this day-of-week and hour:
(A) Lower than typical
    (traffic volume more than 10% below the expected level for this station at this time)
(B) Close to typical
    (traffic volume within ±10% of the expected level — normal fluctuation)
(C) Higher than typical
    (traffic volume more than 10% above the expected level for this station at this time)

Please answer one option.
Answer: The answer is option (""")

    print("\n" + "─" * 72)
    print(f"SPOT CHECK — {qa_pair['qa_id']} (L4 prompt rendering)")
    print(f"Ground truth: {qa_pair['task_config']['answer_label']} "
          f"→ option ({qa_pair['task_config']['answer_option']})")
    print("─" * 72)
    print("\n".join(parts))


def main() -> int:
    """Main pipeline: metadata → sample → extract → build → verify → output."""
    print("=" * 65)
    print("generate_qa_pairs.py — Task 1 QA Pair Generation")
    print("=" * 65)
    print(f"Input:   {INPUT_CSV}")
    print(f"Output:  {OUTPUT_JSONL}")
    print(f"Stats:   {OUTPUT_STATS}")
    print(f"Seed:    {RANDOM_SEED}")

    t_start = time.time()

    # ---- Pass 1: Metadata collection ----
    meta = pass1_collect_metadata(INPUT_CSV)

    # ---- Compute sampling targets ----
    targets = compute_sampling_targets(meta)

    # ---- Stratified sampling ----
    selected = stratified_sample(meta, targets)

    # ---- Pass 2: Full context extraction ----
    rows = pass2_read_selected(INPUT_CSV, selected)

    # ---- Build QA pairs ----
    print("\nBuilding QA pairs...")
    qa_pairs = []
    skipped = 0

    for i, row in enumerate(rows, start=1):
        try:
            qa = _build_qa_pair(row, i)
            valid, err = _validate_qa_pair(qa)
            if valid:
                qa_pairs.append(qa)
            else:
                skipped += 1
                print(f"  WARNING: Skipping row {i}: {err}")
        except Exception as e:
            skipped += 1
            print(f"  WARNING: Error building QA pair {i}: {e}")

    # Re-index after skips
    for j, qa in enumerate(qa_pairs, start=1):
        qa["qa_id"] = f"T1_{j:05d}"

    print(f"  Generated {len(qa_pairs)} QA pairs, {skipped} skipped")

    # ---- Compute statistics ----
    print("\nComputing statistics...")
    stats = _compute_qa_stats(qa_pairs, meta, targets, skipped)

    # ---- Write JSONL ----
    print(f"\nWriting {len(qa_pairs)} QA pairs to {OUTPUT_JSONL}...")
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
    size_mb = OUTPUT_JSONL.stat().st_size / (1024 * 1024)
    print(f"  Written: {len(qa_pairs)} lines, {size_mb:.1f} MB")

    # ---- Write statistics ----
    print(f"Writing stats to {OUTPUT_STATS}...")
    with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    stats_size_kb = OUTPUT_STATS.stat().st_size / 1024
    print(f"  Written: {stats_size_kb:.1f} KB")

    # ---- Verification ----
    print("\n" + "=" * 65)
    print("Verification")
    print("=" * 65)
    checks = _verify(qa_pairs, stats)
    passed = 0
    failed = 0
    for label, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        detail_str = f" ({detail})" if detail else ""
        print(f"  {status}: {label}{detail_str}")
        if ok:
            passed += 1
        else:
            failed += 1

    # ---- Distribution summary ----
    print(f"\nLabel distribution:")
    for lbl in LABEL_ORDER:
        info = stats["sample_statistics"]["label_distribution"][lbl]
        req = "PASS" if info["count"] >= MIN_PER_LABEL else "BELOW"
        print(f"  {lbl:15s}: {info['count']:5d} ({info['pct']:5.1f}%, "
              f"target >= {MIN_PER_LABEL})  {req}")

    print(f"\nCondition coverage:")
    for ckey in ["rain", "event", "crash_gt1"]:
        info = stats["sample_statistics"]["condition_coverage"][ckey]
        req = "PASS" if info["count"] >= MIN_PER_CONDITION else "BELOW"
        print(f"  {ckey:12s}: {info['count']:5d} ({info['pct']:5.1f}%, "
              f"target >= {MIN_PER_CONDITION})  {req}")
    sch = stats["sample_statistics"]["condition_coverage"]["school_holiday"]
    print(f"  school_holiday: {sch['count']:5d} ({sch['pct']:5.1f}%)")

    print(f"\nYear distribution:")
    for yr, info in stats["sample_statistics"]["year_distribution"].items():
        req = "PASS" if abs(info["count"] - 1000) <= 150 else "IMBALANCE"
        print(f"  {yr}: {info['count']:5d} ({info['pct']:5.1f}%)  {req}")

    sp = stats["sample_statistics"]["spatial_coverage"]
    print(f"\nSpatial coverage:")
    print(f"  Road types:      {sp['unique_road_types']}")
    print(f"  LGAs:            {sp['unique_lgas']}")
    print(f"  Road x LGA:      {sp['sample_road_lga_combinations']} / "
          f"{sp['unique_road_lga_combinations']} ({sp['coverage_pct']}%)")
    print(f"  Stations:        {sp['unique_stations']} / 118")

    print(f"\nAnswer option distribution:")
    for opt in ["A", "B", "C"]:
        cnt = stats["sample_statistics"]["answer_option_distribution"][opt]
        print(f"  Option {opt}: {cnt:5d}")

    # ---- Spot check ----
    print("\n" + "=" * 65)
    print("Spot Check (5 random QA pairs)")
    print("=" * 65)
    rng = np.random.default_rng(RANDOM_SEED)
    spot_indices = rng.choice(len(qa_pairs), size=min(5, len(qa_pairs)), replace=False)
    for idx in spot_indices:
        qa = qa_pairs[int(idx)]
        label = qa["task_config"]["answer_label"]
        opt = qa["task_config"]["answer_option"]
        name = qa["sample_data"]["station_name"]
        date = qa["sample_data"]["date"]
        rain = qa["sample_data"]["rain_description"]
        event = qa["sample_data"]["event_text"] or "(none)"
        crash = qa["sample_data"]["crash_text"] or "(none)"
        print(f"  [{qa['qa_id']}] {name} | {date} | label={label} opt={opt}")
        print(f"        rain: {rain}")
        print(f"        event: {event}")
        print(f"        crash: {crash}")
        print()

    # ---- Full prompt rendering for 1 pair ----
    spot_qa = qa_pairs[int(spot_indices[0])]
    _print_prompt_sample(spot_qa)

    # ---- Final summary ----
    elapsed = time.time() - t_start
    print("\n" + "=" * 65)
    if failed == 0:
        print(f"All {passed} checks passed.")
    else:
        print(f"{passed} passed, {failed} FAILED — review warnings above.")
    print(f"\nOutput files:")
    print(f"  {OUTPUT_JSONL}  ({len(qa_pairs)} QA pairs, {size_mb:.1f} MB)")
    print(f"  {OUTPUT_STATS}  ({stats_size_kb:.1f} KB)")
    print(f"Total time: {elapsed:.0f}s")
    print("=" * 65)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
