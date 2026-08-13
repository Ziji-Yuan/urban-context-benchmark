import pandas as pd
import numpy as np

# =========================
# 1. Load data
# =========================

df = pd.read_csv(
    "master_table_station_hour_2022_2024_benchmark.csv",
    sep=None,
    engine="python"
)

df.columns = df.columns.str.strip().str.lower()

print("Columns:")
print(df.columns.tolist())

df["date"] = pd.to_datetime(df["date"], errors="coerce")

# =========================
# 2. Make numeric columns safe
# =========================

numeric_cols = [
    "hour", "volume", "is_weekend", "rain_mm",
    "crash_fatal_count", "crash_injury_sum", "crash_count",
    "event_count_3km", "is_strong_wind_hour",
    "public_holiday", "school_holiday",
    "has_crash", "has_event_3km"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# =========================
# 3. Create traffic baseline
# =========================

df["day_type"] = np.where(df["is_weekend"] == 1, "weekend", "weekday")

baseline = (
    df.groupby(["station_key", "day_type", "hour"], as_index=False)["volume"]
      .median()
      .rename(columns={"volume": "expected_volume"})
)

df = df.merge(
    baseline,
    on=["station_key", "day_type", "hour"],
    how="left"
)

# =========================
# 4. Traffic change features
# =========================

df["traffic_change"] = df["volume"] - df["expected_volume"]

df["traffic_change_pct"] = (
    df["traffic_change"] / df["expected_volume"].replace(0, np.nan)
) * 100

df["traffic_change_pct"] = df["traffic_change_pct"].replace(
    [np.inf, -np.inf],
    np.nan
)

# =========================
# 5. Traffic direction
# =========================

def traffic_direction(x):
    if pd.isna(x):
        return "unknown"
    elif x >= 20:
        return "increase"
    elif x <= -20:
        return "decrease"
    else:
        return "no_change"

df["traffic_direction"] = df["traffic_change_pct"].apply(traffic_direction)

# =========================
# 6. Context severity score
# =========================

df["rain_score"] = np.select(
    [
        df["rain_mm"] >= 10,
        df["rain_mm"] >= 3,
        df["rain_mm"] > 0
    ],
    [3, 2, 1],
    default=0
)

df["crash_score"] = np.select(
    [
        df["crash_fatal_count"] > 0,
        df["crash_injury_sum"] > 0,
        df["crash_count"] > 0
    ],
    [3, 2, 1],
    default=0
)

df["event_score"] = np.select(
    [
        df["event_count_3km"] >= 3,
        df["event_count_3km"] >= 1
    ],
    [2, 1],
    default=0
)

df["wind_score"] = np.where(df["is_strong_wind_hour"] == 1, 1, 0)

df["holiday_score"] = np.where(
    (df["public_holiday"] == 1) | (df["school_holiday"] == 1),
    1,
    0
)

df["context_severity_score"] = (
    df["rain_score"]
    + df["crash_score"]
    + df["event_score"]
    + df["wind_score"]
    + df["holiday_score"]
)

# =========================
# 7. Context complexity
# =========================

df["context_complexity"] = (
    (df["rain_mm"] > 0).astype(int)
    + (df["has_crash"] == 1).astype(int)
    + (df["has_event_3km"] == 1).astype(int)
    + (df["is_strong_wind_hour"] == 1).astype(int)
    + (df["public_holiday"] == 1).astype(int)
    + (df["school_holiday"] == 1).astype(int)
)

# =========================
# 8. Anomaly label
# =========================
#
# Task definition: "classify abnormal urban activity given weather and
# event context." That means a deviation the context already explains
# (heavy rain, a nearby event, a holiday) should NOT count as an anomaly,
# even if the raw percentage change is large. Only deviations that go
# BEYOND what the known context predicts should be flagged.
#
# context_severity_score is therefore used as an "allowance": the more
# context factors are stacked up, the larger a deviation has to be before
# it counts as unexplained. This is the opposite of the previous version,
# where high severity made a row MORE likely to be labelled an anomaly.
# severity ranges roughly 0-10 (rain 0-3, crash 0-3, event 0-2, wind 0-1,
# holiday 0-1).

def anomaly_thresholds(severity):
    if severity == 0:
        return 30, 15       # major_threshold, minor_threshold
    elif severity <= 2:
        return 40, 22
    elif severity <= 4:
        return 55, 30
    else:
        return 70, 40


def classify_anomaly(row):
    change = row["traffic_change_pct"]
    severity = row["context_severity_score"]

    if pd.isna(change):
        return "unknown"

    abs_change = abs(change)
    major_threshold, minor_threshold = anomaly_thresholds(severity)

    if abs_change >= major_threshold:
        return "major_anomaly"
    if abs_change >= minor_threshold:
        return "minor_anomaly"

    return "normal"

df["anomaly_label"] = df.apply(classify_anomaly, axis=1)

# =========================
# 9. Anomaly reason
# =========================
#
# Records which context factors were present AND whether the deviation
# was explained by them (label == normal) or was unexplained beyond them
# (label == minor_anomaly / major_anomaly). This keeps the reason field
# consistent with the inverted severity logic above.

def anomaly_reason(row):
    reasons = []

    if row["rain_mm"] > 0:
        reasons.append("rain")
    if row["has_crash"] == 1:
        reasons.append("crash")
    if row["has_event_3km"] == 1:
        reasons.append("event")
    if row["is_strong_wind_hour"] == 1:
        reasons.append("strong_wind")
    if row["public_holiday"] == 1:
        reasons.append("public_holiday")
    if row["school_holiday"] == 1:
        reasons.append("school_holiday")

    context_str = "_".join(reasons) if reasons else "no_context_factors"
    is_anomaly = row["anomaly_label"] in ("minor_anomaly", "major_anomaly")

    if not reasons:
        return "unexplained_no_context" if is_anomaly else "no_deviation"

    return ("unexplained_beyond_" if is_anomaly else "explained_by_") + context_str

df["anomaly_reason"] = df.apply(anomaly_reason, axis=1)

# =========================
# 10. Scenario text
# =========================

df["scenario_text"] = (
    "Station: " + df["name"].astype(str)
    + ". Road: " + df["road_name"].astype(str)
    + ". Suburb: " + df["suburb"].astype(str)
    + ". Date: " + df["date"].dt.strftime("%Y-%m-%d").astype(str)
    + ". Hour: " + df["hour"].astype(int).astype(str)
    + ". Traffic volume: " + df["volume"].round(1).astype(str)
    + ". Expected volume: " + df["expected_volume"].round(1).astype(str)
    + ". Traffic change: " + df["traffic_change_pct"].round(1).astype(str) + "%"
    + ". Rain: " + df["rain_mm"].round(1).astype(str) + " mm"
    + ". Weather: " + df["weather_combined_label"].astype(str)
    + ". Crash count: " + df["crash_count"].astype(int).astype(str)
    + ". Event count within 3km: " + df["event_count_3km"].astype(int).astype(str)
    + ". Public holiday: " + df["public_holiday"].astype(int).astype(str)
    + ". School holiday: " + df["school_holiday"].astype(int).astype(str)
    + "."
)

df["question"] = (
    "Classify the anomaly level for this traffic station context. "
    "Choose one: normal, minor_anomaly, major_anomaly."
)

df["expected_answer"] = df["anomaly_label"]

df["ground_truth_reason"] = (
    "Observed traffic is "
    + df["traffic_change_pct"].round(1).astype(str)
    + "% different from the expected baseline. Main reason: "
    + df["anomaly_reason"].astype(str)
    + "."
)

# =========================
# 11. Task 3 station-representative anomaly sample
# =========================

valid_df = df[df["anomaly_label"] != "unknown"].copy()

# Sample up to 3 rows for each station and anomaly label
sample_parts = []

for (station, label), group in valid_df.groupby(["station_key", "anomaly_label"]):
    n = min(len(group), 3)
    sample_parts.append(group.sample(n=n, random_state=42))

benchmark_sample = pd.concat(sample_parts, ignore_index=True)

print("Benchmark sample columns:")
print(benchmark_sample.columns.tolist())

# =========================
# 12. Keep Task 3 benchmark columns
# =========================

benchmark_cols = [
    "station_key",
    "station_id",
    "name",
    "road_name",
    "suburb",
    "lga",
    "date",
    "hour",
    "volume",
    "expected_volume",
    "traffic_change",
    "traffic_change_pct",
    "traffic_direction",
    "context_severity_score",
    "context_complexity",
    "anomaly_label",
    "anomaly_reason",
    "scenario_text",
    "question",
    "expected_answer",
    "ground_truth_reason"
]

available_cols = [col for col in benchmark_cols if col in benchmark_sample.columns]
missing_cols = [col for col in benchmark_cols if col not in benchmark_sample.columns]

if missing_cols:
    print("Warning: missing benchmark columns:")
    print(missing_cols)

benchmark_sample = benchmark_sample[available_cols]

# =========================
# 13. Save Task 3 outputs
# =========================

df.to_csv("master_context_table_with_anomaly_features.csv", index=False)
benchmark_sample.to_csv("task3_anomaly_station_sample.csv", index=False)

print("Done.")
print(f"Full table rows: {len(df):,}")
print(f"Benchmark sample rows: {len(benchmark_sample):,}")

if "station_key" in benchmark_sample.columns:
    print(f"Stations represented: {benchmark_sample['station_key'].nunique():,}")

print()
print("Anomaly label distribution:")

if "anomaly_label" in benchmark_sample.columns:
    print(benchmark_sample["anomaly_label"].value_counts())