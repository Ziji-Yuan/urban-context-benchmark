import pandas as pd
import numpy as np

# ======================================================
# Task 3: Anomaly Classification Benchmark Builder
# ======================================================

INPUT_FILE = "master_context_table_with_anomaly_features.csv"
OUTPUT_FILE = "task3_anomaly_classification.csv"
TARGET_N = 600

# ======================================================
# 1. Load data
# ======================================================

df = pd.read_csv(INPUT_FILE, sep=None, engine="python")
df.columns = df.columns.str.strip().str.lower()
df["date"] = pd.to_datetime(df["date"], errors="coerce")


# ======================================================
# 2. Helper functions
# ======================================================

def anomaly_option(label):
    mapping = {
        "normal": "A",
        "minor_anomaly": "B",
        "major_anomaly": "C",
    }
    return mapping.get(label, "unknown")


def clean_text(x):
    if pd.isna(x):
        return "unknown"
    return str(x)


def build_context(row):
    """Build the full contextual description used for Task 3."""
    base = (
        f'A traffic monitoring station "{clean_text(row["name"])}" '
        f'on {clean_text(row["road_name"])} in {clean_text(row["lga"])} area. '
        f'Time: {clean_text(row["day_of_week"])}, '
        f'{row["date"].date()} at {int(row["hour"])}:00.'
    )

    weather = (
        f' Weather: {clean_text(row["weather_combined_label"])}, '
        f'rain {row["rain_mm"]:.1f} mm, '
        f'temperature {row["temperature_2m_c"]:.1f}°C.'
    )

    calendar = (
        f' Public holiday: {int(row["public_holiday"])}. '
        f'School holiday: {int(row["school_holiday"])}. '
        f'Weekend: {int(row["is_weekend"])}.'
    )

    event = (
        f' Nearby events within 3km: {int(row["event_count_3km"])}. '
        f'Nearest event type: {clean_text(row["nearest_event_type"])}.'
    )

    crash = (
        f' Crash count: {int(row["crash_count"])}. '
        f'Injuries: {int(row["crash_injury_sum"])}. '
        f'Fatal crashes: {int(row["crash_fatal_count"])}.'
    )

    urban = (
        f' Urban context: suburb {clean_text(row["suburb"])}, '
        f'land use {clean_text(row["landuse_top1_category_500m"])}, '
        f'food POIs {int(row["poi_food_count_500m"])}, '
        f'education POIs {int(row["poi_education_count_500m"])}, '
        f'public transport POIs {int(row["poi_public_transport_count_500m"])}, '
        f'shop POIs {int(row["poi_shop_count_500m"])}.'
    )

    expected = (
        f' Typical traffic volume for this station at this time is '
        f'approximately {row["expected_volume"]:.1f} vehicles per hour.'
    )

    return base + weather + calendar + event + crash + urban + " " + expected


def sample_balanced(data, label_col, target_n=600, random_state=42):
    """Sample approximately equally across anomaly classes."""
    labels = ["normal", "minor_anomaly", "major_anomaly"]
    per_class = target_n // len(labels)
    remainder = target_n % len(labels)
    parts = []

    for i, label in enumerate(labels):
        group = data[data[label_col] == label]
        n = min(len(group), per_class + (1 if i < remainder else 0))
        if n > 0:
            parts.append(group.sample(n=n, random_state=random_state + i))

    if not parts:
        return data.iloc[0:0].copy()

    sampled = pd.concat(parts, ignore_index=False)

    # Fill any shortfall if a class contains fewer rows than requested.
    if len(sampled) < min(target_n, len(data)):
        remaining = data[~data.index.isin(sampled.index)]
        extra_n = min(target_n - len(sampled), len(remaining))
        if extra_n > 0:
            sampled = pd.concat(
                [
                    sampled,
                    remaining.sample(
                        n=extra_n,
                        random_state=random_state + 99,
                    ),
                ],
                ignore_index=False,
            )

    return sampled.sample(frac=1, random_state=random_state).copy()


# ======================================================
# 3. Prepare Task 3 labels
# ======================================================

df["anomaly_option"] = df["anomaly_label"].apply(anomaly_option)

df = df[
    df["anomaly_label"].isin(
        ["normal", "minor_anomaly", "major_anomaly"]
    )
].copy()


# ======================================================
# 4. Station-representative candidate pool
# ======================================================

# Keep representation across stations and anomaly classes by taking
# up to three observations from each station/class combination.
parts = []

for (station, label), group in df.groupby(
    ["station_key", "anomaly_label"]
):
    n = min(len(group), 3)
    parts.append(group.sample(n=n, random_state=42))

if not parts:
    raise ValueError("No valid Task 3 anomaly rows were found.")

task3_pool = pd.concat(parts, ignore_index=True)


# ======================================================
# 5. Balanced Task 3 sample
# ======================================================

task3 = sample_balanced(
    task3_pool,
    "anomaly_label",
    target_n=TARGET_N,
)

task3["task"] = "task3_anomaly_classification"

task3["question"] = task3.apply(
    lambda r: (
        build_context(r)
        + f' Actual traffic volume was {r["volume"]:.1f} vehicles per hour, '
        + f'which is {r["traffic_change_pct"]:.1f}% different from the '
          'typical volume. '
        + "Considering how much of this deviation the weather, calendar, "
          "event, crash, and urban context above would already explain, "
          "classify the anomaly level: "
          "(A) Normal, the deviation is consistent with what this context "
          "would predict "
          "(B) Minor anomaly, the deviation goes somewhat beyond what this "
          "context explains "
          "(C) Major anomaly, the deviation goes well beyond what this "
          "context explains. "
          "Please answer one option."
    ),
    axis=1,
)

task3["answer"] = task3["anomaly_option"]


# ======================================================
# 6. Save Task 3 benchmark
# ======================================================

keep_cols = [
    "task",
    "question",
    "answer",
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
    "traffic_change_pct",
    "anomaly_label",
    "anomaly_reason",
]

available_cols = [c for c in keep_cols if c in task3.columns]
task3[available_cols].to_csv(OUTPUT_FILE, index=False)

print("Done.")
print(f"Saved {OUTPUT_FILE}: {len(task3):,} rows")
print("\nTask 3 anomaly distribution:")
print(task3["anomaly_label"].value_counts())
print("\nAnswer distribution:")
print(task3["answer"].value_counts())