#!/usr/bin/env python3
"""Generate example prompts for the examples/ folder."""
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model_evaluation.benchmark_soft_prompt_openrouter_noCoT import render_prompt

OUTDIR = Path(__file__).resolve().parent.parent / "examples"
ALL_LEVELS = ["L0", "L1", "L2", "L3", "L4"]

# Load all QA pairs
pairs = []
with open("labeled_data/task1_qa_pairs.jsonl", encoding="utf-8") as f:
    for line in f:
        pairs.append(json.loads(line))

random.seed(42)

# ── 1. One QA pair shown at all 5 levels ──────────────────────────
one = random.choice(pairs)
out_path = OUTDIR / "prompt_all_levels.md"
with open(out_path, "w", encoding="utf-8") as f:
    d = one["sample_data"]
    cfg = one["task_config"]
    f.write(f"# Example: One QA Pair at All 5 Information Levels\n\n")
    f.write(f"**Station:** {d['station_name']} ({d['road_type']}, {d['lga']})\n")
    f.write(f"**Time:** {d['day_of_week']} {d['date']} {d['hour']:02d}:00\n")
    f.write(f"**Actual volume:** {d['volume']:.0f} vph | **Expected:** {d['expected_volume']:.0f} vph ({d['traffic_change_pct']:+.1f}%)\n")
    f.write(f"**Correct answer:** {cfg['answer_option']} ({cfg['answer_label']})\n\n")
    for lv in ALL_LEVELS:
        f.write(f"---\n\n## {lv}\n\n")
        f.write(render_prompt(d, lv))
        f.write("\n\n")

print(f"[1/4] {out_path}")

# ── 2. Three L4 prompts (one per label) ───────────────────────────
by_label = {"decrease": [], "normal": [], "increase": []}
for p in pairs:
    by_label[p["task_config"]["answer_label"]].append(p)

out_path = OUTDIR / "sample_l4_prompts.md"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("# Sample L4 Prompts — One Per Label\n\n")
    for lbl in ["decrease", "normal", "increase"]:
        p = random.choice(by_label[lbl])
        d = p["sample_data"]
        cfg = p["task_config"]
        f.write(f"---\n\n## {lbl.upper()}\n\n")
        f.write(f"**Station:** {d['station_name']} | **Time:** {d['day_of_week']} {d['date']} {d['hour']:02d}:00\n")
        f.write(f"**Volume:** {d['volume']:.0f} vs expected {d['expected_volume']:.0f} ({d['traffic_change_pct']:+.1f}%)\n")
        f.write(f"**Answer:** ({cfg['answer_option']}) {lbl}\n\n")
        f.write(render_prompt(d, "L4"))
        f.write("\n\n")

print(f"[2/4] {out_path}")

# ── 3. Fields per level reference ─────────────────────────────────
out_path = OUTDIR / "fields_by_level.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("Fields Included at Each Information Level\n")
    f.write("=========================================\n\n")
    # L0: minimal context (expected_volume only + time+location)
    l0 = ["station_key", "station_name", "road_type", "lga", "suburb",
          "date", "hour", "day_of_week", "is_weekend",
          "expected_volume"]
    # Build field sets from the prompt definitions
    field_sets = {
        "L0": ["station_name", "road_type", "lga", "date", "day_of_week", "hour", "expected_volume"],
        "L1": ["station_name", "road_type", "lga", "date", "day_of_week", "hour", "expected_volume",
               "temperature_c", "apparent_temperature_c", "relative_humidity_2m_pct",
               "windspeed_10m_kmh", "rain_description", "cloud_category", "visibility"],
        "L2": ["station_name", "road_type", "lga", "date", "day_of_week", "is_weekend", "hour",
               "expected_volume", "temperature_c", "relative_humidity_2m_pct",
               "rain_description", "visibility", "holiday_text", "school_holiday"],
        "L3": ["station_name", "road_type", "lga", "date", "day_of_week", "is_weekend", "hour",
               "expected_volume", "temperature_c", "relative_humidity_2m_pct",
               "rain_description", "visibility", "holiday_text", "school_holiday", "event_text"],
        "L4": ["station_name", "road_type", "lga", "date", "day_of_week", "is_weekend", "hour",
               "expected_volume", "temperature_c", "relative_humidity_2m_pct",
               "rain_description", "visibility", "holiday_text", "school_holiday",
               "event_text", "crash_text", "land_use_description", "top_poi_categories"],
    }
    for lv in ALL_LEVELS:
        fields = field_sets.get(lv, [])
        f.write(f"--- {lv} ({len(fields)} fields) ---\n")
        for name in fields:
            f.write(f"  {name}\n")
        f.write("\n")

print(f"[3/4] {out_path}")

# ── 4. raw JSON snippet of one QA pair ────────────────────────────
out_path = OUTDIR / "qa_pair_snippet.json"
with open(out_path, "w", encoding="utf-8") as f:
    # Simplify for readability
    snippet = {
        "qa_id": one["qa_id"],
        "level": one["task_config"]["prompt_level"],
        "task_config": one["task_config"],
        "sample_data": {k: v for k, v in one["sample_data"].items()
                        if k in field_sets["L4"] + ["volume", "traffic_change_pct", "traffic_change_label"]},
    }
    json.dump(snippet, f, indent=2, ensure_ascii=False)

print(f"[4/4] {out_path}")
print("Done — all examples regenerated in examples/")
