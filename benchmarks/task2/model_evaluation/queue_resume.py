"""Resume queue: track existing PIDs, launch remaining models 3-at-a-time."""
import subprocess
import time
import sys

EXISTING = [52044, 40560, 1848]  # deepseek-r1, mixtral, gemma

REMAINING = [
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "qwen/qwen3-32b",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
]

MAX_WORKERS = 3
OUTDIR = "labeled_data/results_adjust"

def pid_alive(pid):
    r = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        capture_output=True, text=True
    )
    return f"{pid}" in r.stdout and "python" in r.stdout.lower()

def main():
    running = {}  # pid -> label
    for pid in EXISTING:
        running[pid] = f"existing:{pid}"

    q = 0
    total = len(REMAINING)

    print(f"=== Tracking {len(EXISTING)} PIDs: {EXISTING} ===")
    print(f"=== Queue ({total}): {REMAINING} ===")

    while q < total or len(running) > 0:
        done = [pid for pid in running if not pid_alive(pid)]
        for pid in done:
            label = running.pop(pid)
            print(f"=== [{label}] DONE. Slot freed ({len(running)}/3 running) ===")

        while q < total and len(running) < MAX_WORKERS:
            model = REMAINING[q]
            print(f"=== [q{q+1}/{total}] Launching: {model} ===")
            proc = subprocess.Popen(
                [sys.executable, "model_evaluation/benchmark_soft_prompt_openrouter_noCoT.py",
                 "--models", model, "--max", "600", "--seed", "42",
                 "--output-dir", OUTDIR],
            )
            running[proc.pid] = model
            q += 1

        time.sleep(10)

    print("=== ALL DONE ===")

if __name__ == "__main__":
    main()
