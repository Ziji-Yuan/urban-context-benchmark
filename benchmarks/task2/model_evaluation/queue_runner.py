"""Queue manager: run benchmark models 3 at a time."""
import subprocess
import time
import sys

MODELS = [
    "deepseek/deepseek-r1",
    "mistralai/mixtral-8x22b-instruct",
    "google/gemma-2-27b-it",
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "qwen/qwen3-32b",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
]

MAX_WORKERS = 3
OUTDIR = "labeled_data/results_adjust"

def main():
    total = len(MODELS)
    running = {}   # pid -> model_name
    q = 0

    print(f"=== {total} models, max {MAX_WORKERS} concurrent ===")

    while q < total or running:
        # Check for finished processes
        done = [pid for pid, p in running.items() if p.poll() is not None]
        for pid in done:
            rc = running[pid].returncode
            name = running.pop(pid)
            print(f"=== [{name}] done (rc={rc}). Slot freed ===")

        # Fill empty slots
        while q < total and len(running) < MAX_WORKERS:
            model = MODELS[q]
            print(f"=== [{q+1}/{total}] Launching: {model} ===")
            proc = subprocess.Popen(
                [sys.executable, "model_evaluation/benchmark_soft_prompt_openrouter_noCoT.py",
                 "--models", model, "--max", "600", "--seed", "42",
                 "--output-dir", OUTDIR],
            )
            running[proc.pid] = proc
            q += 1

        time.sleep(5)

    print("=== ALL DONE ===")

if __name__ == "__main__":
    main()
