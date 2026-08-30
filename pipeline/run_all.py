"""Sequential driver for the full rebuild. Stops at the first failing stage.

Each stage caches its results, so re-running this after a fix or power outage
resumes from wherever it stopped instead of re-deriving the pipeline.
"""
import os
import subprocess
import sys
import time

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(PROJECT, "_cache", "run_all.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

STEPS = [
    ("setup (representations on full train split)", ["common.py"]),
    ("classical tuning (9 runs)", ["stages.py", "classical"]),
    ("recurrent tuning (18 runs)", ["stages.py", "recurrent"]),
    ("BERT tuning (3 runs)", ["stages.py", "bert"]),
    ("held-out test evaluation (10 models)", ["evaluate.py"]),
    ("novelty: imbalance-strategy comparison", ["novelty.py"]),
]


def log_and_print(msg, log_file):
    print(msg, flush=True)
    log_file.write(msg + "\n")
    log_file.flush()


def run_step(cmd, log_file):
    proc = subprocess.Popen(
        [sys.executable, "-u", *cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        log_file.write(line)
        log_file.flush()
    return proc.wait()


def main():
    t_all = time.time()
    with open(LOG_PATH, "a", encoding="utf-8", errors="replace") as log_file:
        log_and_print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] === run_all.py session started ===", log_file)
        for i, (label, cmd) in enumerate(STEPS, 1):
            log_and_print(f"\n{'='*70}\n[{i}/{len(STEPS)}] {label}\n{'='*70}", log_file)
            t0 = time.time()
            rc = run_step(cmd, log_file)
            if rc != 0:
                log_and_print(f"\n!!! STAGE FAILED: {label} (exit {rc}) after {time.time()-t0:.0f}s", log_file)
                return rc
            log_and_print(f"--- {label} done in {time.time()-t0:.0f}s", log_file)
        log_and_print(f"\nALL STAGES COMPLETE in {(time.time()-t_all)/60:.1f} min", log_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
