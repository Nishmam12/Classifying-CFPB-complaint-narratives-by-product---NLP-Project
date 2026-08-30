"""Sequential driver for the full rebuild. Stops at the first failing stage.

Each stage caches its results, so re-running this after a fix resumes from
wherever it stopped instead of re-deriving the pipeline.
"""
import subprocess
import sys
import time

STEPS = [
    ("setup (representations on full train split)", ["common.py"]),
    ("classical tuning (9 runs)", ["stages.py", "classical"]),
    ("recurrent tuning (18 runs)", ["stages.py", "recurrent"]),
    ("BERT tuning (3 runs)", ["stages.py", "bert"]),
    ("held-out test evaluation (10 models)", ["evaluate.py"]),
    ("novelty: imbalance-strategy comparison", ["novelty.py"]),
]


def main():
    t_all = time.time()
    for i, (label, cmd) in enumerate(STEPS, 1):
        print(f"\n{'='*70}\n[{i}/{len(STEPS)}] {label}\n{'='*70}", flush=True)
        t0 = time.time()
        rc = subprocess.call([sys.executable, "-u", *cmd])
        if rc != 0:
            print(f"\n!!! STAGE FAILED: {label} (exit {rc}) after {time.time()-t0:.0f}s",
                  flush=True)
            return rc
        print(f"--- {label} done in {time.time()-t0:.0f}s", flush=True)
    print(f"\nALL STAGES COMPLETE in {(time.time()-t_all)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
