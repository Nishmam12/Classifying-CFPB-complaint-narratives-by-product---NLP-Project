"""Pretty-print whatever pipeline results currently exist in _cache/."""
import json
import os
import sys

from common import CACHE, cpath


def load(name):
    p = cpath(name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def table(rows, cols, widths=None):
    if not rows:
        return
    widths = widths or {}
    hdr = "  ".join(c.rjust(widths.get(c, max(len(c), 10))) for c in cols)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print("  ".join(str(r.get(c, "")).rjust(widths.get(c, max(len(c), 10))) for c in cols))


W = {"Model": 24, "Config ID": 14, "Category": 20, "Strategy": 22, "Paradigm": 20,
     "Best Config": 14, "Per-class F1": 0}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("all", "tuning"):
        for name, label in [("tune_classical.json", "Classical"),
                            ("tune_recurrent.json", "Recurrent"),
                            ("tune_recurrent_partial.json", "Recurrent (partial)"),
                            ("tune_bert.json", "BERT"),
                            ("tune_bert_partial.json", "BERT (partial)")]:
            rows = load(name)
            if rows:
                print(f"\n=== Tuning: {label} ({len(rows)} runs) ===")
                cols = ["Model", "Config ID", "Val Macro F1", "Val Accuracy",
                        "Val Weighted F1", "Training Time (s)"]
                if any("Converged" in r for r in rows):
                    cols.append("Converged")
                table(rows, cols, W)

    if which in ("all", "test"):
        rows = load("test_results.json")
        if rows:
            print(f"\n=== Held-out test set ({len(rows)} models) ===")
            table(rows, ["Model", "Paradigm", "Best Config", "Test Macro F1",
                         "Test Accuracy", "Test Weighted F1", "Train Time (s)",
                         "Inference Time (s)"], W)

    if which in ("all", "novelty"):
        for n in ["novelty_results.json", "novelty_partial.json"]:
            rows = load(n)
            if rows:
                print(f"\n=== Novelty: imbalance strategies ({len(rows)} rows) ===")
                table(rows, ["Model", "Strategy", "Test Macro F1", "Minority-4 Mean F1",
                             "Test Accuracy", "Time (s)"], W)
                break


if __name__ == "__main__":
    main()
