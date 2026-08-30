"""
Add BERT's Class Weighting arm to the novelty results.

novelty.py now computes this inline, but the run in progress on 2026-08-30 was
launched from the earlier version. The class-weighted BERT was already trained
and scored during the Section 13 test evaluation, so its minority-class F1 is
recoverable from the saved predictions at zero compute cost. Idempotent.
"""
import os

import numpy as np

from common import cpath, load_json, log, metrics, save_json, setup
from novelty import minority_f1


def main():
    results = load_json("novelty_results.json")
    if results is None:
        results = load_json("novelty_partial.json")
        if results is None:
            raise SystemExit("no novelty results yet - run novelty.py first")
        log("using novelty_partial.json (run still in progress)")

    if any(r["Model"] == "BERT Base" and r["Strategy"] == "Class Weighting" for r in results):
        log("BERT Class Weighting arm already present - nothing to do")
        return

    npz = cpath("test_predictions.npz")
    if not os.path.exists(npz):
        raise SystemExit("test_predictions.npz missing")
    z = np.load(npz)
    if "BERT Base" not in z.files:
        raise SystemExit("no saved BERT predictions")

    S = setup(need_seq=False, need_tfidf=False)["S"]
    classes = list(S["classes"])
    minority_ids = [classes.index(c) for c in [
        "Payday / title / personal loan", "Vehicle / consumer loan",
        "Money transfer / virtual currency", "Student loan"]]

    preds = z["BERT Base"]
    y_test = S["y_test"]
    m = metrics(y_test, preds)
    mn, per = minority_f1(y_test, preds, minority_ids)
    test_rows = load_json("test_results.json") or []
    bert = [r for r in test_rows if r["Model"] == "BERT Base"]

    row = {
        "Model": "BERT Base", "Strategy": "Class Weighting",
        "Test Accuracy": round(m["accuracy"], 4),
        "Test Macro F1": round(m["macro_f1"], 4),
        "Test Weighted F1": round(m["weighted_f1"], 4),
        "Minority-4 Mean F1": round(mn, 4),
        "Per-class F1": per,
        "Time (s)": bert[0]["Train Time (s)"] if bert else 0.0,
    }

    # Insert before BERT's focal row so the table reads baseline-then-treatment
    idx = next((i for i, r in enumerate(results)
                if r["Model"] == "BERT Base"), len(results))
    results.insert(idx, row)
    save_json("novelty_results.json", results)
    log(f"added BERT Class Weighting: macro={m['macro_f1']:.4f} "
        f"minority4={mn:.4f} ({len(results)} rows total)")


if __name__ == "__main__":
    main()
