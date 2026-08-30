"""
Final held-out test evaluation for all 10 architectures.

Fixes carried over from the original cell 43:
  * every model trains on the same 200k budget and is scored on the FULL
    303,213-row test split (previously: classical on 1.41M/full test, RNNs on
    200k/2 epochs, BERT on 50k/first 12,500 test rows)
  * the config used per architecture is the one that actually won its tuning
    sweep, read from the tuning JSON rather than hardcoded
  * training time and inference time are recorded separately; the original
    single column was labelled "Inference Time" but timed fit+predict
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from common import (NUM_CLASSES, SEED, class_weights, cpath, load_json, log,
                    metrics, save_json, setup)
from stages import RNN_ARCHES, bert_loader, bert_predict, bert_tokenize, device


def best_config(runs, model_name):
    rows = [r for r in runs if r["Model"] == model_name]
    if not rows:
        raise SystemExit(f"no tuning rows for {model_name} - run the tuning stage first")
    return max(rows, key=lambda r: r["Val Macro F1"])


def add(records, preds_dict, name, paradigm, cfg_id, m, train_s, infer_s, preds):
    records.append({
        "Model": name, "Paradigm": paradigm, "Best Config": cfg_id,
        "Test Accuracy": round(m["accuracy"], 4),
        "Test Macro F1": round(m["macro_f1"], 4),
        "Test Weighted F1": round(m["weighted_f1"], 4),
        "Train Time (s)": round(train_s, 1),
        "Inference Time (s)": round(infer_s, 1),
    })
    preds_dict[name] = preds
    log(f"  {name:26s} macro_f1={m['macro_f1']:.4f} acc={m['accuracy']:.4f} "
        f"(train {train_s:.0f}s / infer {infer_s:.1f}s)")


def main():
    if load_json("test_results.json") and os.path.exists(cpath("test_predictions.npz")):
        log("stage_evaluate: cache hit"); return

    d = setup()
    S = d["S"]
    Xtr, _, Xte, _ = d["tfidf"]
    seq_tr, _, seq_te, len_tr, _, len_te, emb = d["seq"]
    y_train, y_test = S["y_train"], S["y_test"]
    dev = device()

    records = load_json("test_results_partial.json") or []
    preds_dict = {}
    partial_npz = cpath("test_predictions_partial.npz")
    if os.path.exists(partial_npz):
        z = np.load(partial_npz)
        preds_dict = {k: z[k] for k in z.files}

    done = {r["Model"] for r in records}
    if done:
        log(f"=== Held-out test evaluation: resuming, {len(done)} model(s) already evaluated: {', '.join(sorted(done))} ===")

    # ── classical ──
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB

    tuned = load_json("tune_classical.json")
    log("=== Classical ML on held-out test set ===")
    for name, factory in [
        ("Logistic Regression", lambda p: LogisticRegression(
            C=p["C"], class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=SEED)),
        ("Naive Bayes", lambda p: MultinomialNB(alpha=p["alpha"])),
        ("Random Forest", lambda p: RandomForestClassifier(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            class_weight="balanced", random_state=SEED, n_jobs=-1)),
    ]:
        if name in done:
            log(f"  skipping {name} (already evaluated)")
            continue
        row = best_config(tuned, name)
        params = json.loads(row["Hyperparameters"])
        clf = factory(params)
        t0 = time.time(); clf.fit(Xtr, y_train); train_s = time.time() - t0
        t0 = time.time(); preds = clf.predict(Xte); infer_s = time.time() - t0
        add(records, preds_dict, name, "Classical ML", row["Config ID"],
            metrics(y_test, preds), train_s, infer_s, preds)
        save_json("test_results_partial.json", records)
        np.savez_compressed(partial_npz, **preds_dict)

    # ── recurrent ──
    from recurrent import build, make_loader, train_model

    tuned = load_json("tune_recurrent.json")
    log("=== Recurrent networks on held-out test set ===")
    seq_va, len_va = d["seq"][1], d["seq"][4]
    train_loader = make_loader(seq_tr, len_tr, y_train, 512, True)
    val_loader = make_loader(seq_va, len_va, S["y_val"], 1024, False)
    test_loader = make_loader(seq_te, len_te, y_test, 1024, False)
    cw = torch.tensor(class_weights(y_train), dtype=torch.float32).to(dev)
    criterion = nn.CrossEntropyLoss(weight=cw)

    from recurrent import predict

    for name, cell, bi in RNN_ARCHES:
        if name in done:
            log(f"  skipping {name} (already evaluated)")
            continue
        row = best_config(tuned, name)
        cfg = json.loads(row["Hyperparameters"])
        full = {**cfg, "cell_type": cell, "bidirectional": bi}
        torch.cuda.empty_cache()
        model, opt = build(full, emb, NUM_CLASSES, dev)
        # Epoch selection happens on validation; test is touched once, at the end
        t0 = time.time()
        train_model(model, train_loader, val_loader, criterion, opt, cfg["epochs"],
                    dev, S["y_val"], label=name, restore_best=True)
        train_s = time.time() - t0
        t0 = time.time(); preds = predict(model, test_loader, dev); infer_s = time.time() - t0
        add(records, preds_dict, name, "Recurrent Neural Net", row["Config ID"],
            metrics(y_test, preds), train_s, infer_s, preds)
        save_json("test_results_partial.json", records)
        np.savez_compressed(partial_npz, **preds_dict)
        del model, opt
        torch.cuda.empty_cache()

    # ── bert ──
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if "BERT Base" not in done:
        log("=== BERT Base on held-out test set ===")
        tuned = load_json("tune_bert.json")
        row = best_config(tuned, "BERT Base")
        tok = AutoTokenizer.from_pretrained(cpath("bert_best"))
        te_ids, te_mask = bert_tokenize(S["test_contextual"], tok, "test")
        model = AutoModelForSequenceClassification.from_pretrained(cpath("bert_best")).to(dev)
        loader = bert_loader(te_ids, te_mask, y_test, 256, False)
        t0 = time.time(); preds = bert_predict(model, loader, dev); infer_s = time.time() - t0
        add(records, preds_dict, "BERT Base", "Transformer", row["Config ID"],
            metrics(y_test, preds), row["Training Time (s)"], infer_s, preds)
        save_json("test_results_partial.json", records)
        np.savez_compressed(partial_npz, **preds_dict)
    else:
        log("  skipping BERT Base (already evaluated)")

    records.sort(key=lambda r: r["Test Macro F1"], reverse=True)
    save_json("test_results.json", records)
    np.savez_compressed(cpath("test_predictions.npz"), **preds_dict)
    log(f"Wrote test_results.json ({len(records)} models) and test_predictions.npz")


if __name__ == "__main__":
    main()
