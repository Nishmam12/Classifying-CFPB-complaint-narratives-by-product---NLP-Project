"""
Ensemble of the best-performing models (bonus deliverable).

Protocol, which is the whole point of doing this properly:

  1. Produce class PROBABILITIES for all 10 models on both validation and test.
  2. Search ensemble combinations on VALIDATION only.
  3. Evaluate the single winning combination on test, exactly once.

Selecting the combination on test would inflate the reported gain — a search over
500 combinations will always find one that looks good on the split it was chosen
on. The validation-selected ensemble is the number that belongs in the report.

Probabilities are cached per model, so a crash costs one model, not the stage.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from itertools import combinations

from common import (NUM_CLASSES, SEED, cpath, load_json, log, metrics,
                    save_json, setup)
from stages import RNN_ARCHES, bert_loader, bert_tokenize, device

PROBA_DIR = "proba"


def ppath(split, model):
    safe = model.replace(" ", "_").replace("/", "_")
    d = os.path.join(cpath(PROBA_DIR))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{split}_{safe}.npy")


def have(model):
    return all(os.path.exists(ppath(s, model)) for s in ("val", "test"))


def store(model, val_p, test_p):
    np.save(ppath("val", model), val_p.astype(np.float16))
    np.save(ppath("test", model), test_p.astype(np.float16))


def load_proba(model, split):
    return np.load(ppath(split, model)).astype(np.float32)


# ─────────────────────────── produce probabilities ───────────────────────────
def build_classical(d, S):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB

    Xtr, Xva, Xte, _ = d["tfidf"]
    tuned = load_json("tune_classical.json")

    def best(name):
        rows = [r for r in tuned if r["Model"] == name]
        return json.loads(max(rows, key=lambda r: r["Val Macro F1"])["Hyperparameters"])

    specs = [
        ("Logistic Regression", lambda p: LogisticRegression(
            C=p["C"], class_weight="balanced", max_iter=1000, solver="lbfgs",
            random_state=SEED)),
        ("Naive Bayes", lambda p: MultinomialNB(alpha=p["alpha"])),
        ("Random Forest", lambda p: RandomForestClassifier(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            class_weight="balanced", random_state=SEED, n_jobs=-1)),
    ]
    for name, factory in specs:
        if have(name):
            log(f"  cache hit  {name}")
            continue
        t0 = time.time()
        clf = factory(best(name)).fit(Xtr, S["y_train"])
        store(name, clf.predict_proba(Xva), clf.predict_proba(Xte))
        log(f"  built      {name} in {time.time() - t0:.0f}s")


@torch.no_grad()
def _rnn_proba(model, loader, dev):
    model.eval()
    out = []
    for xb, lb, _ in loader:
        logits = model(xb.to(dev), lb)
        out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(out)


def build_recurrent(d, S):
    from recurrent import build, make_loader, train_model
    from common import class_weights

    seq_tr, seq_va, seq_te, len_tr, len_va, len_te, emb = d["seq"]
    dev = device()
    tuned = load_json("tune_recurrent.json")

    train_loader = make_loader(seq_tr, len_tr, S["y_train"], 512, True)
    val_loader = make_loader(seq_va, len_va, S["y_val"], 1024, False)
    test_loader = make_loader(seq_te, len_te, S["y_test"], 1024, False)
    cw = torch.tensor(class_weights(S["y_train"]), dtype=torch.float32).to(dev)
    criterion = nn.CrossEntropyLoss(weight=cw)

    for name, cell, bi in RNN_ARCHES:
        if have(name):
            log(f"  cache hit  {name}")
            continue
        rows = [r for r in tuned if r["Model"] == name]
        cfg = json.loads(max(rows, key=lambda r: r["Val Macro F1"])["Hyperparameters"])
        torch.cuda.empty_cache()
        t0 = time.time()
        model, opt = build({**cfg, "cell_type": cell, "bidirectional": bi},
                           emb, NUM_CLASSES, dev)
        train_model(model, train_loader, val_loader, criterion, opt, cfg["epochs"],
                    dev, S["y_val"], label=name, restore_best=True)
        store(name, _rnn_proba(model, val_loader, dev), _rnn_proba(model, test_loader, dev))
        log(f"  built      {name} in {time.time() - t0:.0f}s")
        del model, opt
        torch.cuda.empty_cache()


@torch.no_grad()
def _bert_proba(model, loader, dev):
    model.eval()
    out = []
    for b_ids, b_mask, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits = model(input_ids=b_ids.to(dev), attention_mask=b_mask.to(dev)).logits
        out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(out)


def build_bert(S):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if have("BERT Base"):
        log("  cache hit  BERT Base")
        return
    dev = device()
    tok = AutoTokenizer.from_pretrained(cpath("bert_best"))
    va_ids, va_mask = bert_tokenize(S["val_contextual"], tok, "val")
    te_ids, te_mask = bert_tokenize(S["test_contextual"], tok, "test")
    model = AutoModelForSequenceClassification.from_pretrained(cpath("bert_best")).to(dev)
    t0 = time.time()
    store("BERT Base",
          _bert_proba(model, bert_loader(va_ids, va_mask, S["y_val"], 256, False), dev),
          _bert_proba(model, bert_loader(te_ids, te_mask, S["y_test"], 256, False), dev))
    log(f"  built      BERT Base in {time.time() - t0:.0f}s")
    del model
    torch.cuda.empty_cache()


# ───────────────────────────── combine and select ─────────────────────────────
def hard_vote(probas, weights, order):
    """Plurality over each model's argmax; ties broken toward the stronger model."""
    stack = np.stack([p.argmax(axis=1) for p in probas])
    n = stack.shape[1]
    counts = np.zeros((n, NUM_CLASSES), dtype=np.int16)
    for row in stack:
        counts[np.arange(n), row] += 1
    top = counts.max(axis=1, keepdims=True)
    tied = counts == top
    out = stack[0].copy()
    for i in range(stack.shape[0]):
        need = ~tied[np.arange(n), out]
        if not need.any():
            break
        out[need] = stack[i][need]
    return out


def soft_vote(probas, weights, order):
    acc = np.zeros_like(probas[0])
    for p, w in zip(probas, weights):
        acc += w * p
    return acc.argmax(axis=1)


def main():
    d = setup()
    S = d["S"]

    log("=== Stage: class probabilities for all 10 models ===")
    build_classical(d, S)
    build_recurrent(d, S)
    build_bert(S)

    val_f1 = {r["Model"]: r["Val Macro F1"] for r in
              (load_json("tune_classical.json") + load_json("tune_recurrent.json")
               + load_json("tune_bert.json"))}
    names = [n for n in val_f1 if have(n)]
    names = sorted(set(names), key=lambda n: -val_f1[n])
    log(f"models available: {len(names)}")

    val_p = {n: load_proba(n, "val") for n in names}
    y_val, y_test = S["y_val"], S["y_test"]

    log("=== Selecting ensemble on VALIDATION ===")
    trials = []
    for k in (3, 5, 7, 9):
        for combo in combinations(names, k):
            order = sorted(combo, key=lambda n: -val_f1[n])
            probas = [val_p[n] for n in order]
            for scheme, fn in (("hard", hard_vote), ("soft", soft_vote)):
                w_uniform = [1.0] * len(order)
                w_perf = [val_f1[n] for n in order]
                for wname, w in (("uniform", w_uniform), ("val-F1", w_perf)):
                    if scheme == "hard" and wname == "val-F1":
                        continue
                    m = metrics(y_val, fn(probas, w, order))
                    trials.append((m["macro_f1"], scheme, wname, order))
    trials.sort(key=lambda t: -t[0])

    log("top 10 candidates by VALIDATION macro F1:")
    for f1, scheme, wname, order in trials[:10]:
        log(f"    {f1:.4f}  {scheme:4s}/{wname:8s} [{len(order)}]  {', '.join(order)}")

    best_val_f1, scheme, wname, order = trials[0]
    fn = hard_vote if scheme == "hard" else soft_vote
    w = [1.0] * len(order) if wname == "uniform" else [val_f1[n] for n in order]

    log(f"\nSELECTED on validation: {scheme}/{wname} over {len(order)} models "
        f"(val macro F1 {best_val_f1:.4f})")
    log(f"  members: {', '.join(order)}")

    log("=== Scoring the selected ensemble on TEST (once) ===")
    test_probas = [load_proba(n, "test") for n in order]
    preds = fn(test_probas, w, order)
    m = metrics(y_test, preds)

    single = {r["Model"]: r for r in load_json("test_results.json")}
    best_single = max(single.values(), key=lambda r: r["Test Macro F1"])
    delta = (m["macro_f1"] - best_single["Test Macro F1"]) * 100

    log(f"  ensemble  macro_f1={m['macro_f1']:.4f} acc={m['accuracy']:.4f} "
        f"weighted_f1={m['weighted_f1']:.4f}")
    log(f"  best single ({best_single['Model']}) macro_f1="
        f"{best_single['Test Macro F1']:.4f}")
    log(f"  delta {delta:+.2f} pp")

    np.savez_compressed(cpath("ensemble_predictions.npz"), ensemble=preds)
    save_json("ensemble_results.json", {
        "scheme": scheme, "weighting": wname, "members": list(order),
        "selected_on": "validation",
        "val_macro_f1": round(best_val_f1, 4),
        "test_accuracy": round(m["accuracy"], 4),
        "test_macro_f1": round(m["macro_f1"], 4),
        "test_weighted_f1": round(m["weighted_f1"], 4),
        "best_single_model": best_single["Model"],
        "best_single_test_macro_f1": best_single["Test Macro F1"],
        "delta_pp": round(delta, 2),
        "candidates_evaluated": len(trials),
    })
    log("wrote ensemble_results.json")


if __name__ == "__main__":
    main()
