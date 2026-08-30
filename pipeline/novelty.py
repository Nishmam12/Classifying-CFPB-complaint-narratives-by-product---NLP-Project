"""
Section 14 novelty angle: controlled imbalance-handling comparison.

The original Section 14 was confounded — it trained the SMOTE arm on a 100k
subsample and compared it against baselines trained on the full 1.41M, so any
difference mixed the strategy with a 14x change in training data. Here every
arm of every comparison uses the identical 200k budget and the identical full
test split, so the only thing that varies is the imbalance strategy.

Strategies compared:
  * None            - untouched objective, no reweighting
  * Class Weighting - inverse-frequency weights (the project baseline)
  * SMOTE           - synthetic minority oversampling (classical models only)
  * Focal Loss      - gamma=2 down-weighting of easy examples (deep models only)
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (NUM_CLASSES, SEED, class_weights, load_json, log, metrics,
                    save_json, setup)
from stages import bert_loader, bert_predict, bert_tokenize, device


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017)."""

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        log_p = F.log_softmax(logits, dim=-1)
        ce = F.nll_loss(log_p, targets, weight=self.weight, reduction="none")
        pt = torch.exp(log_p).gather(1, targets.unsqueeze(1)).squeeze(1)
        return (((1 - pt) ** self.gamma) * ce).mean()


def minority_f1(y_true, y_pred, minority_ids):
    from sklearn.metrics import f1_score

    per = f1_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)),
                   zero_division=0)
    return float(np.mean([per[i] for i in minority_ids])), [round(float(x), 4) for x in per]


def row(results, model, strategy, m, minority, per_class, elapsed):
    results.append({
        "Model": model, "Strategy": strategy,
        "Test Accuracy": round(m["accuracy"], 4),
        "Test Macro F1": round(m["macro_f1"], 4),
        "Test Weighted F1": round(m["weighted_f1"], 4),
        "Minority-4 Mean F1": round(minority, 4),
        "Per-class F1": per_class,
        "Time (s)": round(elapsed, 1),
    })
    log(f"  {model:22s} {strategy:16s} macro={m['macro_f1']:.4f} "
        f"minority4={minority:.4f} ({elapsed:.0f}s)")
    save_json("novelty_partial.json", results)


def main(smote_budget):
    if load_json("novelty_results.json"):
        log("stage_novelty: cache hit"); return

    d = setup()
    S = d["S"]
    Xtr, _, Xte, _ = d["tfidf"]
    seq_tr, _, seq_te, len_tr, _, len_te, emb = d["seq"]
    y_train, y_test = S["y_train"], S["y_test"]
    dev = device()

    # The four rarest classes by corpus share (Payday 1.1%, Vehicle 2.1%,
    # Money transfer 2.1%, Student loan 2.2%) - the ones imbalance handling
    # is supposed to rescue.
    classes = list(S["classes"])
    minority_ids = [classes.index(c) for c in [
        "Payday / title / personal loan", "Vehicle / consumer loan",
        "Money transfer / virtual currency", "Student loan"]]
    log(f"minority class ids: {minority_ids}")

    results = load_json("novelty_partial.json") or []
    done = {(r["Model"], r["Strategy"]) for r in results}
    if done:
        log(f"=== Novelty imbalance comparison: resuming, {len(done)} row(s) already done ===")

    cw_arr = class_weights(y_train)

    # ── classical: None / Class Weighting / SMOTE ──
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB

    tuned = load_json("tune_classical.json")

    def best_params(name):
        rows = [r for r in tuned if r["Model"] == name]
        return json.loads(max(rows, key=lambda r: r["Val Macro F1"])["Hyperparameters"])

    lr_p, nb_p, rf_p = best_params("Logistic Regression"), best_params("Naive Bayes"), best_params("Random Forest")

    def classical(name, weighted):
        w = "balanced" if weighted else None
        if name == "Logistic Regression":
            return LogisticRegression(C=lr_p["C"], class_weight=w, max_iter=1000,
                                      solver="lbfgs", random_state=SEED)
        if name == "Naive Bayes":
            return MultinomialNB(alpha=nb_p["alpha"])  # NB has no class_weight
        return RandomForestClassifier(n_estimators=rf_p["n_estimators"],
                                      max_depth=rf_p["max_depth"], class_weight=w,
                                      random_state=SEED, n_jobs=-1)

    log("=== Classical: None vs Class Weighting ===")
    for name in ["Logistic Regression", "Naive Bayes", "Random Forest"]:
        for strat, weighted in [("None", False), ("Class Weighting", True)]:
            if name == "Naive Bayes" and strat == "Class Weighting":
                continue  # MultinomialNB exposes no class_weight - documented, not skipped silently
            if (name, strat) in done:
                log(f"  skipping {name} / {strat} (already computed)")
                continue
            t0 = time.time()
            clf = classical(name, weighted).fit(Xtr, y_train)
            preds = clf.predict(Xte)
            mn, per = minority_f1(y_test, preds, minority_ids)
            row(results, name, strat, metrics(y_test, preds), mn, per, time.time() - t0)

    smote_needed = any((name, "SMOTE") not in done for name in ["Logistic Regression", "Naive Bayes", "Random Forest"])
    if smote_needed:
        log(f"=== Classical: SMOTE (budget {smote_budget:,}) ===")
        from imblearn.over_sampling import SMOTE
        from sklearn.model_selection import train_test_split

        if smote_budget < len(y_train):
            idx, _ = train_test_split(np.arange(len(y_train)), train_size=smote_budget,
                                      stratify=y_train, random_state=SEED)
            Xs, ys = Xtr[idx], y_train[idx]
        else:
            Xs, ys = Xtr, y_train
        t0 = time.time()
        Xsm, ysm = SMOTE(random_state=SEED).fit_resample(Xs, ys)
        log(f"  SMOTE: {Xs.shape} -> {Xsm.shape} in {time.time()-t0:.0f}s")
        save_json("smote_meta.json", {"budget": int(smote_budget), "before": list(Xs.shape),
                                      "after": list(Xsm.shape), "seconds": round(time.time()-t0, 1)})

        for name in ["Logistic Regression", "Naive Bayes", "Random Forest"]:
            if (name, "SMOTE") in done:
                log(f"  skipping {name} / SMOTE (already computed)")
                continue
            t0 = time.time()
            clf = classical(name, False).fit(Xsm, ysm)
            preds = clf.predict(Xte)
            mn, per = minority_f1(y_test, preds, minority_ids)
            row(results, name, "SMOTE", metrics(y_test, preds), mn, per, time.time() - t0)

    # ── Bi-LSTM: None / Class Weighting / Focal ──
    from recurrent import build, make_loader, train_model

    log("=== Bi-LSTM: None vs Class Weighting vs Focal Loss ===")
    tuned_r = load_json("tune_recurrent.json")
    rrows = [r for r in tuned_r if r["Model"] == "Bidirectional LSTM"]
    cfg = json.loads(max(rrows, key=lambda r: r["Val Macro F1"])["Hyperparameters"])
    full = {**cfg, "cell_type": "lstm", "bidirectional": True}
    from recurrent import predict

    seq_va, len_va = d["seq"][1], d["seq"][4]
    train_loader = make_loader(seq_tr, len_tr, y_train, 512, True)
    val_loader = make_loader(seq_va, len_va, S["y_val"], 1024, False)
    test_loader = make_loader(seq_te, len_te, y_test, 1024, False)
    cw_t = torch.tensor(cw_arr, dtype=torch.float32).to(dev)

    for strat, crit in [
        ("None", nn.CrossEntropyLoss()),
        ("Class Weighting", nn.CrossEntropyLoss(weight=cw_t)),
        ("Focal Loss (gamma=2)", FocalLoss(gamma=2.0, weight=cw_t)),
    ]:
        if ("Bidirectional LSTM", strat) in done:
            log(f"  skipping Bidirectional LSTM / {strat} (already computed)")
            continue
        torch.cuda.empty_cache()
        model, opt = build(full, emb, NUM_CLASSES, dev)
        t0 = time.time()
        # Epoch selection on validation, so test stays held out for scoring only
        train_model(model, train_loader, val_loader, crit, opt, cfg["epochs"],
                    dev, S["y_val"], label=f"BiLSTM/{strat}", restore_best=True)
        preds = predict(model, test_loader, dev)
        mn, per = minority_f1(y_test, preds, minority_ids)
        row(results, "Bidirectional LSTM", strat, metrics(y_test, preds), mn, per,
            time.time() - t0)
        del model, opt
        torch.cuda.empty_cache()

    # ── BERT: Class Weighting (from main run) / Focal ──
    if ("BERT Base", "Focal Loss (gamma=2)") not in done:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        log("=== BERT: Focal Loss ===")
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        tr_ids, tr_mask = bert_tokenize(S["train_contextual"], tok, "train")
        te_ids, te_mask = bert_tokenize(S["test_contextual"], tok, "test")
        bcfg = load_json("bert_best_cfg.json")

        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=NUM_CLASSES).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=bcfg["lr"], weight_decay=0.01)
        crit = FocalLoss(gamma=2.0, weight=cw_t)
        loader = bert_loader(tr_ids, tr_mask, y_train, bcfg["batch_size"], True)
        t0 = time.time()
        for epoch in range(1, bcfg["epochs"] + 1):
            model.train()
            total = 0.0
            for b_ids, b_mask, b_y in loader:
                b_ids, b_mask, b_y = b_ids.to(dev), b_mask.to(dev), b_y.to(dev)
                opt.zero_grad()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    logits = model(input_ids=b_ids, attention_mask=b_mask).logits
                loss = crit(logits.float(), b_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total += loss.item()
            log(f"    BERT/focal epoch {epoch}/{bcfg['epochs']} loss={total/len(loader):.4f}")
        preds = bert_predict(model, bert_loader(te_ids, te_mask, y_test, 256, False), dev)
        mn, per = minority_f1(y_test, preds, minority_ids)
        row(results, "BERT Base", "Focal Loss (gamma=2)", metrics(y_test, preds), mn, per,
            time.time() - t0)
    else:
        log("  skipping BERT Base / Focal Loss (gamma=2) (already computed)")

    save_json("novelty_results.json", results)
    log(f"Wrote novelty_results.json ({len(results)} rows)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smote-budget", type=int, default=200_000)
    main(p.parse_args().smote_budget)
