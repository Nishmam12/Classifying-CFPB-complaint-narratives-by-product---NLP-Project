"""
Training / tuning / evaluation stages.

Every stage is independently runnable and caches its results to _cache/, so a
crash costs seconds to retry rather than re-deriving the pipeline. No metric is
ever hardcoded: each number written here is computed in this process.

Protocol (the fix for the original apples-to-oranges master table):
  * every architecture trains on the SAME stratified 200k budget
  * every architecture is tuned against the FULL validation split
  * every architecture is scored on the FULL 303,213-row test split
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from common import (CACHE, NUM_CLASSES, SEED, class_weights, cpath, load_json,
                    log, metrics, save_json, setup)

RNN_ARCHES = [
    ("SimpleRNN", "rnn", False),
    ("GRU", "gru", False),
    ("LSTM", "lstm", False),
    ("Bidirectional SimpleRNN", "rnn", True),
    ("Bidirectional GRU", "gru", True),
    ("Bidirectional LSTM", "lstm", True),
]
RNN_CONFIGS = [
    {"hidden_dim": 64, "lr": 1e-3, "dropout": 0.3, "epochs": 4},
    {"hidden_dim": 128, "lr": 1e-3, "dropout": 0.3, "epochs": 4},
    {"hidden_dim": 128, "lr": 5e-4, "dropout": 0.4, "epochs": 4},
]
BERT_CONFIGS = [
    {"lr": 2e-5, "batch_size": 32, "epochs": 2},
    {"lr": 3e-5, "batch_size": 32, "epochs": 2},
    {"lr": 5e-5, "batch_size": 32, "epochs": 2},
]
BERT_MODEL = "bert-base-uncased"


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def record(runs, model, category, config_id, params, m, elapsed, extra=None):
    row = {
        "Model": model, "Category": category, "Config ID": config_id,
        "Hyperparameters": json.dumps(params, sort_keys=True),
        "Val Accuracy": round(m["accuracy"], 4),
        "Val Macro F1": round(m["macro_f1"], 4),
        "Val Weighted F1": round(m["weighted_f1"], 4),
        "Training Time (s)": round(elapsed, 2),
    }
    if extra:
        row.update(extra)
    runs.append(row)
    log(f"  [{model} / {config_id}] macro_f1={m['macro_f1']:.4f} acc={m['accuracy']:.4f} "
        f"({elapsed:.1f}s)")
    return row


# ─────────────────────────── classical ───────────────────────────
def stage_classical():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB

    if load_json("tune_classical.json"):
        log("stage_classical: cache hit"); return
    d = setup(need_seq=False)
    S = d["S"]
    Xtr, Xva, _, _ = d["tfidf"]
    y_train, y_val = S["y_train"], S["y_val"]

    runs = []
    log("=== Classical ML tuning (9 runs) ===")
    # max_iter raised from 200: the original run hit the cap on all three
    # configs and reported metrics from unconverged models.
    for C in [0.1, 1.0, 5.0]:
        t0 = time.time()
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=1000,
                                 solver="lbfgs", random_state=SEED)
        clf.fit(Xtr, y_train)
        m = metrics(y_val, clf.predict(Xva))
        n_iter = int(np.max(clf.n_iter_))
        record(runs, "Logistic Regression", "Classical ML", f"LR-C{C}",
               {"C": C, "class_weight": "balanced", "max_iter": 1000, "solver": "lbfgs"},
               m, time.time() - t0, {"Converged": bool(n_iter < 1000), "n_iter": n_iter})

    for alpha in [0.01, 0.1, 1.0]:
        t0 = time.time()
        clf = MultinomialNB(alpha=alpha).fit(Xtr, y_train)
        m = metrics(y_val, clf.predict(Xva))
        record(runs, "Naive Bayes", "Classical ML", f"NB-a{alpha}", {"alpha": alpha},
               m, time.time() - t0)

    for depth in [20, 30, 50]:
        t0 = time.time()
        clf = RandomForestClassifier(n_estimators=50, max_depth=depth,
                                     class_weight="balanced", random_state=SEED, n_jobs=-1)
        clf.fit(Xtr, y_train)
        m = metrics(y_val, clf.predict(Xva))
        record(runs, "Random Forest", "Classical ML", f"RF-n50-d{depth}",
               {"n_estimators": 50, "max_depth": depth, "class_weight": "balanced"},
               m, time.time() - t0)

    save_json("tune_classical.json", runs)
    log(f"stage_classical: wrote {len(runs)} runs")


# ─────────────────────────── recurrent ───────────────────────────
def stage_recurrent():
    from recurrent import build, make_loader, train_model

    if load_json("tune_recurrent.json"):
        log("stage_recurrent: cache hit"); return
    d = setup(need_tfidf=False)
    S = d["S"]
    seq_tr, seq_va, _, len_tr, len_va, _, emb = d["seq"]
    dev = device()

    train_loader = make_loader(seq_tr, len_tr, S["y_train"], 512, True)
    val_loader = make_loader(seq_va, len_va, S["y_val"], 1024, False)
    cw = torch.tensor(class_weights(S["y_train"]), dtype=torch.float32).to(dev)
    criterion = nn.CrossEntropyLoss(weight=cw)

    runs = []
    log("=== Recurrent tuning (18 runs) ===")
    for name, cell, bi in RNN_ARCHES:
        for i, cfg in enumerate(RNN_CONFIGS, 1):
            full = {**cfg, "cell_type": cell, "bidirectional": bi}
            t0 = time.time()
            model, opt = build(full, emb, NUM_CLASSES, dev)
            m, _ = train_model(model, train_loader, val_loader, criterion, opt,
                               cfg["epochs"], dev, S["y_val"], label=f"{name}/C{i}")
            record(runs, name, "Recurrent Neural Net", f"Config-{i}", cfg, m, time.time() - t0)
            del model, opt
            torch.cuda.empty_cache()
            save_json("tune_recurrent_partial.json", runs)

    save_json("tune_recurrent.json", runs)
    log(f"stage_recurrent: wrote {len(runs)} runs")


# ───────────────────────────── bert ──────────────────────────────
def bert_tokenize(texts, tokenizer, name):
    """Tokenize once and cache; re-tokenising 300k narratives per stage is waste."""
    ids_f, mask_f = cpath(f"bert_{name}_ids.npy"), cpath(f"bert_{name}_mask.npy")
    if os.path.exists(ids_f) and os.path.exists(mask_f):
        return np.load(ids_f), np.load(mask_f)
    log(f"  tokenizing {name} ({len(texts):,} docs) ...")
    t0 = time.time()
    ids = np.zeros((len(texts), 256), dtype=np.int32)
    mask = np.zeros((len(texts), 256), dtype=np.int8)
    CH = 20_000
    for s in range(0, len(texts), CH):
        enc = tokenizer(list(texts[s:s + CH]), truncation=True, padding="max_length",
                        max_length=256, return_tensors="np")
        ids[s:s + CH] = enc["input_ids"]
        mask[s:s + CH] = enc["attention_mask"]
    np.save(ids_f, ids); np.save(mask_f, mask)
    log(f"  tokenized {name} in {time.time()-t0:.1f}s")
    return ids, mask


def bert_loader(ids, mask, y, batch_size, shuffle):
    from torch.utils.data import DataLoader, TensorDataset
    return DataLoader(
        TensorDataset(torch.from_numpy(ids.astype(np.int64)),
                      torch.from_numpy(mask.astype(np.int64)),
                      torch.from_numpy(np.asarray(y).astype(np.int64))),
        batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def bert_predict(model, loader, dev):
    model.eval()
    out = []
    for b_ids, b_mask, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits = model(input_ids=b_ids.to(dev), attention_mask=b_mask.to(dev)).logits
        out.append(torch.argmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(out)


def stage_bert():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if load_json("tune_bert.json"):
        log("stage_bert: cache hit"); return
    d = setup(need_seq=False, need_tfidf=False)
    S = d["S"]
    dev = device()
    tok = AutoTokenizer.from_pretrained(BERT_MODEL)

    tr_ids, tr_mask = bert_tokenize(S["train_contextual"], tok, "train")
    va_ids, va_mask = bert_tokenize(S["val_contextual"], tok, "val")

    # Class weighting applied to BERT too. The original run left BERT on plain
    # cross-entropy while every other model was class-weighted, which confounded
    # the cross-paradigm comparison.
    cw = torch.tensor(class_weights(S["y_train"]), dtype=torch.float32).to(dev)
    criterion = nn.CrossEntropyLoss(weight=cw)

    # Resume support: each config costs ~28 min, so a partial run is worth
    # keeping. Configs already recorded in the partial file are not repeated.
    runs = load_json("tune_bert_partial.json") or []
    done = {r["Config ID"] for r in runs}
    best_f1 = max((r["Val Macro F1"] for r in runs), default=-1.0)
    if done:
        log(f"=== BERT tuning: resuming, {len(done)} config(s) already done: "
            f"{', '.join(sorted(done))} ===")
    else:
        log("=== BERT tuning (3 runs) ===")

    for i, cfg in enumerate(BERT_CONFIGS, 1):
        if f"BERT-Config-{i}" in done:
            log(f"  skipping BERT-Config-{i} (already tuned)")
            continue
        torch.cuda.empty_cache()
        t0 = time.time()
        model = AutoModelForSequenceClassification.from_pretrained(
            BERT_MODEL, num_labels=NUM_CLASSES).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
        loader = bert_loader(tr_ids, tr_mask, S["y_train"], cfg["batch_size"], True)
        vloader = bert_loader(va_ids, va_mask, S["y_val"], cfg["batch_size"] * 4, False)

        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            total = 0.0
            for b_ids, b_mask, b_y in loader:
                b_ids, b_mask, b_y = b_ids.to(dev), b_mask.to(dev), b_y.to(dev)
                opt.zero_grad()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    logits = model(input_ids=b_ids, attention_mask=b_mask).logits
                loss = criterion(logits.float(), b_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total += loss.item()
            log(f"    BERT/C{i} epoch {epoch}/{cfg['epochs']} loss={total/len(loader):.4f}")

        m = metrics(S["y_val"], bert_predict(model, vloader, dev))
        record(runs, "BERT Base", "Transformer", f"BERT-Config-{i}", cfg, m, time.time() - t0)
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            model.save_pretrained(cpath("bert_best")); tok.save_pretrained(cpath("bert_best"))
            save_json("bert_best_cfg.json", {"config_id": f"BERT-Config-{i}", **cfg})
        del model, opt
        torch.cuda.empty_cache()
        save_json("tune_bert_partial.json", runs)

    save_json("tune_bert.json", runs)
    log(f"stage_bert: wrote {len(runs)} runs, best macro_f1={best_f1:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["classical", "recurrent", "bert", "all"])
    a = p.parse_args()
    fns = {"classical": stage_classical, "recurrent": stage_recurrent, "bert": stage_bert}
    for k in (["classical", "recurrent", "bert"] if a.stage == "all" else [a.stage]):
        fns[k]()
