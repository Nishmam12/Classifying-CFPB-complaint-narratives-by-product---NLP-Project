"""
Qwen2.5-1.5B adapted with LoRA — a causal-decoder arm for the paradigm comparison.

Ported from the `feat/cfpb-nlp-models-and-report` branch (author: Shoumodip Paul).
The modelling idea is his; this version changes only what stopped the result from
being comparable with the other ten models:

  1. Data now comes from pipeline/common.py, so this arm sees the *same*
     200,000-row stratified train budget, the same validation split and the same
     303,213-row test split as every other model. The original read
     train/val/test_subset.parquet from a machine-local directory with its own
     label mapping and a 5,000-row test set, which is why its numbers could not
     be placed in the master table.
  2. The best epoch is now restored before test evaluation. The original saved
     the best checkpoint to disk but scored whatever weights the final epoch
     happened to leave in memory, so `best_val_f1` and the reported test metrics
     could come from different models.
  3. Absolute E:\\ paths are gone; nothing here is machine-specific.
  4. Inference is timed on its own, in total seconds over the test set, which is
     the unit test_results.json uses. ms/sample is derived, not measured
     separately, and the loss computation is outside the timed region.

Results land in _cache/qwen_lora_results.json with a `protocol` block recording
the split sizes actually used, so the number can never be quoted without its
provenance.
"""
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from common import (NUM_CLASSES, SEED, TRAIN_BUDGET, cpath, log, metrics,
                    save_json, setup)
from stages import device

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MAX_LEN = 256
BATCH_SIZE = 16
GRAD_ACCUM = 2
EPOCHS = 2
LR = 2e-4
WEIGHT_DECAY = 0.01
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def qwen_tokenize(texts, tokenizer, name):
    """Tokenize once and cache, mirroring bert_tokenize in stages.py."""
    ids_f, mask_f = cpath(f"qwen_{name}_ids.npy"), cpath(f"qwen_{name}_mask.npy")
    if os.path.exists(ids_f) and os.path.exists(mask_f):
        return np.load(ids_f), np.load(mask_f)
    log(f"  tokenizing {name} ({len(texts):,} docs) ...")
    t0 = time.time()
    ids = np.zeros((len(texts), MAX_LEN), dtype=np.int32)
    mask = np.zeros((len(texts), MAX_LEN), dtype=np.int8)
    CH = 20_000
    for s in range(0, len(texts), CH):
        enc = tokenizer(list(texts[s:s + CH]), truncation=True,
                        padding="max_length", max_length=MAX_LEN,
                        return_tensors="np")
        ids[s:s + CH] = enc["input_ids"]
        mask[s:s + CH] = enc["attention_mask"]
    np.save(ids_f, ids)
    np.save(mask_f, mask)
    log(f"  tokenized {name} in {time.time() - t0:.0f}s")
    return ids, mask


def loader(ids, mask, y, batch_size, shuffle):
    from torch.utils.data import DataLoader, TensorDataset
    return DataLoader(
        TensorDataset(torch.from_numpy(ids.astype(np.int64)),
                      torch.from_numpy(mask.astype(np.int64)),
                      torch.from_numpy(np.asarray(y).astype(np.int64))),
        batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict(model, dl, dev):
    model.eval()
    out = []
    for b_ids, b_mask, _ in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits = model(input_ids=b_ids.to(dev), attention_mask=b_mask.to(dev)).logits
        out.append(logits.float().argmax(dim=1).cpu().numpy())
    return np.concatenate(out)


def main():
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    S = setup(need_seq=False, need_tfidf=False)["S"]
    y_train, y_val, y_test = S["y_train"], S["y_val"], S["y_test"]
    dev = device()
    log(f"=== Qwen2.5-1.5B LoRA on {dev} ===")
    log(f"  train {len(y_train):,} | val {len(y_val):,} | test {len(y_test):,}")
    if len(y_train) != TRAIN_BUDGET:
        log(f"  WARNING: train split is {len(y_train):,}, not the {TRAIN_BUDGET:,} "
            f"budget the other models used - results are not comparable")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Right padding is required: Qwen2ForSequenceClassification locates the last
    # non-pad token via pad_token_id and pools there.
    tok.padding_side = "right"

    tr_ids, tr_mask = qwen_tokenize(S["train_contextual"], tok, "train")
    va_ids, va_mask = qwen_tokenize(S["val_contextual"], tok, "val")
    te_ids, te_mask = qwen_tokenize(S["test_contextual"], tok, "test")

    train_dl = loader(tr_ids, tr_mask, y_train, BATCH_SIZE, True)
    val_dl = loader(va_ids, va_mask, y_val, BATCH_SIZE * 2, False)
    test_dl = loader(te_ids, te_mask, y_test, BATCH_SIZE * 2, False)

    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_CLASSES, torch_dtype=torch.bfloat16)
    base.config.pad_token_id = tok.pad_token_id
    model = get_peft_model(base, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT, target_modules=TARGET_MODULES,
        modules_to_save=["score"])).to(dev)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"  trainable {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    from common import class_weights
    cw = torch.tensor(class_weights(y_train), dtype=torch.float32).to(dev)
    criterion = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps = (len(train_dl) // GRAD_ACCUM) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    best_f1, best_state, best_epoch = -1.0, None, -1
    t_train = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, t0 = 0.0, time.time()
        opt.zero_grad()
        for step, (b_ids, b_mask, b_y) in enumerate(train_dl):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits = model(input_ids=b_ids.to(dev),
                               attention_mask=b_mask.to(dev)).logits
            loss = criterion(logits.float(), b_y.to(dev)) / GRAD_ACCUM
            loss.backward()
            running += loss.item() * GRAD_ACCUM
            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
            if (step + 1) % 200 == 0:
                log(f"    epoch {epoch} step {step + 1}/{len(train_dl)} "
                    f"loss {running / (step + 1):.4f}")
        m = metrics(y_val, predict(model, val_dl, dev))
        log(f"  epoch {epoch}: val macro_f1={m['macro_f1']:.4f} "
            f"acc={m['accuracy']:.4f} ({time.time() - t0:.0f}s)")
        # Keep the best epoch in memory. The original scored the final epoch
        # regardless of which one validation actually preferred.
        if m["macro_f1"] > best_f1:
            best_f1, best_epoch = m["macro_f1"], epoch
            best_state = copy.deepcopy(model.state_dict())
    train_time = time.time() - t_train

    if best_state is not None:
        model.load_state_dict(best_state)
    log(f"  restored epoch {best_epoch} (val macro_f1={best_f1:.4f})")

    save_dir = cpath("qwen_lora_best")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)

    t0 = time.time()
    preds = predict(model, test_dl, dev)
    infer_time = time.time() - t0
    m = metrics(y_test, preds)
    log(f"  TEST macro_f1={m['macro_f1']:.4f} acc={m['accuracy']:.4f} "
        f"weighted_f1={m['weighted_f1']:.4f} ({infer_time:.1f}s)")

    np.savez_compressed(cpath("qwen_test_predictions.npz"), preds=preds)
    save_json("qwen_lora_results.json", {
        "Model": "Qwen2.5-1.5B (LoRA)",
        "Category": "Causal Decoder SLM",
        "Hyperparameters": json.dumps({
            "r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
            "lr": LR, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM, "max_len": MAX_LEN,
            "target_modules": TARGET_MODULES}, sort_keys=True),
        "Val Macro F1": round(best_f1, 4),
        "Best Epoch": best_epoch,
        "Test Accuracy": round(m["accuracy"], 4),
        "Test Macro F1": round(m["macro_f1"], 4),
        "Test Weighted F1": round(m["weighted_f1"], 4),
        "Training Time (s)": round(train_time, 2),
        "Inference Time (s)": round(infer_time, 2),
        "ms per document": round(1000 * infer_time / len(y_test), 3),
        "trainable_params": trainable,
        "total_params": total,
        "protocol": {
            "train_rows": int(len(y_train)),
            "val_rows": int(len(y_val)),
            "test_rows": int(len(y_test)),
            "comparable_with_test_results_json": bool(len(y_train) == TRAIN_BUDGET),
        },
    })
    log("wrote qwen_lora_results.json")


if __name__ == "__main__":
    main()
