"""One-epoch smoke test: verifies packing works and measures throughput."""
import time

import numpy as np
import torch
import torch.nn as nn

from common import class_weights, setup, log, NUM_CLASSES
from recurrent import build, make_loader, train_model

d = setup(need_tfidf=False)
S = d["S"]
seq_tr, seq_va, seq_te, len_tr, len_va, len_te, emb = d["seq"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"device={device}  emb={emb.shape}")
log(f"length stats: mean={len_tr.mean():.1f} median={np.median(len_tr):.0f} "
    f"p95={np.percentile(len_tr,95):.0f} max={len_tr.max()}")

# Evaluate against a 50k val slice for the smoke test only
NVAL = 50_000
train_loader = make_loader(seq_tr, len_tr, S["y_train"], 512, True)
val_loader = make_loader(seq_va[:NVAL], len_va[:NVAL], S["y_val"][:NVAL], 1024, False)

cw = torch.tensor(class_weights(S["y_train"]), dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=cw)

for cell in ["rnn", "gru"]:
    cfg = {"hidden_dim": 128, "lr": 1e-3, "dropout": 0.3, "cell_type": cell, "bidirectional": False}
    model, opt = build(cfg, emb, NUM_CLASSES, device)
    t0 = time.time()
    best, _ = train_model(model, train_loader, val_loader, criterion, opt, 1, device,
                          S["y_val"][:NVAL], label=cell)
    log(f"  {cell}: {time.time()-t0:.1f}s/epoch  macro_f1={best['macro_f1']:.4f}")
    del model
    torch.cuda.empty_cache()
