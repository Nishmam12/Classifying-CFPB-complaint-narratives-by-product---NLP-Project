# Handoff prompt — full model training on a new PC

Copy the block below into a fresh Claude Code session on the new machine, with the working
directory set to the copied project folder.

**What to copy across (~2.4 GB):**

```
CSE440 Lab Project/
├── df_model_preprocessed.parquet     2.3 GB   preprocessed corpus - REUSED
├── pipeline/                         ~90 KB   all training code
├── CFPB_Complaint_Classification_v2.ipynb
├── CFPB_Complaint_Classification_v2.BACKUP-2026-08-30-1130.ipynb
├── CLAUDE.md
├── PROJECT_LOG.md
└── HANDOFF.md
```

**Do NOT copy `_cache/`.** All model training happens fresh on the new machine, so every
result comes from one consistent run on one GPU. The cache would only mix the two.

---

I'm moving a CSE440 lab project to this PC. Preprocessing and EDA are already done — the
preprocessed corpus is in `df_model_preprocessed.parquet` and should be reused as-is. **All
model training runs fresh here.** Read `CLAUDE.md` and `PROJECT_LOG.md` first.

## Project

9-class classification of CFPB consumer complaint narratives (2,021,420 rows, target column
`product_9`). Ten architectures are compared: Logistic Regression, Naive Bayes, Random
Forest, SimpleRNN, GRU, LSTM, the three bidirectional variants, and BERT Base. The
deliverable is `CFPB_Complaint_Classification_v2.ipynb` with every code cell showing real
output.

Splits are stratified 70/15/15: train 1,414,994 / validation 303,213 / test 303,213.

## What to run

Everything. There is no cache on this machine, so `run_all.py` builds from the parquet
upward:

```bash
cd pipeline
python -u run_all.py
```

| Stage | What it does | Expected time |
|---|---|---|
| 1. setup | splits, TF-IDF (25k features), Word2Vec CBOW 100d, padded sequences | ~12 min |
| 2. classical | 9 tuning runs (LR ×3, NB ×3, RF ×3) | ~4 min |
| 3. recurrent | 18 tuning runs (6 architectures × 3 configs) | ~34 min |
| 4. bert | 3 tuning configs, 2 epochs each | ~84 min |
| 5. evaluate | all 10 models on the full test split | ~20 min |
| 6. novelty | imbalance-strategy comparison | ~60–90 min |

**Total ~3.5–4.5 hours** on an RTX 4070 Ti Super. Scale by your GPU.

Progress appends to `_cache/run_all.log`. Every stage caches its results, so if one fails,
fix it and rerun the same command — completed stages are skipped, nothing is lost.

Then build the notebook:

```bash
python inject.py /tmp/dryrun.ipynb     # dry run first, inspect the audit output
python inject.py                       # writes the real notebook
```

Finally update `CLAUDE.md` and `PROJECT_LOG.md`, and verify the notebook.

## Environment

Python 3.13 with a CUDA GPU. BERT needs ~8 GB VRAM at batch 32. Versions used previously:

```
torch 2.11.0+cu128   transformers 5.5.0   scikit-learn 1.9.0   gensim 4.4.0
imbalanced-learn 0.14.2   scipy 1.18.0   pandas   numpy   matplotlib   seaborn
pyarrow   joblib
```

Newer versions are fine, but note `imbalanced-learn` ≥ 0.11 removed `SMOTE(n_jobs=...)` —
the code already accounts for this.

## Experimental protocol — do not alter

This is the entire point of the rebuild. The original notebook compared models trained on
wildly different budgets, which confounded architecture with training-set size.

- All ten architectures train on the **same stratified 200,000-row budget**.
- All are tuned against the **full validation split**.
- All are scored on the **full test split**.
- Representations (TF-IDF, Word2Vec) are fit on the **full 1.41M** training split; only the
  classifier fitting budget is capped. Validation and test stay transform-only.

The 200k budget is set by what BERT can afford in reasonable time. If this machine has a
much faster GPU and you want to raise it, change `TRAIN_BUDGET` in `pipeline/common.py` —
but it must stay identical for all ten models or the comparison breaks.

## Corrections already applied — do not regress

- **`pack_padded_sequence` on every RNN.** Sequences pad to 256 but the median cleaned
  narrative is 44 tokens, so the old code read a hidden state produced by ~200 padding
  steps. Fixing it took SimpleRNN from 0.2222 to 0.6121 macro F1.
- **Epoch selection uses validation, never test.** Best-epoch weights are restored before a
  single test scoring pass.
- **Logistic Regression `max_iter` 200 → 1000.** All three configs then converge (84, 144,
  277 iterations); previously all three silently hit the cap.
- **Class weighting applied to BERT** as well as everything else, so the cross-paradigm
  comparison isn't confounded by an inconsistent objective.
- **No hardcoded metrics anywhere.** A previous runner hand-typed BERT numbers that appeared
  in no captured output. Every figure must be computed in-process.
- **Discussion markdown is generated from measured results** by `pipeline/discussion.py`,
  including a retraction branch that refuses to claim the novelty finding if the data does
  not support it.

## Reference values from the previous machine

Use these as a sanity check only — **do not copy them into the notebook.** The new run
produces its own numbers, and small differences are expected (Word2Vec is multi-worker and
not bit-reproducible).

Classical, validation macro F1: LR C=1.0 **0.7346**, NB α=0.01 0.7195, RF depth=50 0.6899.

Recurrent: Bi-GRU **0.7201**, Bi-LSTM 0.7178, GRU 0.7153, LSTM 0.7098, Bi-SimpleRNN 0.6748,
SimpleRNN 0.6121.

BERT config 1 (lr 2e-5): **0.7647** macro F1, 0.8558 accuracy.

If the new run lands within roughly ±0.02 of these, it reproduced correctly. A SimpleRNN
score near 0.2 or below means the padding fix regressed.

## Gotchas

- The notebook on disk still carries stale error outputs from an old broken runner.
  `inject.py` replaces them — do not hand-edit. `inject.py` should report
  `cells with NO output: []`.
- Backup of the pre-rebuild notebook:
  `CFPB_Complaint_Classification_v2.BACKUP-2026-08-30-1130.ipynb`.
- SMOTE at the 200k budget over 25,000-dimensional sparse TF-IDF is the one unmeasured step
  and may be slow. `novelty.py` checkpoints to `novelty_partial.json` after every row, so a
  slow or failed SMOTE cannot cost the earlier arms. If it overruns, cap it with
  `python novelty.py --smote-budget 100000`.
- BERT tuning writes `tune_bert_partial.json` after each config and resumes from it, so an
  interrupted BERT stage does not restart from config 1.
- **Do not run any other agent against this notebook concurrently.** An earlier session had
  a second agent repeatedly overwriting it with broken output, which is what caused the
  original mess.

Start by confirming the parquet loads (`python -c "import pandas as pd;
print(pd.read_parquet('df_model_preprocessed.parquet', columns=['product_9']).shape)"` should
print `(2021420, 1)`), then run `run_all.py`.
