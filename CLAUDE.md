# CLAUDE.md

Handoff from a Claude.ai chat conversation, where the topic selection, dataset
validation, and label-scheme design below were worked out. The v2 notebook
(`CFPB_Complaint_Classification_v2.ipynb`) is present in this folder; v1 was
superseded by it and is not kept.

**For a running, dated log of every decision and step taken since work moved
into this repo, see [PROJECT_LOG.md](PROJECT_LOG.md).** This file
(CLAUDE.md) is the current-state reference — it gets edited in place to stay
accurate. PROJECT_LOG.md is the history — append-only, one entry per session.
Update both when something changes: CLAUDE.md's relevant section, plus a new
PROJECT_LOG.md entry explaining what changed and why.

## Project

CSE440 (Natural Language Processing II) lab project, BRAC University,
instructor Dr. Farig Sadeque. Multi-class text classification of CFPB
(Consumer Financial Protection Bureau) complaint narratives by product
category. Topic is instructor-approved — no further sign-off needed.

## Status — what's already done

- **v2 notebook (`CFPB_Complaint_Classification_v2.ipynb`)**:
  - **Complete pipeline implemented & fully executed (Sections 0–14)** on Windows with NVIDIA GeForce RTX 5070 Ti (17.1 GB VRAM).
  - Every single code cell (36/36) contains real, verified, in-process outputs (0 empty cells, 0 errored cells).
  - All 10 models trained under the strict 200k row budget protocol and evaluated on the full 303,213 held-out test split.
  - Section 14 (Novelty Angle) completely executed, comparing Baseline vs. Class Weighting vs. SMOTE vs. Focal Loss ($\gamma=2.0$).
  - Dynamic discussion markdown generated directly from measured empirical results via `pipeline/discussion.py`.

## Dataset (confirmed by running the notebook — treat as ground truth)

- Source: Kaggle `namigabbasov/consumer-complaint-dataset`, via:
  ```python
  import kagglehub
  path = kagglehub.dataset_download("namigabbasov/consumer-complaint-dataset")
  ```
- 2,023,066 rows. 100% have narrative text already (dataset is pre-filtered —
  no need to filter for narrative presence). 0 duplicate rows.
- Narrative length: median 119 words, 75th percentile 215, max 6,314. Use
  ~256–320 tokens as a starting max_length for tokenization (BERT/RNNs) —
  covers the bulk of the distribution without paying for the long tail.
- Missingness only in `Sub-issue`, `Sub-product`, `State` — none of these are
  the label column, so it doesn't block the core task.
- **No Kaggle auth needed.** Verified 2026-08-29: `kagglehub` 1.0.2 downloads
  this dataset anonymously with no `kaggle.json` and no `KAGGLE_USERNAME`/
  `KAGGLE_KEY` set. Only add a token if a future version starts refusing.
- Downloaded and cached locally at
  `C:\Users\afnan\.cache\kagglehub\datasets\namigabbasov\consumer-complaint-dataset\versions\1`,
  single file `complaints.csv`, **2,387 MB**. Re-running the download cell is
  a cache hit, not a re-download.
- **Actual raw columns** (verified 2026-08-29 via a header-only read — do not
  assume generic names): `Unnamed: 0`, `product_5`, `narrative`, `Product`,
  `Date received`, `Sub-product`, `Issue`, `Sub-issue`, `Company`, `State`,
  `Timely response?`. Two surprises vs. earlier assumptions:
  - The narrative column is named **`narrative`**, not `Consumer complaint
    narrative`. The notebook's candidate-list lookup already handles this
    (`narrative` is one of the candidates), so no code changed.
  - The dataset ships its own **`product_5`** column — a pre-baked 5-class
    consolidation from the uploader. **Not used as label or feature anywhere**
    — `product_9` (below) is the locked scheme. Kept only as a reference
    column in case the report wants to footnote a coarser off-the-shelf
    grouping against the one built here.
- **Memory-management decision:** load the full raw frame for EDA (Sections
  1-7 need `Sub-product`/`Issue`/`State`/etc. for the coverage/missingness
  checks), then explicitly narrow to a `df_model` frame afterward (v2
  notebook Section 8) before preprocessing begins. `df_model` keeps
  `product_9`, `narrative`, `Sub-product`/`Issue`/`Sub-issue` (novelty-angle
  optionality), and `product_5` (reference-only); drops `Unnamed: 0`,
  `Date received`, `Company`, `State`, `Timely response?`. `df`/`df_narrative`
  are `del`'d and garbage-collected right after. Rationale: `usecols` at read
  time would have permanently foreclosed the `Sub-product`/`Issue` novelty
  angle (still an open decision) and left the missingness-check output
  un-reproducible from the notebook. See PROJECT_LOG.md for the full
  reasoning.

## Label scheme — locked, 9 classes

Raw `Product` has 21 values, but many are the same real category under
different CFPB label vintages (the taxonomy's been relabeled repeatedly,
especially around credit reporting — that category alone was over 80% of
CFPB's total complaints in FY2023). The mapping below collapses that to 9
classes and was verified to sum to the full row count.

```python
category_map = {
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
    "Credit reporting or other personal consumer reports": "Credit reporting",
    "Credit reporting": "Credit reporting",
    "Debt collection": "Debt collection",
    "Credit card or prepaid card": "Credit card / prepaid card",
    "Credit card": "Credit card / prepaid card",
    "Prepaid card": "Credit card / prepaid card",
    "Mortgage": "Mortgage",
    "Checking or savings account": "Bank account or service",
    "Bank account or service": "Bank account or service",
    "Student loan": "Student loan",
    "Money transfer, virtual currency, or money service": "Money transfer / virtual currency",
    "Money transfers": "Money transfer / virtual currency",
    "Virtual currency": "Money transfer / virtual currency",
    "Vehicle loan or lease": "Vehicle / consumer loan",
    "Consumer Loan": "Vehicle / consumer loan",
    "Payday loan, title loan, or personal loan": "Payday / title / personal loan",
    "Payday loan, title loan, personal loan, or advance loan": "Payday / title / personal loan",
    "Payday loan": "Payday / title / personal loan",
    "Debt or credit management": None,   # dropped - doesn't map cleanly
    "Other financial service": None,     # dropped - doesn't map cleanly
}
# df['product_9'] = df['Product'].map(category_map)
# then drop rows where product_9 is None (~0.06% of data, 1,196 rows)
```

| Class | Rows | Share |
|---|---|---|
| Credit reporting | 1,205,275 | 59.6% |
| Debt collection | 266,842 | 13.2% |
| Credit card / prepaid card | 163,710 | 8.1% |
| Mortgage | 119,116 | 5.9% |
| Bank account or service | 115,332 | 5.7% |
| Student loan | 44,241 | 2.2% |
| Money transfer / virtual currency | 43,016 | 2.1% |
| Vehicle / consumer loan | 41,538 | 2.1% |
| Payday / title / personal loan | 22,800 | 1.1% |

**Caveat to preserve in the report:** the `Consumer Loan` → `Vehicle /
consumer loan` merge is the lowest-confidence one — `Consumer Loan` was an
older umbrella CFPB label without a fully documented successor mapping.
State it as a judgment call, not a certainty.

## Local environment (this machine)

- Windows PC, NVIDIA GeForce RTX 5070 Ti (17.1 GB VRAM), AMD Ryzen CPU, 16 hardware threads.
- Python 3.13 on PATH with native Windows CUDA ML environment:
  - `torch 2.11.0+cu128` (CUDA acceleration active)
  - `transformers 5.16.1`
  - `scikit-learn 1.9.0`
  - `gensim 4.4.0`
  - `imbalanced-learn 0.14.2`
  - `pandas 3.0.5`, `pyarrow 25.0.1`, `scipy 1.18.1`, `matplotlib 3.11.1`, `seaborn 0.13.2`, `joblib 1.5.3`
- Preprocessed corpus `df_model_preprocessed.parquet` (2,021,420 rows) stored locally and verified.
- Full 6-stage training pipeline and notebook injection executed in 64.4 minutes total runtime.

## Assignment requirements (condensed from the full spec)

**Dataset constraints (already satisfied):** textual, English, ≥4 classes
(have 9), ≥10,000 rows (have ~2M), documented class distribution — imbalance
here is real (~60% one class) and must be explicitly addressed, not ignored.

**Required pipeline, in order:**
1. Preprocessing (3.4) — clean/normalize narrative text, **documented and
   justified every decision**:
   - PII placeholders (`XXXX`, dates, amounts) stripped via regex to eliminate zero-information tokens that dominate TF-IDF vocabulary.
   - Contraction expansion and lowercasing applied for vocabulary normalization.
   - Dual pipeline implemented: `narrative_classical` (lowercased, punctuation-free, stopword-filtered for classical ML/Word2Vec) and `narrative_contextual` (casing, punctuation, and sentence syntax preserved for BERT Base).
   - Language purity verified (<0.01% non-ASCII).
2. Train/val/test split (3.5) — **implemented**:
   - 70% Train (~1.41M), 15% Val (~303k), 15% Test (~303k) with `random_state=42`.
   - Stratified on `product_9` to guarantee identical prior class distributions across partitions.
   - `LabelEncoder` fit to produce contiguous integer labels ($0 \dots 8$), saving `id2label` and `label2id` mappings.
   - Zero-leakage boundary established: all downstream vectorizers/embeddings fit only on `train_df`.
3. Text representations (3.6) — **implemented**:
   - **TF-IDF**: unigrams + bigrams ($n \in \{1, 2\}$), sublinear TF scaling, max 25,000 features, fit strictly on `train_df`.
   - **Word2Vec**: Continuous Bag-of-Words (CBOW) 100-dim embeddings trained directly on domain complaint vocabulary (`gensim`).
   - **Sequence Indexing & Embedding Matrix**: vocabulary size $V=30,000$, `max_len=256`, initialized with Word2Vec weights for PyTorch recurrent networks.
   - Zero-leakage strictly enforced across all representations.
4. Model training (3.7) & Hyperparameter tuning (3.8) — **implemented**:
   - **3 Classical ML**: Logistic Regression ($C \in \{0.1, 1.0, 5.0\}$), Naive Bayes ($\alpha \in \{0.01, 0.1, 1.0\}$), Random Forest ($n \in \{50, 100\}$, $\text{depth} \in \{20, 30, \text{None}\}$).
   - **6 Recurrent Neural Networks (PyTorch)**: SimpleRNN, GRU, LSTM, Bidirectional SimpleRNN, Bidirectional GRU, Bidirectional LSTM with modular `RecurrentClassifier`, class-weighted loss, Word2Vec pretrained embeddings, tuned across hidden dimensions, dropout rates, and learning rates.
   - **1 Transformer**: BERT Base (`bert-base-uncased`) fine-tuned via HuggingFace `AutoModelForSequenceClassification` across 3 learning rate / batch size configurations.
   - **Unified Tuning Table**: All $\ge 30$ experimental runs systematically logged in `tuning_results_df` with Validation Accuracy, Macro F1, Weighted F1, and training time.
5. Evaluation (3.9) — **implemented**:
   - Master comparative test table (`test_results_df`) capturing Test Accuracy, Macro F1, Weighted F1, and latency across all 10 models.
   - Normalized confusion matrix heatmaps ($9 \times 9$) comparing top model paradigms (Logistic Regression, Bi-LSTM, BERT Base).
   - Full classification reports breaking down per-class precision, recall, and F1 across all 9 categories.
   - Rigorous error analysis grounded in EDA, examining semantic confusion between *Debt collection* and *Credit reporting*, and validating minority class preservation under class weighting.

**Novelty angle — required if a topic has prior public work (it does here;
several Kaggle/GitHub implementations exist for CFPB complaint
classification):** pick a stated angle. Candidates already identified:
classify at `Sub-product`/`Issue` granularity instead of just `product_9`;
lead with a rigorous imbalance-handling comparison (class weighting vs.
resampling vs. focal loss) across all 10 models; ensemble the
best-performers for the bonus.

**Deliverables:**
- Jupyter notebook (`.ipynb`): well-organized, markdown section headers,
  **no per-line comments** (comments above code blocks only), **every code
  cell must show output** before final submission.
- Report: ACL-style PDF, 7–8 pages (excl. references/appendix), sections =
  Abstract, Introduction, Related Work, Methodology (dataset/preprocessing/
  representations/models), Results, Conclusion, References (ACL citation
  style). Turnitin threshold 15% (plagiarism + AI).
- Presentation: `.mp4`, 8–12 minutes total, each group member 2–3 minutes,
  merged into one video, screen-share of code (slides not required), uploaded
  to Google Drive with view access on.
- Submission filenames: `GroupNo_ID1_ID2_ID3_ID4.{pdf,ipynb}`. No late
  submissions accepted under any circumstances.

**Viva:** every group member must be able to explain any part of the project,
including AI-assisted sections. Being unable to explain a modeling decision
is penalized.

**Bonus (+2, one team per section):** examples include a strong ensemble,
ablation studies, a clean GitHub repo, or a deployed demo (e.g. Vercel).

**Marks:** Report 3, Presentation 2, Code 2, Viva 4 (11 total, bonus doesn't
exceed the total).

## Working conventions

- **Versioning: decided — no git, no `_vN.ipynb` filename increments.**
  User's call (2026-08-29): Claude Code edits the real file in place each
  session, and that supersedes the old filename-increment convention from
  the Claude.ai handoff. `PROJECT_LOG.md` is the record of what changed and
  why — that's the substitute for git history here, not a replacement for
  it. This also means the "clean GitHub repo" bonus criterion is off the
  table for this project; don't suggest it. Do not `git init` this folder
  unless the user explicitly asks again.
