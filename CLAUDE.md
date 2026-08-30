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

**All experiments are complete and measured (2026-08-30).** Every number in the
notebook is computed; nothing is hardcoded.

- **Notebook:** 57 cells (37 code, 20 markdown), 10 figures. Audited: zero code
  cells without output, zero cells with error output. Sections 0–14 all populated.
- **Experiments run:** 30 tuning runs (9 classical + 18 recurrent + 3 BERT),
  10-model held-out test evaluation, 13 controlled imbalance-strategy runs.
  Total wall clock ≈ 3.5 h on the RTX 4070 Ti Super.
- **Report:** ACL-format LaTeX in `report/` — `acl_report.tex`, `custom.bib`
  (24 entries), `figures/` (10 PNGs exported from notebook outputs). Needs
  `acl.sty` + `acl_natbib.bst` from the ACL style-files repo to compile; no
  LaTeX toolchain is installed on this machine, so it has **not** been
  compile-verified.

### Headline results (held-out test, 303,213 docs)

| Model | Macro F1 | Accuracy | Train (s) | Infer (s) |
|---|---|---|---|---|
| **BERT Base** (best) | 0.7647 | 0.8562 | 1678.1 | 267.3 |
| Logistic Regression | 0.7367 | 0.8390 | 34.5 | 0.2 |
| GRU | 0.7274 | 0.8326 | 77.2 | 5.8 |
| Bidirectional GRU | 0.7234 | 0.8267 | 130.6 | 8.4 |
| Naive Bayes | 0.7211 | 0.8329 | 0.1 | 0.2 |
| Bidirectional LSTM | 0.7173 | 0.8175 | 139.1 | 9.2 |
| LSTM | 0.7110 | 0.8180 | 78.2 | 5.8 |
| Random Forest | 0.6920 | 0.8121 | 31.0 | 0.8 |
| Bidirectional SimpleRNN | 0.6600 | 0.7824 | 124.2 | 8.9 |
| **SimpleRNN** (worst) | 0.5685 | 0.6722 | 80.5 | 5.9 |

### Three findings that contradict earlier drafts of this file

1. **Class weighting usually hurts.** For Logistic Regression, Naive Bayes and
   Bi-LSTM, the best minority-class F1 comes from *no mitigation at all*. Class
   weighting costs Bi-LSTM 7.9 pp on the four rarest classes. It helps decisively
   only for Random Forest (+24.3 pp), which collapses to 0.3389 untreated.
   The earlier claim that weighting "validated minority class preservation" is
   false — it trades precision for recall and F1 penalises the trade.
2. **The dominant confusion is Money transfer → Bank account (14–17%)**, not
   Debt collection ↔ Credit reporting (7–8%). The word clouds explain why: those
   two classes share *account*, *bank*, *money* and the same institution names.
3. **Bidirectionality only helps ungated cells.** SimpleRNN +9.15 pp, LSTM
   +0.63 pp, GRU **−0.40 pp**. The last two are inside the measured ±0.85 pp
   run-to-run variance. The earlier "consistently 3–6 pp" claim is wrong.

### Known limitation to disclose

PyTorch initialisation is **not seeded** (classical models are, via
`random_state=42`). Measured run-to-run variance on recurrent models is
≈ ±0.85 pp Macro F1. Differences smaller than that are ties, and the report says so.

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

## Experimental protocol — fixed training budget (locked)

**Every one of the 10 architectures trains on the same stratified 200,000-row
subsample and is scored on the full 303,213-row test split.**

Why: fine-tuning BERT over the full 1.41M training split is not tractable on one
consumer GPU in this timeframe, so the transformer sets the ceiling. Training the
classical models on 1.41M while BERT saw 50k — which an earlier version of the
notebook did — confounds architecture with training-set size and makes the
ranking uninterpretable. Capping everything at the same budget is what makes the
Section 3.9 table a valid comparison.

Representations (TF-IDF vocabulary, Word2Vec embeddings) are still fit on the
**full 1.41M** training split. Only the classifier fitting budget is capped, and
validation/test remain transform-only, so the zero-leakage boundary holds.

Measured cost of the budget: Logistic Regression scores 0.7346 on 200k vs 0.7373
on the full split — 0.27 pp. Cheap, and worth stating in the viva.

Documented in the notebook as Section 12.1. **Do not change this without
changing it for all ten models.**

## Padding masking — required, do not regress

Recurrent models read their final hidden state through `pack_padded_sequence`.
Sequences pad to 256 but the median *cleaned* narrative is **44 tokens**, so
reading `hidden[-1]` off the padded sequence exposes the classifier to a state
advanced through ~200 `<PAD>` steps.

Fixing this took SimpleRNN from 0.2222 to 0.6121 validation Macro F1 (2.8×) and
narrowed the SimpleRNN-vs-gated gap from 51.7 pp to 10.3 pp. The project's
earlier "SimpleRNN struggled due to severe vanishing gradients" explanation was
mostly a padding artefact.

Note the median: 44 tokens is `narrative_classical` **after** stopword removal.
The 119-word median quoted elsewhere in this file is the **raw** text. Both are
correct; don't conflate them.

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

- Windows 11 PC, username `nabil`, 32GB RAM, RTX 4070 Ti Super (16GB VRAM, Ada
  Lovelace, compute capability 8.9). Project root (moved 2026-08-30):
  `G:\CSE440 Lab Project\CSE440 Lab Project`.
- Python 3.13. **ML stack is installed and verified working** (2026-08-30):
  `torch 2.11.0+cu128` (CUDA available), `transformers 5.5.0`,
  `scikit-learn 1.9.0`, `gensim 4.4.0`, `imbalanced-learn 0.14.2`,
  `scipy 1.18.0`, `wordcloud 1.9.6`, plus pandas/numpy/matplotlib/seaborn/
  pyarrow/joblib.
- **GPU thermals under sustained load:** peaks at 89 °C with fan at 91% and
  software thermal throttling active (hardware throttling never triggered — it
  is safe, just cooling-limited at ~84% of its 285 W power budget). Clocks drop
  to ~2760 of 3165 MHz. BERT epochs take ~11.5–12 min as a result. Not worth
  fixing mid-run; `nvidia-smi -pl 250` or an undervolt would help future runs.
- **BERT padding waste:** 45% of transformer compute goes to `<PAD>` tokens
  (mean 141 real tokens of a 256 window). Naive dynamic padding does **not**
  help — 21.8% of documents hit the 256 cap, so a random batch of 32 contains
  a full-length document 99.97% of the time. Only length-grouped batching would
  recover it, at the cost of non-i.i.d. batches.
- Moved off Colab free tier — its ~12–13GB RAM ceiling was crashing on this
  dataset; 32GB locally clears that.
- Framework: **recommended but not firmly locked** — PyTorch, installed
  native on Windows with CUDA support via pip (no WSL2 needed). Reasoning:
  TensorFlow dropped native-Windows GPU support after v2.10 — anything newer
  needs WSL2 to see the GPU at all. The assignment's model names
  (`SimpleRNN`, `GRU`, `LSTM`, `Bidirectional`) read like literal Keras class
  names, so if exact naming match matters more than setup simplicity, Keras
  + WSL2 is the alternative — ask before assuming either way if it isn't
  obvious from what's already in the repo.
- Disk: 230GB free on C: as of 2026-08-29. Not a constraint — the earlier
  "disk is tight" note is stale.

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
4. Model training (3.7) & Hyperparameter tuning (3.8) — **complete, 30 runs**:
   - **3 Classical ML**: Logistic Regression ($C \in \{0.1, 1.0, 5.0\}$, best
     $C{=}1.0$), Naive Bayes ($\alpha \in \{0.01, 0.1, 1.0\}$, best $0.01$),
     Random Forest ($n{=}50$, $\text{depth} \in \{20, 30, 50\}$, best $50$).
     **`max_iter` raised 200 → 1000**: at 200 all three LR configs silently hit
     the cap and reported unconverged metrics. They now converge in 84 / 144 /
     277 iterations, and the tuning table carries a `Converged` column proving it.
   - **6 Recurrent Neural Networks (PyTorch)**: modular `RecurrentClassifier`
     with `pack_padded_sequence`, Word2Vec-initialised embeddings, Adam,
     grad-norm clipping at 1.0, 4 epochs. 3 configs each varying hidden dim
     $\{64, 128\}$, learning rate $\{10^{-3}, 5{\times}10^{-4}\}$ and dropout
     $\{0.3, 0.4\}$.
   - **1 Transformer**: BERT Base fine-tuned with AdamW + bfloat16 autocast,
     batch 32, 2 epochs, lr $\in \{2, 3, 5\} \times 10^{-5}$. **Best is the
     lowest lr** ($2{\times}10^{-5}$, 0.7647); the two higher rates are
     indistinguishable (0.7480 / 0.7499). Class weighting is applied to BERT
     too, so the objective is consistent across all paradigms.
   - **Epoch selection is on validation, never test.** Best-epoch weights are
     restored before the single test scoring pass.
5. Evaluation (3.9) — **complete**:
   - Master test table across all 10 models: Accuracy, Macro F1, Weighted F1,
     and **Train Time and Inference Time as separate columns**. The earlier
     single "Inference Time" column actually timed fit+predict, which hid the
     real efficiency story: LR classifies all 303,213 test documents in **0.2 s**
     versus BERT's 267 s, a 1,336× gap for 2.8 pp of Macro F1.
   - Normalized $9 \times 9$ confusion matrices for all three paradigms.
   - Full per-class classification reports across all 9 categories.
   - Best and worst model explicitly identified (BERT Base / SimpleRNN).
   - Error analysis grounded in the word clouds: the dominant corridor is
     **Money transfer → Bank account (14–17%)**, and *Payday loan* is hardest
     everywhere (recall 0.64–0.67) as the rarest class at 1.13%.

**Novelty angle — chosen, executed, and measured.** The angle is a controlled
imbalance-handling comparison: no mitigation vs. class weighting vs. SMOTE vs.
focal loss, across 5 architectures, 13 runs, all on the identical 200k budget
and identical full test split so only the strategy varies.

Minority-4 F1 (mean F1 over the four rarest classes):

| Model | None | Class Wt. | SMOTE | Focal | Best |
|---|---|---|---|---|---|
| Logistic Regression | **0.6829** | 0.6347 | 0.6353 | — | None |
| Naive Bayes | **0.6304** | n/a | 0.5853 | — | None |
| Random Forest | 0.3389 | 0.5820 | **0.6198** | — | SMOTE |
| Bidirectional LSTM | **0.6718** | 0.5931 | — | 0.6010 | None |
| BERT Base | — | **0.6789** | — | 0.6736 | Class Wt. |

**The original hypothesis was rejected by the data.** We expected focal loss to
win for deep models and SMOTE for linear ones. Neither holds: focal loses to
class weighting on BERT and to nothing-at-all on Bi-LSTM; SMOTE loses on both
linear models and wins only on the non-linear Random Forest. Because "None" wins
for both a classical and a deep model, the winning strategies overlap across
paradigms and the architecture-dependence claim **must be rejected**.

The reportable finding is the negative one: **imbalance mitigation compensates
for an architecture that cannot absorb skew; it does not improve one that can.**
Random Forest with SMOTE (0.6198, 839 s) still loses to plain Logistic
Regression with no mitigation at all (0.6829, 28 s).

Naive Bayes has no class-weight arm because `MultinomialNB` exposes no
`class_weight` parameter — stated, not silently skipped.

## Ensemble — done, bonus criterion satisfied (2026-08-30)

Soft voting, validation-F1 weighted, over **BERT Base + Random Forest + Naive
Bayes**. Selected from **1,506 candidates on validation**, then scored on test
exactly once.

| System | Test Macro F1 | Accuracy |
|---|---|---|
| BERT Base (best single) | 0.7647 | 0.8562 |
| **Ensemble** | **0.7792** | **0.8651** |
| Difference | **+1.45 pp** | +0.89 pp |

Three things make this worth defending at viva:

1. **It transferred.** Validation 0.7795 → test 0.7792, a gap of 0.0003 after a
   1,506-candidate search. Selecting on test instead would have been selection
   bias; an earlier probe that did exactly that found only +0.56 pp, because it
   could only hard-vote saved labels rather than soft-vote probabilities.
2. **The members are not the top three models.** Random Forest ranks 8th and
   Naive Bayes 5th, but they disagree with BERT on **14.0%** and **13.7%** of
   documents — more than any recurrent model (10–12%). Decorrelation beats
   individual strength in a vote.
3. **It is effectively free.** Both added members are the cheapest in the study,
   so the ensemble costs **1.0 s more than BERT alone** over 303,213 documents
   (+0.4%) for +1.45 pp. Unlike the accuracy/latency trade elsewhere in the
   report, there is no trade-off to weigh.

Gain concentrates on the rare classes: **+2.35 pp** mean over the four smallest
versus +0.57 pp over the four largest, and **every one of the nine classes
improves** — *Payday loan*, the rarest at 1.13%, gains the most (+3.34 pp).

Produced by `pipeline/ensemble.py`; the report subsection is generated from
`ensemble_results.json` by `pipeline/gen_tex.py` into
`report/ensemble_section.tex`, which `acl_report.tex` `\input`s. **Do not hand-edit
that .tex** — regenerate it.

**Caveat to disclose:** the validation split has now been used three times over
(hyperparameter tuning, epoch selection, ensemble selection). With 303,213
validation documents the overfitting risk is small, and the clean val→test
transfer is evidence it did not happen, but the reuse should be stated.

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

## Repo layout — how results are actually produced

The notebook is the deliverable, but the experiments are driven by a staged,
cached pipeline so that a failure costs seconds to retry instead of re-deriving
everything. **No metric anywhere is hardcoded.**

```
pipeline/
  common.py         splits, TF-IDF, Word2Vec, sequences; all disk-cached
  recurrent.py      RecurrentClassifier (with pack_padded_sequence) + train loop
  stages.py         classical / recurrent / BERT tuning stages
  evaluate.py       10-model held-out test evaluation
  novelty.py        13-run imbalance-strategy comparison
  discussion.py     generates Sections 13.2 and 14.1 markdown FROM the results
  inject.py         writes sources + real outputs into the notebook
  nb_cells.py       corrected cell sources
  export_figs.py    exports notebook figures to report/figures/
  show.py           pretty-prints whatever results exist in _cache/
  run_all.py        sequential driver for all 6 stages
_cache/             ~3.7 GB of cached artefacts + results JSON + run_all.log
report/             acl_report.tex, custom.bib, figures/, README.md
```

Run everything with `cd pipeline && python -u run_all.py`. Stages cache-check
and skip completed work, so re-running after a fix resumes where it stopped.
Build the notebook with `python inject.py` (dry-run to a scratch path first).

**Two invariants worth preserving:**

1. `inject.py` always rebuilds from
   `CFPB_Complaint_Classification_v2.BACKUP-2026-08-30-1130.ipynb`, not from its
   own previous output. Its cell indices refer to the original layout, and its
   own insertions (the protocol note, the word clouds) shift them — building
   from a previous output would send the discussion markdown to the wrong cells
   and overwrite a code cell. Keep that backup file.
2. `discussion.py` **derives** its claims. The architecture-dependence sentence
   only appears if the deep-model and classical-model winner sets are genuinely
   disjoint; otherwise it writes an explicit retraction. This is deliberate —
   it is what stopped the notebook re-asserting a finding the data contradicts.

## Working conventions

- **Versioning: decided — no git, no `_vN.ipynb` filename increments.**
  User's call (2026-08-29): Claude Code edits the real file in place each
  session, and that supersedes the old filename-increment convention from
  the Claude.ai handoff. `PROJECT_LOG.md` is the record of what changed and
  why — that's the substitute for git history here, not a replacement for
  it. This also means the "clean GitHub repo" bonus criterion is off the
  table for this project; don't suggest it. Do not `git init` this folder
  unless the user explicitly asks again.
