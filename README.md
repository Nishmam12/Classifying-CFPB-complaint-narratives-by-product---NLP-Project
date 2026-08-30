# CFPB Financial Complaint Multi-Class NLP Classification

[![Vercel Deployment](https://img.shields.io/badge/Deployed-Vercel-black?style=flat&logo=vercel)](https://vercel.com)
[![Cloudflare Pages](https://img.shields.io/badge/Deployed-Cloudflare%20Pages-F38020?style=flat&logo=cloudflare)](https://pages.cloudflare.com)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org)
[![HuggingFace Transformers](https://img.shields.io/badge/Transformers-BERT%20Base-yellow?style=flat&logo=huggingface)](https://huggingface.co)
[![BRAC University](https://img.shields.io/badge/BRAC%20University-CSE440%20NLP%20II-navy)](https://www.bracu.ac.bd)

> **CSE440 (Natural Language Processing II) Lab Project**  
> **Instructor:** Labib Hasan Khan , Ariyan Hossain
> **Topic:** Multi-Class Consumer Financial Protection Bureau (CFPB) Complaint Narrative Classification  
> **Live Interactive Web Demo:** [Click Here](https://web-blush-nine-97.vercel.app/)

---

## 👥 Team Members & Contributions

| Member Name | Student ID | Primary Role / Contribution |
|:---|:---|:---|
| **Nabil Ishmam** | *24241245* | BERT Base Fine-Tuning, Transformer Pipeline & Web Deployment |
| **Afnan Mojumder** | *23301519* | Recurrent Models (Bi-LSTM, GRU, SimpleRNN) & Word2Vec Embeddings |
| **Shoumodip Paul** | *23201447* | Classical ML Models (LR, NB, RF), TF-IDF & Imbalance Mitigations |
| **Quazi Unjurn Daniel** | *23201133* | Soft-Voting Ensemble, ACL LaTeX Report & Error Analysis |

---

## 🌟 Key Highlights & Bonus Marks Implementation

This project implements all avenues for the **+2 Bonus Marks**:

1. **🚀 Deployed Interactive Web Demo (Vercel & Cloudflare):**
   * Real-time text complaint classifier with probability gauges across all 9 classes.
   * Model switcher across **Ensemble**, **BERT Base**, **Bi-LSTM**, **GRU**, and **Logistic Regression**. The browser demo scores text with a lightweight lexical approximation of each model, not the trained weights, so its live probabilities illustrate behaviour rather than reproduce the benchmark below.
   * Interactive 10-model leaderboard and ablation study explorer.

2. **⚡ Ensemble Model Innovation (+1.45 pp Boost):**
   * Soft-voting ensemble over **BERT Base + Random Forest + Naive Bayes**, weighted by validation macro F1. Members were chosen from **1,506 candidate combinations scored on the validation split**, and the winning configuration was scored on test exactly once (validation 0.7795 → test 0.7792, so the gain transfers).
   * The two added members rank 8th and 5th individually; they earn their place by *disagreeing* with BERT on 14.0% and 13.7% of test documents, where the recurrent models mostly echo it.
   * Achieves **0.7792 Macro F1** and **86.51% Accuracy**, setting a new top benchmark for **+1.0 s** of inference over BERT alone (a 0.4% increase across 303,213 documents).

3. **🔬 Controlled Ablation & Imbalance Mitigation Study (13 Runs):**
   * Evaluated **Natural Prior vs Class Weighting vs SMOTE Synthesis vs Focal Loss ($\gamma=2$)** across 5 distinct model paradigms.
   * Discovered that class weighting incurs a $-7.87\text{ pp}$ penalty on deep neural models by skewing false positives, but rescues tree-based models (Random Forest $+24.31\text{ pp}$).

4. **📊 Comprehensive 10-Model Empirical Benchmarking:**
   * Trained on a fair budget (200,000 docs) and evaluated on 303,213 held-out test documents.
   * Audited 60-cell Jupyter notebook with zero unexecuted cells and zero hardcoded metrics.

5. **📝 ACL-Format LaTeX Research Paper:**
   * Complete 8-page academic paper with 26 cited references in `report/acl_report.tex`.

6. **🧠 Fourth Paradigm — Causal-Decoder SLM (Qwen2.5-1.5B + LoRA):**
   * Beyond the ten benchmarked systems, a 1.54B causal decoder was adapted with LoRA ($r=16$, $lpha=32$), training ~1.2% of its weights.
   * Reported in the report **appendix** rather than the main table: that run was scored on a 5,000-document subset from a separately prepared split. Resampling our own stored predictions shows a class-balanced subset of that size inflates macro F1 by ~6 pp, so it is not interchangeable with a full-split number. `pipeline/qwen_lora.py` reproduces the arm against the shared 200k / 303,213 protocol.

---

## 🏆 10-Model Benchmark Leaderboard

| Rank | Model Architecture | Paradigm | Test Macro F1 | Test Accuracy | Test Weighted F1 | Train Time | Infer Time |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Ensemble (BERT + RF + NB)** | **Ensemble** | **0.7792** | **86.51%** | **86.84%** | 1709.2 s | 268.3 s |
| 🥈 | **BERT Base** | Transformer | **0.7647** | 85.62% | 86.13% | 1678.1 s | 267.3 s |
| 🥉 | **Logistic Regression** | Classical ML | **0.7367** | 83.90% | 84.68% | 34.5 s | 0.2 s |
| 4 | **GRU** | Recurrent Net | **0.7274** | 83.26% | 84.20% | 77.2 s | 5.8 s |
| 5 | **Bidirectional GRU** | Recurrent Net | **0.7234** | 82.67% | 83.76% | 130.6 s | 8.4 s |
| 6 | **Naive Bayes** | Classical ML | **0.7211** | 83.29% | 83.57% | 0.1 s | 0.2 s |
| 7 | **Bidirectional LSTM** | Recurrent Net | **0.7173** | 81.75% | 82.74% | 139.1 s | 9.2 s |
| 8 | **LSTM** | Recurrent Net | **0.7110** | 81.80% | 82.76% | 78.2 s | 5.8 s |
| 9 | **Random Forest** | Classical ML | **0.6920** | 81.21% | 81.97% | 31.0 s | 0.8 s |
| 10 | **Bidirectional SimpleRNN** | Recurrent Net | **0.6600** | 78.24% | 79.44% | 124.2 s | 8.9 s |
| 11 | **SimpleRNN** | Recurrent Net | **0.5685** | 67.22% | 69.43% | 80.5 s | 5.9 s |

---

## 📁 Repository Structure

```
├── CFPB_Complaint_Classification_v2.ipynb  # Primary deliverable notebook (60 cells)
├── README.md                               # Project documentation & benchmark summary
├── .gitignore                              # Git exclusions for large datasets/weights
│
├── web/                                    # Production Web Application (Vercel/Cloudflare)
│   ├── index.html                          # Responsive SPA interface
│   ├── package.json                        # Node build scripts
│   ├── vite.config.js                      # Vite bundler configuration
│   ├── vercel.json                         # Vercel deployment routing
│   ├── src/
│   │   ├── app.js                          # UI reactive controller
│   │   ├── classifier.js                   # Inference engine & ensemble logic
│   │   ├── data.js                         # Benchmark data & sample narratives
│   │   └── style.css                       # Glassmorphic dark/light theme
│   └── public/figures/                     # High-res confusion matrices & charts
│
├── pipeline/                               # Cached, modular Python experiment stages
│   ├── common.py                           # Stratified 70/15/15 splits & TF-IDF
│   ├── recurrent.py                        # PyTorch RecurrentClassifier architecture
│   ├── stages.py                           # Tuning sweeps for all models
│   ├── evaluate.py                         # 10-model held-out test evaluation
│   ├── novelty.py                          # 13-run imbalance comparison
│   ├── ensemble.py                         # Soft-voting ensemble search & scoring
│   ├── qwen_lora.py                        # Qwen2.5-1.5B LoRA causal-decoder arm
│   └── export_figs.py                      # Figure exporter for LaTeX report
│
├── report/                                 # ACL-Format LaTeX Report
│   ├── acl_report.tex                      # Paper source code
│   ├── custom.bib                          # 26 cited bibtex references
│   ├── qwen_appendix.tex                   # Appendix: causal-decoder (Qwen) arm
│   └── figures/                            # 10 publication-quality PNG figures
│
└── best_bert_model/                        # Fine-tuned BERT Base configuration
    ├── config.json
    └── tokenizer.json
```

---

## 🚀 How to Run & Deploy

### 1. Running the Web Demo Locally
```bash
cd web
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

### 2. Deploying to Vercel
```bash
cd web
npx vercel
```
Or connect your GitHub repository on [vercel.com](https://vercel.com) and set the Root Directory to `web/`.

### 3. Deploying to Cloudflare Pages
```bash
cd web
npm run build
npx wrangler pages deploy dist --project-name=cfpb-nlp-demo
```
Or import your GitHub repo into Cloudflare Pages dashboard with build directory `dist`.

---

## 📜 Citation & Academic Context
Developed for CSE440 (NLP II) at BRAC University under the supervision of Dr. Farig Sadeque.
Dataset provided by the Consumer Financial Protection Bureau (CFPB) / Kaggle.
