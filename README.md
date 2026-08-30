# Classifying Consumer Financial Complaints by Product Category: A Comparative Study of Text Representations, Neural Architectures, and Causal Decoder SLMs

**Course:** CSE440 — Natural Language Processing II  
**Institution:** BRAC University  

### Authors (Group 1)
- **Nabil Ishmam** (24241245) — `nabil.ishmam@g.bracu.ac.bd`
- **Quazi Unjurn Daniel** (23201133) — `quazi.unjurn.daniel@g.bracu.ac.bd`
- **Shoumodip Paul** (23201447) — `shoumodip.paul@g.bracu.ac.bd`
- **Afnan Mojumder** (23301519) — `afnan.mojumder@g.bracu.ac.bd`

---

## 📌 Project Overview
This repository contains the complete empirical codebase, research paper, and benchmark results for multi-class classification of **2.02 million Consumer Financial Protection Bureau (CFPB) complaint narratives** into nine consolidated product categories under severe class imbalance (52.9:1 skew).

We benchmark **11 architectures** across 4 modeling paradigms under an identical 200,000-document training budget and a held-out 303,213-document test split with strict zero-leakage protocol:
1. **Classical Machine Learning:** Multinomial Naive Bayes, Logistic Regression, Random Forest (over 25,000 sublinear TF-IDF features).
2. **Recurrent Neural Networks:** SimpleRNN, GRU, LSTM, Bidirectional SimpleRNN, Bidirectional GRU, Bidirectional LSTM (over 100D domain Word2Vec CBOW embeddings).
3. **Bidirectional Transformers:** BERT Base (`bert-base-uncased`, 110M parameters).
4. **Causal Decoder Small Language Models (SLMs):** Qwen2.5 (1.5B) adapted via Parameter-Efficient Fine-Tuning (LoRA, $r=16$, updating 1.18% of weights).

---

## 📊 Master Benchmark Results (Held-Out Test Set: 303,213 Samples)

| Rank | Model Architecture | Paradigm | Parameters | Test Accuracy | Test Macro F1 | Test Weighted F1 | Inference Latency |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | **BERT Base** (`bert-base-uncased`) | Bidirectional Encoder | 110M | **86.51%** | **0.7738** | **0.8691** | 4.82 ms |
| 2 | **Qwen2.5 (1.5B LoRA)** ⭐ | **Causal Decoder SLM** | **1.54B (18.5M train)** | **82.84%** | **0.7427** | **0.8345** | **62.16 ms** |
| 3 | **Logistic Regression** | Classical ML | 25K TF-IDF | **83.90%** | **0.7367** | **0.8468** | **0.10 ms** |
| 4 | **Bidirectional GRU** | Recurrent Neural Net | 3.2M | **82.84%** | **0.7248** | **0.8373** | 5.30 ms |
| 5 | **Naive Bayes** | Classical ML | 25K TF-IDF | **83.29%** | **0.7211** | **0.8357** | **0.10 ms** |
| 6 | **LSTM** | Recurrent Neural Net | 3.2M | **82.78%** | **0.7186** | **0.8370** | 3.70 ms |
| 7 | **Bidirectional LSTM** | Recurrent Neural Net | 3.2M | **82.75%** | **0.7169** | **0.8375** | 5.50 ms |
| 8 | **GRU** | Recurrent Neural Net | 3.2M | **81.51%** | **0.7163** | **0.8262** | 3.80 ms |
| 9 | **Random Forest** | Classical ML | 50 Trees | **81.21%** | **0.6920** | **0.8197** | 0.50 ms |
| 10 | **Bidirectional SimpleRNN** | Recurrent Neural Net | 3.2M | **77.75%** | **0.6568** | **0.7951** | 5.40 ms |
| 11 | **SimpleRNN** | Recurrent Neural Net | 3.2M | **73.43%** | **0.5837** | **0.7524** | 4.00 ms |

---

## 📁 Repository Structure

```
├── CFPB_Complaint_Classification_v2.ipynb    # Complete end-to-end executable notebook (36/36 cells executed)
├── 1_24241245_23201133_23201447_23301519.tex # Publication-ready LaTeX source file (Overleaf compatible)
├── 1_24241245_23201133_23201447_23301519.pdf # Compiled academic report deliverable
├── report.tex                               # LaTeX report alias
├── UPDATED_FINAL_REPORT.md                  # Comprehensive markdown report
├── figures/                                 # 11 High-resolution 300 DPI report figures
│   ├── fig1_raw_products.png
│   ├── fig2_consolidated_9classes.png
│   ├── fig3_word_count_distribution.png
│   ├── fig4_word_clouds.png
│   ├── fig5_preprocessing_lengths.png
│   ├── fig6_split_stratification.png
│   ├── fig7_validation_tuning_ranking.png
│   ├── fig8_master_test_benchmark.png
│   ├── fig9_confusion_matrices_grid.png
│   ├── fig10_imbalance_novelty.png
│   └── fig11_latency_pareto_frontier.png
├── pipeline/                                # Modular training, evaluation, and plotting scripts
├── results/                                 # Final benchmark metrics, evaluation CSVs, and plots
└── README.md
```

---

## 🚀 How to Run

1. **Install Dependencies:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   pip install transformers peft accelerate scikit-learn pandas numpy matplotlib seaborn wordcloud reportlab pymupdf
   ```

2. **Execute Jupyter Notebook:**
   Open `CFPB_Complaint_Classification_v2.ipynb` in VS Code or JupyterLab and run all cells sequentially.
