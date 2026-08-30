# Classifying Consumer Financial Complaints by Product Category: A Comparative Study of Text Representations, Neural Architectures, and Causal Decoder Small Language Models (Qwen2.5)

**Authors:**
- **Nabil Ishmam** (24241245) — `nabil.ishmam@g.bracu.ac.bd`
- **Quazi Unjurn Daniel** (23201133) — `quazi.unjurn.daniel@g.bracu.ac.bd`
- **Shoumodip Paul** (23201447) — `shoumodip.paul@g.bracu.ac.bd`
- **Afnan Mojumder** (23301519) — `afnan.mojumder@g.bracu.ac.bd`

**Institution:** Department of Computer Science and Engineering, BRAC University  
**Course:** CSE440 — Natural Language Processing II  
**Deliverable File:** [`1_24241245_23201133_23201447_23301519.pdf`](1_24241245_23201133_23201447_23301519.pdf)  

---

## Abstract

We classify 2.02 million U.S. Consumer Financial Protection Bureau (CFPB) complaint narratives into nine consolidated product categories, benchmarking eleven architectures spanning four distinct NLP paradigms: classical bag-of-words over sublinear TF-IDF, six recurrent neural networks over domain Word2Vec CBOW embeddings, a fine-tuned BERT Base bidirectional transformer, and a modern 1.54B parameter Causal Decoder Small Language Model (**Qwen2.5-1.5B**) adapted via Parameter-Efficient Fine-Tuning (LoRA, $r=16$). All models are evaluated on an identical 303,213-document held-out test split under a rigorous zero-leakage protocol. 

**BERT Base** attains the overall peak performance (**Test Macro F1 = 0.7738, Accuracy = 86.51%**), while **Qwen2.5 (1.5B LoRA)** achieves **0.7427 Macro F1 and 82.84% Accuracy** by training only 1.18% (18.48M) of its parameters. Qwen2.5 outperforms all six recurrent architectures (Bi-GRU 0.7248, Bi-LSTM 0.7169) by capturing multi-head self-attention dependencies, but incurs an inference latency penalty (62.16 ms vs 4.82 ms for BERT and 0.10 ms for Logistic Regression). Additionally, our novelty investigation into class imbalance (52.9:1 skew) shows that SMOTE oversampling dramatically rescues minority-class F1 for tree ensembles (+0.2809 gain on Random Forest), whereas linear and deep models perform best without synthetic oversampling.

---

## 1. Primary Benchmark Results

| Rank | Model Architecture | Paradigm | Parameter Count | Training Strategy | Test Accuracy | Test Macro F1 | Test Weighted F1 | Inference Latency |
|:---:|:---|:---|:---:|:---|:---:|:---:|:---:|:---:|
| 1 | **BERT Base** (`bert-base-uncased`) | Bidirectional Encoder | 110M | Full Fine-Tuning | **86.51%** | **0.7738** | **0.8691** | 4.82 ms |
| 2 | **Qwen2.5 (1.5B LoRA)** ⭐ | **Causal Decoder SLM** | **1.54B** | **PEFT / LoRA ($r=16$)** | **82.84%** | **0.7427** | **0.8345** | **62.16 ms** |
| 3 | **Logistic Regression** | Classical ML | 25K TF-IDF | Class-Weighted $C=1.0$ | **83.90%** | **0.7367** | **0.8468** | **0.10 ms** |
| 4 | **Bidirectional GRU** | Recurrent Neural Net | 3.2M | Domain Word2Vec CBOW | **82.84%** | **0.7248** | **0.8373** | 5.30 ms |
| 5 | **Naive Bayes** | Classical ML | 25K TF-IDF | Multinomial ($\alpha=0.01$) | **83.29%** | **0.7211** | **0.8357** | **0.10 ms** |
| 6 | **LSTM** | Recurrent Neural Net | 3.2M | Domain Word2Vec CBOW | **82.78%** | **0.7186** | **0.8370** | 3.70 ms |
| 7 | **Bidirectional LSTM** | Recurrent Neural Net | 3.2M | Domain Word2Vec CBOW | **82.75%** | **0.7169** | **0.8375** | 5.50 ms |
| 8 | **GRU** | Recurrent Neural Net | 3.2M | Domain Word2Vec CBOW | **81.51%** | **0.7163** | **0.8262** | 3.80 ms |
| 9 | **Random Forest** | Classical ML | 50 Trees | Depth 50, Balanced | **81.21%** | **0.6920** | **0.8197** | 0.50 ms |
| 10 | **Bidirectional SimpleRNN** | Recurrent Neural Net | 3.2M | Domain Word2Vec CBOW | **77.75%** | **0.6568** | **0.7951** | 5.40 ms |
| 11 | **SimpleRNN** | Recurrent Neural Net | 3.2M | Domain Word2Vec CBOW | **73.43%** | **0.5837** | **0.7524** | 4.00 ms |

---

## 2. Key Empirical Findings & Discussion

### 2.1 Bidirectional Encoders (BERT) vs. Causal Decoders (Qwen2.5 LoRA)
1. **Unconstrained Bidirectional Context is Superior for Discriminative Classification**:
   - BERT Base achieves higher Macro F1 (**0.7738** vs. **0.7427**) and Accuracy (**86.51%** vs. **82.84%**).
   - Because BERT performs bidirectional attention across all sequence positions, every hidden token representation incorporates future and past narrative context simultaneously. In contrast, Qwen2.5 is pre-trained with lower-triangular causal masking, meaning token $t$ cannot attend to $t+1$.
2. **Qwen2.5 Beats All Recurrent Networks**:
   - Qwen2.5 (1.5B LoRA) surpasses the strongest recurrent model (Bi-GRU: 0.7248) and Bi-LSTM (0.7169) by over **+1.8 to +2.5 percentage points in Macro F1**.
   - The multi-head self-attention mechanism over 28 transformer decoder layers successfully prevents the catastrophic forgetting and sequential bottleneck inherent to recurrent architectures.
3. **Parameter Efficiency via LoRA**:
   - LoRA ($r=16, \alpha=32$) trained only **18.48 million parameters (1.18%)** of the 1.54B parameter Qwen2.5 backbone, enabling fine-tuning on a single consumer GPU without updating full weight tensors.

### 2.2 Production Pareto Efficiency Frontier
- **Peak Accuracy**: **BERT Base** (86.51% Accuracy, 0.7738 Macro F1, 4.82 ms latency).
- **Maximum Throughput**: **Logistic Regression (TF-IDF)** (83.90% Accuracy, 0.7367 Macro F1, **0.10 ms latency** — $600\times$ faster than Qwen2.5 and $48\times$ faster than BERT).
- **Highest Expressivity SLM**: **Qwen2.5 (1.5B LoRA)** (82.84% Accuracy, 0.7427 Macro F1, 62.16 ms latency).

---

## 3. Associated Report Figures

1. **Figure 1 — Master Comparative Bar Chart**: `results/plots/master_macro_f1_vs_accuracy_qwen.png`
2. **Figure 2 — Latency vs. Macro F1 Pareto Efficiency Frontier**: `results/plots/latency_vs_macro_f1_frontier_qwen.png`
3. **Figure 3 — Normalized Confusion Matrix for Qwen2.5 LoRA**: `results/plots/cm_qwen2.5_1.5b_lora.png`
