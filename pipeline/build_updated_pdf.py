import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)

sys.stdout.reconfigure(encoding='utf-8')

pdf_path = r"E:\BRACU\26-SUMMER\440\CSE440 Lab Project\1_24241245_23201133_23201447_23301519.pdf"
backup_path = r"E:\BRACU\26-SUMMER\440\CSE440 Lab Project\1_24241245_23201133_23201447_23301519_ORIGINAL.pdf"

if not os.path.exists(backup_path) and os.path.exists(pdf_path):
    import shutil
    shutil.copy2(pdf_path, backup_path)
    print(f"Backed up original PDF to {backup_path}")

doc = SimpleDocTemplate(
    pdf_path,
    pagesize=letter,
    leftMargin=40,
    rightMargin=40,
    topMargin=40,
    bottomMargin=40
)

styles = getSampleStyleSheet()

# Custom styles
title_style = ParagraphStyle(
    'DocTitle',
    parent=styles['Heading1'],
    fontName='Helvetica-Bold',
    fontSize=16,
    leading=20,
    alignment=1, # Center
    textColor=colors.HexColor('#111827')
)

author_style = ParagraphStyle(
    'DocAuthors',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=8.5,
    leading=11,
    alignment=1,
    textColor=colors.HexColor('#374151')
)

abstract_heading = ParagraphStyle(
    'AbstractHeading',
    parent=styles['Heading2'],
    fontName='Helvetica-Bold',
    fontSize=10.5,
    leading=13,
    alignment=1,
    textColor=colors.HexColor('#1e3a8a')
)

abstract_body = ParagraphStyle(
    'AbstractBody',
    parent=styles['Normal'],
    fontName='Helvetica-Oblique',
    fontSize=8.5,
    leading=11.5,
    alignment=4, # Justify
    textColor=colors.HexColor('#1f2937')
)

h1_style = ParagraphStyle(
    'SectionHeading',
    parent=styles['Heading1'],
    fontName='Helvetica-Bold',
    fontSize=11.5,
    leading=15,
    spaceBefore=10,
    spaceAfter=4,
    textColor=colors.HexColor('#1e3a8a')
)

h2_style = ParagraphStyle(
    'SubSectionHeading',
    parent=styles['Heading2'],
    fontName='Helvetica-Bold',
    fontSize=9.5,
    leading=13,
    spaceBefore=6,
    spaceAfter=2,
    textColor=colors.HexColor('#1f2937')
)

body_style = ParagraphStyle(
    'BodyDark',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=8.5,
    leading=11.5,
    alignment=4, # Justify
    spaceBefore=2,
    spaceAfter=4,
    textColor=colors.HexColor('#111827')
)

caption_style = ParagraphStyle(
    'Caption',
    parent=styles['Normal'],
    fontName='Helvetica-Oblique',
    fontSize=7.5,
    leading=9.5,
    alignment=1,
    textColor=colors.HexColor('#4b5563')
)

table_cell_style = ParagraphStyle(
    'TableCell',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=7.5,
    leading=9.5,
    textColor=colors.HexColor('#111827')
)

table_header_style = ParagraphStyle(
    'TableHeader',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=7.5,
    leading=9.5,
    textColor=colors.white
)

story = []

# Title
story.append(Paragraph("Classifying Consumer Financial Complaints by Product Category:<br/>A Comparative Study of Text Representations, Neural Architectures, and Causal Decoder Small Language Models (Qwen2.5)", title_style))
story.append(Spacer(1, 8))

# Authors Table
authors_data = [
    [
        Paragraph("<b>Nabil Ishmam</b><br/>24241245<br/>BRAC University<br/>nabil.ishmam@g.bracu.ac.bd", author_style),
        Paragraph("<b>Quazi Unjurn Daniel</b><br/>23201133<br/>BRAC University<br/>quazi.unjurn.daniel@g.bracu.ac.bd", author_style),
        Paragraph("<b>Shoumodip Paul</b><br/>23201447<br/>BRAC University<br/>shoumodip.paul@g.bracu.ac.bd", author_style),
        Paragraph("<b>Afnan Mojumder</b><br/>23301519<br/>BRAC University<br/>afnan.mojumder@g.bracu.ac.bd", author_style)
    ]
]
t_auth = Table(authors_data, colWidths=[130, 130, 130, 130])
t_auth.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'TOP')]))
story.append(t_auth)
story.append(Spacer(1, 10))

# Abstract Box
story.append(Paragraph("<b>Abstract</b>", abstract_heading))
story.append(Spacer(1, 3))
abstract_text = (
    "We classify 2.02 million U.S. Consumer Financial Protection Bureau (CFPB) complaint narratives into nine consolidated "
    "product categories, benchmarking eleven architectures spanning four distinct NLP paradigms: classical bag-of-words over sublinear "
    "TF-IDF, six recurrent neural networks over domain Word2Vec CBOW embeddings, a fine-tuned BERT Base bidirectional transformer, "
    "and a modern 1.54B parameter Causal Decoder Small Language Model (<b>Qwen2.5-1.5B</b>) adapted via Parameter-Efficient Fine-Tuning "
    "(LoRA, r=16). All models are evaluated on an identical 303,213-document held-out test split under a rigorous zero-leakage protocol. "
    "<b>BERT Base</b> attains the overall peak performance (<b>Test Macro F1 = 0.7738, Accuracy = 86.51%</b>), while <b>Qwen2.5 (1.5B LoRA)</b> "
    "achieves <b>0.7427 Macro F1 and 82.84% Accuracy</b> by training only 1.18% (18.48M) of its parameters. Qwen2.5 outperforms all six recurrent "
    "architectures (Bi-GRU 0.7248, Bi-LSTM 0.7169) by capturing multi-head self-attention dependencies, but incurs an inference latency "
    "penalty (62.16 ms vs 4.82 ms for BERT and 0.10 ms for Logistic Regression). Additionally, our novelty investigation into class imbalance (52.9:1 skew) "
    "shows that SMOTE oversampling dramatically rescues minority-class F1 for tree ensembles (+0.2809 gain on Random Forest), whereas linear "
    "and deep models perform best without synthetic oversampling."
)
story.append(Paragraph(abstract_text, abstract_body))
story.append(Spacer(1, 10))
story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#d1d5db'), spaceBefore=4, spaceAfter=8))

# 1. Introduction
story.append(Paragraph("1. Introduction", h1_style))
intro_p1 = (
    "The Consumer Financial Protection Bureau (CFPB) maintains the largest public repository of consumer financial complaints in the United States, "
    "logging over 2.02 million narratives since 2011. Efficient multi-class routing of these narratives is essential for regulatory triage, compliance auditing, "
    "and enforcement prioritization. However, real-world complaint streams exhibit severe linguistic challenges: extreme label skew (59.6% credit reporting vs. "
    "1.1% payday loans), verbose legal jargon, domain acronyms, and missing context. In this empirical study, we investigate eleven architectures spanning "
    "Classical Machine Learning, Gated Recurrent Neural Networks, Bidirectional Pretrained Encoders, and modern Causal Decoder SLMs adapted via LoRA."
)
story.append(Paragraph(intro_p1, body_style))

# 2. Related Work
story.append(Paragraph("2. Related Work", h1_style))
rel_work = (
    "<b>Classical Baselines & Term Weighting:</b> TF-IDF n-gram weighting with regularized Logistic Regression and Naive Bayes represents the foundational "
    "baseline for text classification (Joachims, 1998; McCallum and Nigam, 1998).<br/>"
    "<b>Recurrent Networks:</b> Gated recurrent units—LSTM (Hochreiter and Schmidhuber, 1997) and GRU (Cho et al., 2014)—overcome vanishing gradients in "
    "vanilla SimpleRNN (Elman, 1990) when paired with distributed embeddings (Mikolov et al., 2013).<br/>"
    "<b>Bidirectional Pretrained Transformers:</b> BERT (Devlin et al., 2019) revolutionized text classification by learning bidirectional self-attention "
    "representations across masked token contexts.<br/>"
    "<b>Causal Decoder SLMs & Parameter-Efficient Fine-Tuning (LoRA):</b> Recent breakthroughs in Large and Small Language Models (Qwen2.5; Yang et al., 2024) "
    "leverage autoregressive causal transformer decoders pre-trained on trillions of tokens. Low-Rank Adaptation (LoRA; Hu et al., 2021) freezes the base backbone "
    "and injects trainable rank-decomposition matrices into multi-head attention projections, enabling multi-billion parameter SLMs to adapt to specialized "
    "downstream classification tasks with minimal parameter updates and VRAM overhead."
)
story.append(Paragraph(rel_work, body_style))

# 3. Methodology & Architectures
story.append(Paragraph("3. Methodology & Experimental Framework", h1_style))
meth_p1 = (
    "<b>3.1 Dataset Partitioning:</b> The 2,021,420 valid complaints are consolidated into 9 mutually exclusive product categories and split strictly into "
    "<b>70% Train (1,414,994), 15% Validation (303,213), and 15% Test (303,213)</b> using stratified sampling on random seed 42.<br/>"
    "<b>3.2 Representations:</b> (i) 25,000 sublinear TF-IDF unigram+bigram features; (ii) 100-dimensional domain Word2Vec CBOW embeddings trained over 1.41M complaint texts; "
    "(iii) 256-token padded sequence indices for PyTorch recurrent networks; (iv) WordPiece tokenization for BERT; (v) BPE tokenization for Qwen2.5.<br/>"
    "<b>3.3 Qwen2.5 (1.5B) LoRA Formulation:</b> We formulate sequence classification with Qwen2.5-1.5B by attaching a linear classification head to the pooled "
    "last-token causal hidden state. Trainable low-rank adapters ($r=16, \\alpha=32, \\text{dropout}=0.05$) are injected across all attention projection layers "
    "(<code>q_proj, k_proj, v_proj, o_proj</code>) and MLP projection layers (<code>gate_proj, up_proj, down_proj</code>). Out of 1.56 billion total parameters, "
    "only <b>18,478,592 parameters (1.18%)</b> are updated, trained using AdamW ($lr=2\\times 10^{-4}$), class-weighted cross-entropy loss, and bfloat16 mixed precision."
)
story.append(Paragraph(meth_p1, body_style))

# 4. Results & Discussion
story.append(Paragraph("4. Results & Empirical Discussion", h1_style))
res_p1 = (
    "Table 1 summarizes the primary benchmark results on the 303,213 held-out test split across all eleven evaluated models. "
    "Figure 1 illustrates the comparative performance across Accuracy and Macro F1, while Figure 2 depicts the Pareto efficiency frontier "
    "relating accuracy gains to inference latency."
)
story.append(Paragraph(res_p1, body_style))

# Main Benchmark Table
table_data = [
    [Paragraph("<b>Model Architecture</b>", table_header_style),
     Paragraph("<b>Paradigm</b>", table_header_style),
     Paragraph("<b>Parameters</b>", table_header_style),
     Paragraph("<b>Test Acc (%)</b>", table_header_style),
     Paragraph("<b>Test Macro F1</b>", table_header_style),
     Paragraph("<b>Test W-F1</b>", table_header_style),
     Paragraph("<b>Latency (ms)</b>", table_header_style)],
    
    [Paragraph("<b>BERT Base</b>", table_cell_style), Paragraph("Bidirectional Encoder", table_cell_style), Paragraph("110M", table_cell_style), Paragraph("<b>86.51%</b>", table_cell_style), Paragraph("<b>0.7738</b>", table_cell_style), Paragraph("<b>0.8691</b>", table_cell_style), Paragraph("4.82 ms", table_cell_style)],
    [Paragraph("<b>Qwen2.5 (1.5B LoRA)</b> ⭐", table_cell_style), Paragraph("Causal Decoder SLM", table_cell_style), Paragraph("1.54B (18.5M train)", table_cell_style), Paragraph("<b>82.84%</b>", table_cell_style), Paragraph("<b>0.7427</b>", table_cell_style), Paragraph("<b>0.8345</b>", table_cell_style), Paragraph("62.16 ms", table_cell_style)],
    [Paragraph("Logistic Regression", table_cell_style), Paragraph("Classical ML (TF-IDF)", table_cell_style), Paragraph("25K Weights", table_cell_style), Paragraph("83.90%", table_cell_style), Paragraph("0.7367", table_cell_style), Paragraph("0.8468", table_cell_style), Paragraph("<b>0.10 ms</b>", table_cell_style)],
    [Paragraph("Bidirectional GRU", table_cell_style), Paragraph("Recurrent Neural Net", table_cell_style), Paragraph("3.2M", table_cell_style), Paragraph("82.84%", table_cell_style), Paragraph("0.7248", table_cell_style), Paragraph("0.8373", table_cell_style), Paragraph("5.30 ms", table_cell_style)],
    [Paragraph("Naive Bayes", table_cell_style), Paragraph("Classical ML (TF-IDF)", table_cell_style), Paragraph("25K Priors", table_cell_style), Paragraph("83.29%", table_cell_style), Paragraph("0.7211", table_cell_style), Paragraph("0.8357", table_cell_style), Paragraph("0.10 ms", table_cell_style)],
    [Paragraph("LSTM", table_cell_style), Paragraph("Recurrent Neural Net", table_cell_style), Paragraph("3.2M", table_cell_style), Paragraph("82.78%", table_cell_style), Paragraph("0.7186", table_cell_style), Paragraph("0.8370", table_cell_style), Paragraph("3.70 ms", table_cell_style)],
    [Paragraph("Bidirectional LSTM", table_cell_style), Paragraph("Recurrent Neural Net", table_cell_style), Paragraph("3.2M", table_cell_style), Paragraph("82.75%", table_cell_style), Paragraph("0.7169", table_cell_style), Paragraph("0.8375", table_cell_style), Paragraph("5.50 ms", table_cell_style)],
    [Paragraph("GRU", table_cell_style), Paragraph("Recurrent Neural Net", table_cell_style), Paragraph("3.2M", table_cell_style), Paragraph("81.51%", table_cell_style), Paragraph("0.7163", table_cell_style), Paragraph("0.8262", table_cell_style), Paragraph("3.80 ms", table_cell_style)],
    [Paragraph("Random Forest", table_cell_style), Paragraph("Classical ML (TF-IDF)", table_cell_style), Paragraph("50 Trees", table_cell_style), Paragraph("81.21%", table_cell_style), Paragraph("0.6920", table_cell_style), Paragraph("0.8197", table_cell_style), Paragraph("0.50 ms", table_cell_style)],
    [Paragraph("Bidirectional SimpleRNN", table_cell_style), Paragraph("Recurrent Neural Net", table_cell_style), Paragraph("3.2M", table_cell_style), Paragraph("77.75%", table_cell_style), Paragraph("0.6568", table_cell_style), Paragraph("0.7951", table_cell_style), Paragraph("5.40 ms", table_cell_style)],
    [Paragraph("SimpleRNN", table_cell_style), Paragraph("Recurrent Neural Net", table_cell_style), Paragraph("3.2M", table_cell_style), Paragraph("73.43%", table_cell_style), Paragraph("0.5837", table_cell_style), Paragraph("0.7524", table_cell_style), Paragraph("4.00 ms", table_cell_style)]
]

t_main = Table(table_data, colWidths=[115, 85, 75, 55, 65, 65, 65])
t_main.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e3a8a')),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#d1d5db')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f9fafb')])
]))
story.append(t_main)
story.append(Spacer(1, 4))
story.append(Paragraph("<b>Table 1:</b> Master performance benchmark on the 303,213-sample held-out test split.", caption_style))
story.append(Spacer(1, 10))

# Insert Figures
img_bar_path = r"E:\BRACU\26-SUMMER\440\CSE440 Lab Project\results\plots\master_macro_f1_vs_accuracy_qwen.png"
if os.path.exists(img_bar_path):
    story.append(Image(img_bar_path, width=480, height=270))
    story.append(Paragraph("<b>Figure 1:</b> Test Accuracy and Macro F1 comparison across all eleven architectures.", caption_style))
    story.append(Spacer(1, 8))

# Subsections
story.append(Paragraph("4.1 Bidirectional Encoders (BERT) vs. Causal Decoders (Qwen2.5 LoRA)", h2_style))
disc_qwen = (
    "A key architectural question investigated in this work is whether modern 1.5B+ parameter autoregressive decoder SLMs "
    "outperform classical bidirectional encoder transformers on long financial text classification.<br/>"
    "<b>Empirical Findings:</b> (1) <b>BERT Base maintains superiority in Macro F1 (0.7738 vs. 0.7427) and Accuracy (86.51% vs. 82.84%)</b>. "
    "Because BERT utilizes unconstrained bidirectional self-attention, each token representation directly incorporates both preceding and succeeding "
    "clauses simultaneously. In contrast, Qwen2.5 is constrained by causal triangular masking during pre-training, making classification dependent "
    "on cumulative sequence representations at the final token.<br/>"
    "(2) <b>Qwen2.5 LoRA strongly outperforms all recurrent neural networks</b> (0.7427 vs. 0.7248 Bi-GRU and 0.7169 Bi-LSTM), demonstrating that "
    "multi-head self-attention over pretrained causal representations is more expressive than sequential recurrence over Word2Vec embeddings.<br/>"
    "(3) <b>Inference Efficiency:</b> BERT processes test samples in <b>4.82 ms/sample</b>, representing a <b>13x speedup over Qwen2.5 (62.16 ms/sample)</b>. "
    "For high-throughput regulatory pipelines processing millions of filings daily, BERT Base and Logistic Regression (0.10 ms) occupy the optimal "
    "Pareto efficiency frontier."
)
story.append(Paragraph(disc_qwen, body_style))

img_scatter_path = r"E:\BRACU\26-SUMMER\440\CSE440 Lab Project\results\plots\latency_vs_macro_f1_frontier_qwen.png"
if os.path.exists(img_scatter_path):
    story.append(Image(img_scatter_path, width=460, height=270))
    story.append(Paragraph("<b>Figure 2:</b> Performance vs. Latency Pareto efficiency frontier across the four model paradigms.", caption_style))
    story.append(Spacer(1, 8))

# Confusion Matrix for Qwen
story.append(Paragraph("4.2 Granular Error Analysis & Confusion Matrix for Qwen2.5", h2_style))
img_cm_path = r"E:\BRACU\26-SUMMER\440\CSE440 Lab Project\results\plots\cm_qwen2.5_1.5b_lora.png"
if os.path.exists(img_cm_path):
    story.append(Image(img_cm_path, width=380, height=300))
    story.append(Paragraph("<b>Figure 3:</b> Normalized confusion matrix for Qwen2.5 (1.5B LoRA) across the 9 CFPB categories.", caption_style))
    story.append(Spacer(1, 6))

cm_disc = (
    "Figure 3 details the per-class confusion patterns of Qwen2.5 (1.5B LoRA). Like BERT and Logistic Regression, Qwen achieves high true-positive "
    "rates on <i>Credit reporting</i> (0.92), <i>Mortgage</i> (0.88), and <i>Student loan</i> (0.87). The primary residual confusion occurs between "
    "<i>Debt collection</i> and <i>Credit reporting</i> (12% error rate), directly reflecting underlying complaint semantics where consumers contest "
    "unauthorized collection entries appearing on their credit bureau files."
)
story.append(Paragraph(cm_disc, body_style))

# 5. Conclusion
story.append(Paragraph("5. Conclusion & Recommendations", h1_style))
conc_p = (
    "Across 2.02 million CFPB complaints, our findings demonstrate that: (1) <b>BERT Base</b> provides the strongest overall classification performance "
    "(0.7738 Macro F1, 86.51% Accuracy) with fast inference (4.82 ms); (2) <b>Qwen2.5 (1.5B LoRA)</b> represents a powerful parameter-efficient decoder "
    "benchmark (0.7427 Macro F1) that beats all recurrent models while training only 1.18% of weights; (3) <b>Logistic Regression</b> remains the most cost-effective "
    "production baseline (0.7367 Macro F1 at 0.10 ms latency); (4) SMOTE oversampling is critical for tree ensembles under severe class imbalance (+0.2809 F1 boost on RF), "
    "while deep models excel with class-weighted loss."
)
story.append(Paragraph(conc_p, body_style))

# Build Document
doc.build(story)
print(f"Successfully generated updated academic PDF: {pdf_path}")
