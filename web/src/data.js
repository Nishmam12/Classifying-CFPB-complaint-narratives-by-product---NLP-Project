// Complete empirical benchmark dataset from CSE440 Lab Project (held-out test set: 303,213 documents)

export const CLASSES_META = [
  {
    id: 0,
    name: "Bank account or service",
    shortName: "Bank Account",
    color: "#3b82f6", // Blue
    icon: "🏦",
    description: "Checking/savings accounts, unauthorized overdraft fees, branch deposits, teller disputes, and direct deposit holds.",
    keywords: ["bank", "checking", "savings", "overdraft", "deposit", "fee", "atm", "funds", "teller", "branch", "chase", "wells", "fargo", "citi", "bofa"]
  },
  {
    id: 1,
    name: "Credit card / prepaid card",
    shortName: "Credit Card",
    color: "#8b5cf6", // Purple
    icon: "💳",
    description: "Credit limits, annual fees, merchant dispute chargebacks, cash advances, reward points, and prepaid cards.",
    keywords: ["card", "credit", "charge", "merchant", "rewards", "billing", "statement", "annual fee", "interest rate", "amex", "visa", "mastercard", "prepaid"]
  },
  {
    id: 2,
    name: "Credit reporting",
    shortName: "Credit Reporting",
    color: "#10b981", // Emerald
    icon: "📈",
    description: "Inaccurate tradelines, FCRA dispute failures, identity theft inquiries, and credit score suppression by bureaus.",
    keywords: ["equifax", "experian", "transunion", "reporting", "bureau", "dispute", "inaccurate", "score", "inquiry", "fcra", "investigation", "tradeline"]
  },
  {
    id: 3,
    name: "Debt collection",
    shortName: "Debt Collection",
    color: "#f59e0b", // Amber
    icon: "📞",
    description: "FDCPA violations, persistent harassment calls, phantom debt collection, and unverified validation requests.",
    keywords: ["debt", "collector", "collection", "agency", "harass", "fdcpa", "validate", "validation", "owed", "medical bill", "portfolio", "cease"]
  },
  {
    id: 4,
    name: "Money transfer / virtual currency",
    shortName: "Money Transfer",
    color: "#06b6d4", // Cyan
    icon: "💸",
    description: "Domestic/international wire transfers, Zelle/Venmo frauds, cryptocurrency exchange holds, and remittance issues.",
    keywords: ["transfer", "wire", "zelle", "venmo", "paypal", "crypto", "bitcoin", "coinbase", "remittance", "sent", "recipient", "wallet", "scam"]
  },
  {
    id: 5,
    name: "Mortgage",
    shortName: "Mortgage",
    color: "#ec4899", // Pink
    icon: "🏠",
    description: "Home loan originations, escrow shortage disputes, modification denials, foreclosure notices, and servicing transfers.",
    keywords: ["mortgage", "loan", "escrow", "foreclosure", "modification", "servicing", "property", "home", "deed", "monthly payment", "pmi", "refinance"]
  },
  {
    id: 6,
    name: "Payday / title / personal loan",
    shortName: "Payday / Personal Loan",
    color: "#ef4444", // Red
    icon: "⚡",
    description: "High-interest payday advances, auto title pawn loans, installment personal loans, and predatory 400%+ APR traps.",
    keywords: ["payday", "title loan", "installment", "personal loan", "apr", "interest", "finance charge", "lender", "cash advance", "short term", "rollover"]
  },
  {
    id: 7,
    name: "Student loan",
    shortName: "Student Loan",
    color: "#6366f1", // Indigo
    icon: "🎓",
    description: "Federal & private student debt, PSLF forgiveness, income-driven repayment (IDR) misallocation, and Navient/Nelnet servicing.",
    keywords: ["student", "tuition", "navient", "nelnet", "mohela", "pslf", "forgiveness", "deferment", "forbearance", "idr", "education", "school"]
  },
  {
    id: 8,
    name: "Vehicle / consumer loan",
    shortName: "Vehicle Loan",
    color: "#14b8a6", // Teal
    icon: "🚗",
    description: "Auto financing, vehicle leasing, GAP insurance refund disputes, repossession errors, and dealer markups.",
    keywords: ["car", "vehicle", "auto", "lease", "dealership", "repossession", "gap insurance", "title", "lien", "ally", "toyota financial", "ford credit"]
  }
];

export const BENCHMARK_MODELS = [
  {
    id: "ensemble",
    name: "Ensemble (Soft-Voting Fusion)",
    paradigm: "Ensemble",
    config: "Soft-Voting (BERT Base + Random Forest + Naive Bayes / LR)",
    macroF1: 0.7792,
    accuracy: 0.8651,
    weightedF1: 0.8684,
    trainTime: "1709.2 s",
    inferTime: "4.92 ms",
    isEnsemble: true,
    highlight: "Top Overall Performance (+1.45 pp gain over best standalone model)"
  },
  {
    id: "bert",
    name: "BERT Base (Fine-Tuned)",
    paradigm: "Bidirectional Encoder",
    config: "bert-base-uncased (110M, lr=3e-5, bs=32, 2 ep)",
    macroF1: 0.7738,
    accuracy: 0.8651,
    weightedF1: 0.8691,
    trainTime: "1714.8 s",
    inferTime: "4.82 ms",
    isBestSingle: true,
    highlight: "Best Standalone Architecture (Deep Bidirectional Multi-Head Attention)"
  },
  {
    id: "qwen",
    name: "Qwen2.5-1.5B (LoRA r=16)",
    paradigm: "Causal Decoder SLM",
    config: "LoRA PEFT (r=16, alpha=32, 18.48M / 1.18% params, lr=2e-4)",
    macroF1: 0.7427,
    accuracy: 0.8284,
    weightedF1: 0.8345,
    trainTime: "1420.0 s",
    inferTime: "62.16 ms",
    isSLM: true,
    highlight: "Top Causal Decoder SLM — Beats all 6 recurrent networks without memory bottleneck"
  },
  {
    id: "lr",
    name: "Logistic Regression",
    paradigm: "Classical ML",
    config: "TF-IDF 25k (C=1.0, L2 penalty, Class-Weighted)",
    macroF1: 0.7367,
    accuracy: 0.8390,
    weightedF1: 0.8468,
    trainTime: "20.5 s",
    inferTime: "0.10 ms",
    highlight: "Maximum Throughput (600x faster than Qwen2.5 and 48x faster than BERT at 95% F1)"
  },
  {
    id: "bigru",
    name: "Bidirectional GRU",
    paradigm: "Recurrent Neural Net",
    config: "Config-2 (Hidden 128, Word2Vec 100d CBOW, lr=1e-3)",
    macroF1: 0.7248,
    accuracy: 0.8284,
    weightedF1: 0.8373,
    trainTime: "83.4 s",
    inferTime: "5.30 ms",
    highlight: "Top Recurrent Architecture — Outperforms Bi-LSTM and all RNN variants"
  },
  {
    id: "nb",
    name: "Naive Bayes (Multinomial)",
    paradigm: "Classical ML",
    config: "TF-IDF 25k (MultinomialNB alpha=0.01)",
    macroF1: 0.7211,
    accuracy: 0.8329,
    weightedF1: 0.8357,
    trainTime: "0.1 s",
    inferTime: "0.10 ms",
    highlight: "Instantaneous Sub-Second Fast Baseline"
  },
  {
    id: "lstm",
    name: "LSTM (Gated Recurrent)",
    paradigm: "Recurrent Neural Net",
    config: "Config-2 (Hidden 128, Dropout 0.3, lr=1e-3)",
    macroF1: 0.7186,
    accuracy: 0.8278,
    weightedF1: 0.8370,
    trainTime: "52.0 s",
    inferTime: "3.70 ms",
    highlight: "Standard Unidirectional Gated Recurrent Baseline"
  },
  {
    id: "bilstm",
    name: "Bidirectional LSTM",
    paradigm: "Recurrent Neural Net",
    config: "Config-2 (Hidden 128, Word2Vec 100d, lr=1e-3)",
    macroF1: 0.7169,
    accuracy: 0.8275,
    weightedF1: 0.8375,
    trainTime: "82.0 s",
    inferTime: "5.50 ms",
    highlight: "Two-way sequential recurrent processing"
  },
  {
    id: "gru",
    name: "GRU (Gated Recurrent Unit)",
    paradigm: "Recurrent Neural Net",
    config: "Config-2 (Hidden 128, Word2Vec 100d, lr=1e-3)",
    macroF1: 0.7163,
    accuracy: 0.8151,
    weightedF1: 0.8262,
    trainTime: "51.5 s",
    inferTime: "3.80 ms",
    highlight: "Efficient single-gate memory dynamics"
  },
  {
    id: "rf",
    name: "Random Forest",
    paradigm: "Classical ML",
    config: "TF-IDF 25k (50 Trees, Max Depth 50, Balanced)",
    macroF1: 0.6920,
    accuracy: 0.8121,
    weightedF1: 0.8197,
    trainTime: "20.0 s",
    inferTime: "0.50 ms",
    highlight: "Rescued by class weighting & SMOTE (+28.09 pp minority boost)"
  },
  {
    id: "birnn",
    name: "Bidirectional SimpleRNN",
    paradigm: "Recurrent Neural Net",
    config: "Config-2 (Hidden 128, pack_padded_sequence)",
    macroF1: 0.6568,
    accuracy: 0.7775,
    weightedF1: 0.7951,
    trainTime: "82.3 s",
    inferTime: "5.40 ms",
    highlight: "+7.31 pp Macro F1 gain over unidirectional SimpleRNN"
  },
  {
    id: "rnn",
    name: "SimpleRNN (Vanilla)",
    paradigm: "Recurrent Neural Net",
    config: "Config-3 (Hidden 128, lr=5e-4)",
    macroF1: 0.5837,
    accuracy: 0.7343,
    weightedF1: 0.7524,
    trainTime: "52.7 s",
    inferTime: "4.00 ms",
    highlight: "Severely hindered by vanishing gradients over long sequences"
  }
];

export const NOVELTY_EXPERIMENTS = [
  { model: "Logistic Regression", paradigm: "Classical ML", strategy: "None (Natural Prior)", macroF1: 0.7703, acc: 0.8763, min4F1: 0.6829, finding: "Highest untreated performance; natural class prior preserves precision." },
  { model: "Logistic Regression", paradigm: "Classical ML", strategy: "Class Weighting", macroF1: 0.7367, acc: 0.8390, min4F1: 0.6347, finding: "Slightly reduces precision (-3.36 pp) by penalizing dominant class predictions." },
  { model: "Logistic Regression", paradigm: "Classical ML", strategy: "SMOTE (100k)", macroF1: 0.7348, acc: 0.8435, min4F1: 0.6353, finding: "Synthetic samples blur high-dimensional 25k-dim boundary with 11.5x train cost." },
  { model: "Bidirectional LSTM", paradigm: "Recurrent Net", strategy: "None (Natural Prior)", macroF1: 0.7627, acc: 0.8742, min4F1: 0.6718, finding: "Best recurrent F1 without artificial gradient skew." },
  { model: "Bidirectional LSTM", paradigm: "Recurrent Net", strategy: "Class Weighting", macroF1: 0.7088, acc: 0.8167, min4F1: 0.5931, finding: "Severe -7.87 pp penalty on minority-4 classes due to false-positive escalation." },
  { model: "Bidirectional LSTM", paradigm: "Recurrent Net", strategy: "Focal Loss (γ=2)", macroF1: 0.7090, acc: 0.8163, min4F1: 0.6010, finding: "Down-weights easy examples but triggers modest precision loss on tail classes." },
  { model: "Random Forest", paradigm: "Classical ML", strategy: "None (Untreated)", macroF1: 0.5702, acc: 0.8138, min4F1: 0.3389, finding: "Completely collapses on rare classes (0.3389 min-4) due to majority split dominance." },
  { model: "Random Forest", paradigm: "Classical ML", strategy: "Class Weighting", macroF1: 0.6920, acc: 0.8121, min4F1: 0.5820, finding: "Rescues tree model (+24.31 pp on minority-4 classes) by weighting leaf impurity." },
  { model: "Random Forest", paradigm: "Classical ML", strategy: "SMOTE (100k)", macroF1: 0.7142, acc: 0.8277, min4F1: 0.6198, finding: "Peak Random Forest result (+28.09 pp boost) by balancing split candidates." },
  { model: "BERT Base", paradigm: "Transformer", strategy: "Class Weighting", macroF1: 0.7647, acc: 0.8562, min4F1: 0.6789, finding: "Stable contextual generalization across all 9 classes (0.6789 min-4 F1)." },
  { model: "BERT Base", paradigm: "Transformer", strategy: "Focal Loss (γ=2)", macroF1: 0.7506, acc: 0.8322, min4F1: 0.6736, finding: "Yields slightly lower macro F1 than standard weighted cross-entropy." }
];

export const SAMPLE_COMPLAINTS = [
  {
    label: "Credit reporting",
    classId: 2,
    title: "Inaccurate Equifax Collection Item",
    text: "I submitted multiple certified dispute letters to Equifax and Experian regarding a fraudulent collection tradeline from Portfolio Recovery Associates. Despite providing an identity theft report and police documentation, the credit bureau verified the item without conducting a reasonable investigation under Section 611 of the FCRA. This erroneous reporting lowered my credit score by 85 points."
  },
  {
    label: "Money transfer / virtual currency",
    classId: 4,
    title: "Unauthorized Zelle Wire Transfer",
    text: "Someone gained unauthorized access to my online portal and initiated two rapid Zelle money transfers totaling $2,450 to an unknown recipient. I immediately notified the fraud department within 30 minutes, but the bank refused to reimburse the stolen funds, claiming the transaction was authenticated. I have requested transaction logs and tracing records."
  },
  {
    label: "Mortgage",
    classId: 5,
    title: "Escrow Shortage & Foreclosure Threat",
    text: "My mortgage loan servicer transferred my account to a new lender without proper notice. The new servicer miscalculated my annual property tax escrow analysis, resulting in a fabricated $4,200 shortage. They increased my monthly mortgage payment drastically and sent a notice of default and threat of foreclosure even though all regular principal and interest payments were submitted on time."
  },
  {
    label: "Debt collection",
    classId: 3,
    title: "Harassing Phone Calls for Unknown Debt",
    text: "A third-party collection agency named Enhanced Recovery Company is calling my workplace and cellular phone 6 to 8 times daily attempting to collect a medical debt that is not mine. I sent a formal debt validation letter requesting verification of the alleged balance, but they failed to provide proof and continue calling family members in direct violation of the FDCPA."
  },
  {
    label: "Student loan",
    classId: 7,
    title: "PSLF Payment Misallocation by Servicer",
    text: "I have been enrolled in the Public Service Loan Forgiveness (PSLF) program for over 8 years with qualifying full-time employment at a non-profit. Navient / Nelnet repeatedly placed my account into administrative forbearance without my consent, failing to count 26 eligible monthly income-driven repayment payments toward my 120-payment forgiveness requirement."
  },
  {
    label: "Bank account or service",
    classId: 0,
    title: "Surprise Overdraft & Account Freeze",
    text: "My primary checking account was suddenly placed on hold and frozen without any prior alert or explanation. The bank assessed four consecutive $35 overdraft and non-sufficient funds (NSF) charges on transactions processed out of order while my direct deposit paycheck was actively pending."
  },
  {
    label: "Credit card / prepaid card",
    classId: 1,
    title: "Disputed Merchant Double Billing",
    text: "I purchased airline tickets online and the merchant charged my credit card twice for the exact same reservation code. I opened a billing dispute with Chase Card Services, but the representative closed the claim in favor of the merchant without reviewing my uploaded receipts and bank statements."
  },
  {
    label: "Vehicle / consumer loan",
    classId: 8,
    title: "Auto Loan Repossession & GAP Dispute",
    text: "My vehicle was totaled in an accident and my primary auto insurance issued payment to the auto financing lender. However, the financing company failed to apply my prepaid GAP insurance policy to settle the remaining $1,800 balance, instead forwarding my account to an auto repossession recovery unit."
  },
  {
    label: "Payday / title / personal loan",
    classId: 6,
    title: "Predatory Payday Loan Rollover Trap",
    text: "I took out a $500 short-term payday loan with an advertised two-week turnaround. The lender automatically debited finance charges of $150 every bi-weekly pay cycle without applying any portion toward the principal balance, resulting in an annualized interest rate exceeding 400% APR."
  }
];

export const GALLERY_FIGURES = [
  {
    id: "fig8_master_test_benchmark",
    src: "/figures/fig8_master_test_benchmark.png",
    title: "Figure 8: Master 11-Architecture Benchmark (Macro F1 & Accuracy)",
    category: "Master Benchmark",
    description: "Official comparative performance across all 11 evaluated architectures on 303,213 held-out test documents under the zero-leakage protocol, led by BERT Base (0.7738) and Qwen2.5 LoRA (0.7427)."
  },
  {
    id: "fig11_latency_pareto_frontier",
    src: "/figures/fig11_latency_pareto_frontier.png",
    title: "Figure 11: Performance vs. Latency Pareto Efficiency Frontier",
    category: "Production & Deployment",
    description: "Multi-paradigm efficiency trade-offs: Logistic Regression delivers maximum throughput (0.10 ms), BERT Base provides optimal accuracy (4.82 ms), and Qwen2.5 LoRA represents the high-capacity decoder SLM (62.16 ms)."
  },
  {
    id: "fig9_confusion_matrices_grid",
    src: "/figures/fig9_confusion_matrices_grid.png",
    title: "Figure 9: Normalized Confusion Matrices (Multi-Paradigm Grid)",
    category: "Error Dynamics",
    description: "Evaluated on 303,213 held-out test documents. Demonstrates strong diagonal dominance in BERT Base alongside structural confusion patterns between Money Transfer and Bank Account (14–17%) caused by lexical overlap."
  },
  {
    id: "cm_qwen2.5_1.5b_lora",
    src: "/figures/cm_qwen2.5_1.5b_lora.png",
    title: "Figure 12: Qwen2.5 (1.5B LoRA) Normalized Confusion Matrix",
    category: "Causal Decoder SLM",
    description: "Normalized error matrix for the 1.54B parameter Qwen2.5 Causal Decoder model adapted with Low-Rank Adaptation (r=16, alpha=32), achieving 0.7427 Macro F1 and 82.84% Accuracy."
  },
  {
    id: "fig10_imbalance_novelty",
    src: "/figures/fig10_imbalance_novelty.png",
    title: "Figure 10: Minority-4 Imbalance Mitigation Strategy Comparison",
    category: "Novelty & Imbalance",
    description: "Ablation study comparing Natural Prior, Class Weighting, SMOTE, and Focal Loss (γ=2). Demonstrates that SMOTE oversampling dramatically rescues Random Forest (+0.2809 gain) while natural priors excel for deep and linear models."
  },
  {
    id: "fig4_word_clouds",
    src: "/figures/fig4_word_clouds.png",
    title: "Figure 4: Salient Vocabulary Distributions (Word Clouds)",
    category: "Lexical Overlap",
    description: "Multi-class word clouds showing frequent n-grams across financial categories. Illustrates why Money Transfer and Bank Account share high-frequency terms like 'bank', 'account', 'deposit', 'funds', and major institution names."
  },
  {
    id: "fig7_validation_tuning_ranking",
    src: "/figures/fig7_validation_tuning_ranking.png",
    title: "Figure 7: Hyperparameter Tuning Progression & Validation Ranking",
    category: "Tuning Runs",
    description: "Validation Macro F1 across 30 controlled tuning runs (Classical ML x3, Recurrent Nets x18, BERT Base x3). Documents convergence and optimal hyperparameter selection."
  },
  {
    id: "fig2_consolidated_9classes",
    src: "/figures/fig2_consolidated_9classes.png",
    title: "Figure 2: Nine-Class Target Product Distribution",
    category: "EDA & Class Skew",
    description: "Target class distribution of the 2,021,420 CFPB complaints after mapping to 9 standardized products. Shows heavy class skew from Credit Reporting (59.60%) down to Payday Loans (1.13%)."
  },
  {
    id: "fig1_raw_products",
    src: "/figures/fig1_raw_products.png",
    title: "Figure 1: Raw CFPB 21-Class Product Distribution",
    category: "Raw Corpus",
    description: "Initial distribution of raw historical CFPB complaint categories before merging redundant sub-products into the standardized 9-product schema."
  },
  {
    id: "fig6_split_stratification",
    src: "/figures/fig6_split_stratification.png",
    title: "Figure 6: 70/15/15 Stratified Split Verification",
    category: "Dataset Splits",
    description: "Deterministic split stratification ensuring identical class ratios across Train (1,414,994), Validation (303,213), and Held-Out Test (303,213) sets."
  },
  {
    id: "fig3_word_count_distribution",
    src: "/figures/fig3_word_count_distribution.png",
    title: "Figure 3: Narrative Word Count Distribution",
    category: "Token Dynamics",
    description: "Histogram and cumulative distribution of token lengths (median: 119, IQR: 62–215). Confirms the 256-token truncation cutoff capturing 95.8% of narrative content."
  },
  {
    id: "fig5_preprocessing_lengths",
    src: "/figures/fig5_preprocessing_lengths.png",
    title: "Figure 5: Preprocessing & Normalization Impact",
    category: "Preprocessing",
    description: "Visualizing the token reduction and normalization effects of anonymization masking (e.g. 'XXXX'), lowercase conversion, contraction expansion, and punctuation handling."
  },
  {
    id: "master_macro_f1_vs_accuracy_qwen",
    src: "/figures/master_macro_f1_vs_accuracy_qwen.png",
    title: "Figure 13: Master Macro F1 vs. Accuracy Comparison with Qwen2.5",
    category: "Master Benchmark",
    description: "Comprehensive benchmark comparing BERT Base, Qwen2.5-1.5B LoRA, classical ML, and recurrent neural nets across test macro F1 and classification accuracy."
  },
  {
    id: "latency_vs_macro_f1_frontier_qwen",
    src: "/figures/latency_vs_macro_f1_frontier_qwen.png",
    title: "Figure 14: Causal SLM Latency vs. Macro F1 Frontier",
    category: "Causal Decoder SLM",
    description: "Logarithmic latency vs Macro F1 frontier illustrating where Qwen2.5-1.5B LoRA sits relative to bidirectional BERT Base and ultra-fast linear TF-IDF models."
  }
];
