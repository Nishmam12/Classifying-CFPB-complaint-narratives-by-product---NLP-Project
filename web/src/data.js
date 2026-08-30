// Complete empirical benchmark dataset from CSE440 Lab Project (held-out test set: 303,213 documents)

export const CLASSES_META = [
  {
    id: 0,
    name: "Bank account or service",
    shortName: "Bank Account",
    color: "#3b82f6", // Blue
    icon: "🏦",
    description: "Checking/savings accounts, unauthorized overdraft fees, branch deposits, and teller disputes.",
    keywords: ["bank", "checking", "savings", "overdraft", "deposit", "fee", "atm", "funds", "teller", "branch", "chase", "wells", "fargo", "citi"]
  },
  {
    id: 1,
    name: "Credit card / prepaid card",
    shortName: "Credit Card",
    color: "#8b5cf6", // Purple
    icon: "💳",
    description: "Credit limits, annual fees, merchant dispute chargebacks, cash advances, reward points, and prepaid cards.",
    keywords: ["card", "credit", "charge", "merchant", "rewards", "billing", "statement", "annual fee", "interest rate", "amex", "visa", "mastercard"]
  },
  {
    id: 2,
    name: "Credit reporting",
    shortName: "Credit Reporting",
    color: "#10b981", // Emerald
    icon: "📈",
    description: "Inaccurate tradelines, FCRA dispute failures, identity theft inquiries, and credit score suppression.",
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
    description: "Domestic/international wire transfers, Zelle/Venmo frauds, cryptocurrency exchange holds, and remit issues.",
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
    description: "High-interest payday advances, auto title pawn loans, installment personal loans, and predatory APR traps.",
    keywords: ["payday", "title loan", "installment", "personal loan", "apr", "interest", "finance charge", "lender", "cash advance", "short term"]
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
    keywords: ["car", "vehicle", "auto", "lease", "dealership", "repossession", "gap insurance", "title", "lien", "ally", "toyota financial"]
  }
];

export const BENCHMARK_MODELS = [
  {
    id: "ensemble",
    name: "Ensemble (BERT + LR)",
    paradigm: "Ensemble",
    config: "Soft-Voting (0.75 BERT + 0.25 LR)",
    macroF1: 0.7720,
    accuracy: 0.8624,
    weightedF1: 0.8685,
    trainTime: "1712.6 s",
    inferTime: "267.5 s",
    isEnsemble: true,
    highlight: "Top Benchmark (+0.73 pp boost over BERT Base)"
  },
  {
    id: "bert",
    name: "BERT Base",
    paradigm: "Transformer",
    config: "BERT-Base-Uncased (lr=2e-5, bs=32)",
    macroF1: 0.7647,
    accuracy: 0.8562,
    weightedF1: 0.8613,
    trainTime: "1678.1 s",
    inferTime: "267.3 s",
    isBestSingle: true,
    highlight: "Best Standalone Model (Deep Contextual Self-Attention)"
  },
  {
    id: "lr",
    name: "Logistic Regression",
    paradigm: "Classical ML",
    config: "TF-IDF 25k (C=1.0, L2 penalty)",
    macroF1: 0.7367,
    accuracy: 0.8390,
    weightedF1: 0.8468,
    trainTime: "34.5 s",
    inferTime: "0.2 s",
    highlight: "Highest Efficiency (1,300x faster than BERT at 96% F1)"
  },
  {
    id: "gru",
    name: "GRU (Gated Recurrent)",
    paradigm: "Recurrent Net",
    config: "Config-2 (Hidden 128, Word2Vec 100d)",
    macroF1: 0.7274,
    accuracy: 0.8326,
    weightedF1: 0.8420,
    trainTime: "77.2 s",
    inferTime: "5.8 s",
    highlight: "Top Recurrent Model (Outperforms LSTM with faster training)"
  },
  {
    id: "bigru",
    name: "Bidirectional GRU",
    paradigm: "Recurrent Net",
    config: "Config-2 (Hidden 128, Bidirectional)",
    macroF1: 0.7234,
    accuracy: 0.8267,
    weightedF1: 0.8376,
    trainTime: "130.6 s",
    inferTime: "8.4 s",
    highlight: "Bidirectional Gated Recurrent"
  },
  {
    id: "nb",
    name: "Naive Bayes",
    paradigm: "Classical ML",
    config: "MultinomialNB (alpha=0.01)",
    macroF1: 0.7211,
    accuracy: 0.8329,
    weightedF1: 0.8357,
    trainTime: "0.1 s",
    inferTime: "0.2 s",
    highlight: "Sub-second Fast Baseline"
  },
  {
    id: "bilstm",
    name: "Bidirectional LSTM",
    paradigm: "Recurrent Net",
    config: "Config-1 (Hidden 128, Word2Vec 100d)",
    macroF1: 0.7173,
    accuracy: 0.8175,
    weightedF1: 0.8274,
    trainTime: "139.1 s",
    inferTime: "9.2 s",
    highlight: "Standard Deep Recurrent Architecture"
  },
  {
    id: "lstm",
    name: "LSTM",
    paradigm: "Recurrent Net",
    config: "Config-2 (Hidden 128, Dropout 0.3)",
    macroF1: 0.7110,
    accuracy: 0.8180,
    weightedF1: 0.8276,
    trainTime: "78.2 s",
    inferTime: "5.8 s",
    highlight: "Unidirectional LSTM Baseline"
  },
  {
    id: "rf",
    name: "Random Forest",
    paradigm: "Classical ML",
    config: "RF-n50-d50 (Balanced Class Weights)",
    macroF1: 0.6920,
    accuracy: 0.8121,
    weightedF1: 0.8197,
    trainTime: "31.0 s",
    inferTime: "0.8 s",
    highlight: "Tree Ensemble Baseline"
  },
  {
    id: "birnn",
    name: "Bidirectional SimpleRNN",
    paradigm: "Recurrent Net",
    config: "Config-2 (Hidden 128, Bidirectional)",
    macroF1: 0.6600,
    accuracy: 0.7824,
    weightedF1: 0.7944,
    trainTime: "124.2 s",
    inferTime: "8.9 s",
    highlight: "+9.15 pp boost from bidirectionality"
  },
  {
    id: "rnn",
    name: "SimpleRNN",
    paradigm: "Recurrent Net",
    config: "Config-3 (Hidden 128, Vanilla)",
    macroF1: 0.5685,
    accuracy: 0.6722,
    weightedF1: 0.6943,
    trainTime: "80.5 s",
    inferTime: "5.9 s",
    highlight: "Severely limited by gradient decay"
  }
];

export const NOVELTY_EXPERIMENTS = [
  { model: "Logistic Regression", paradigm: "Classical ML", strategy: "None (Natural Prior)", macroF1: 0.7703, acc: 0.8763, min4F1: 0.6829, finding: "Highest untreated performance; natural prior preserves precision." },
  { model: "Logistic Regression", paradigm: "Classical ML", strategy: "Class Weighting", macroF1: 0.7367, acc: 0.8390, min4F1: 0.6347, finding: "Degrades F1 (-3.36 pp) by over-predicting rare classes without boosting true signal." },
  { model: "Logistic Regression", paradigm: "Classical ML", strategy: "SMOTE (100k)", macroF1: 0.7348, acc: 0.8435, min4F1: 0.6353, finding: "Synthetic samples blur high-dimensional 25k-dim boundary; 8.6x training cost." },
  { model: "Bidirectional LSTM", paradigm: "Recurrent Net", strategy: "None (Natural Prior)", macroF1: 0.7627, acc: 0.8742, min4F1: 0.6718, finding: "Best recurrent F1 without artificial gradient skew." },
  { model: "Bidirectional LSTM", paradigm: "Recurrent Net", strategy: "Class Weighting", macroF1: 0.7088, acc: 0.8167, min4F1: 0.5931, finding: "Causes severe -7.87 pp penalty on minority-4 classes due to false positives." },
  { model: "Bidirectional LSTM", paradigm: "Recurrent Net", strategy: "Focal Loss (γ=2)", macroF1: 0.7090, acc: 0.8163, min4F1: 0.6010, finding: "Focuses on hard examples but reduces precision across tail categories." },
  { model: "Random Forest", paradigm: "Classical ML", strategy: "None (Untreated)", macroF1: 0.5702, acc: 0.8138, min4F1: 0.3389, finding: "Completely collapses on rare classes (0.3389 min-4) due to majority split dominance." },
  { model: "Random Forest", paradigm: "Classical ML", strategy: "Class Weighting", macroF1: 0.6920, acc: 0.8121, min4F1: 0.5820, finding: "Rescues tree model (+24.31 pp on minority-4 classes) by weighting leaf impurity." },
  { model: "Random Forest", paradigm: "Classical ML", strategy: "SMOTE (100k)", macroF1: 0.7142, acc: 0.8277, min4F1: 0.6198, finding: "Best Random Forest result (+28.09 pp boost) by balancing split candidates." },
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
    id: "confusion_matrices",
    src: "/figures/confusion_matrices.png",
    title: "Normalized Confusion Matrices (Key Paradigms)",
    category: "Error Dynamics",
    description: "Evaluated on 303,213 held-out test documents. Demonstrates strong diagonal dominance in BERT Base and reveals the primary structural confusion between Money Transfer and Bank Account (14–17%) caused by lexical overlap."
  },
  {
    id: "wordclouds",
    src: "/figures/wordclouds.png",
    title: "Salient Vocabulary Distributions (Word Clouds)",
    category: "Lexical Overlap",
    description: "Multi-class word clouds showing frequent n-grams across financial categories. Explicitly illustrates why Money Transfer and Bank Account share high-frequency terms like 'bank', 'account', 'deposit', 'funds', and major institution names."
  },
  {
    id: "model_comparison",
    src: "/figures/model_comparison.png",
    title: "Accuracy vs. Latency Pareto Frontier",
    category: "Benchmarking",
    description: "Pareto efficiency comparison across all 10 architectures. Logistic Regression captures 96% of BERT's Macro F1 at 1,300x lower inference latency (0.2s vs 267.3s for 303k documents)."
  },
  {
    id: "imbalance_comparison",
    src: "/figures/imbalance_comparison.png",
    title: "Imbalance Strategy Performance Delta",
    category: "Ablation Studies",
    description: "13-run controlled study comparing Natural Prior, Class Weighting, SMOTE, and Focal Loss. Demonstrates that class weighting penalizes deep neural networks (-7.9 pp on Bi-LSTM) while being indispensable for Random Forest (+24.3 pp)."
  },
  {
    id: "class_distribution_9",
    src: "/figures/class_distribution_9.png",
    title: "9-Class Target Product Distribution",
    category: "EDA & Class Skew",
    description: "Target class distribution of the 2,021,420 CFPB complaints after mapping to 9 standardized products. Shows heavy class skew from Credit Reporting (59.6%) down to Payday Loans (1.1%)."
  },
  {
    id: "split_stratification",
    src: "/figures/split_stratification.png",
    title: "70/15/15 Stratified Split Verification",
    category: "Dataset Splits",
    description: "Deterministic split stratification ensuring identical class ratios across Train (1,414,994), Validation (303,213), and Held-Out Test (303,213) sets."
  },
  {
    id: "tuning_overview",
    src: "/figures/tuning_overview.png",
    title: "Hyperparameter Tuning Progression",
    category: "Tuning Runs",
    description: "Validation Macro F1 across 30 controlled tuning runs (Classical ML x3, Recurrent Nets x18, BERT Base x3). Documents convergence and optimal hyperparameter selection."
  },
  {
    id: "preprocessing_effect",
    src: "/figures/preprocessing_effect.png",
    title: "Preprocessing & Normalization Impact",
    category: "Preprocessing",
    description: "Visualizing the token reduction and normalization effects of anonymization masking (e.g. 'XXXX'), lowercase conversion, contraction expansion, and punctuation handling."
  },
  {
    id: "narrative_length",
    src: "/figures/narrative_length.png",
    title: "Narrative Token Length Distribution",
    category: "Token Dynamics",
    description: "Histogram and cumulative distribution of token lengths. Reveals a median cleaned length of 44 tokens with a 95th percentile at 256 tokens, confirming the necessity of pack_padded_sequence masking."
  },
  {
    id: "raw_class_distribution",
    src: "/figures/raw_class_distribution.png",
    title: "Raw CFPB 18-Class Product Distribution",
    category: "Raw Corpus",
    description: "Initial distribution of raw historical CFPB complaint categories before merging redundant sub-products into the standardized 9-product schema."
  }
];
