import os
import sys
import time
import json
import re
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout.reconfigure(encoding='utf-8')

# 1. Configuration & Paths
MODEL_PATH = r"E:\BRACU\26-SUMMER\440\CSE440 Lab Project_4extra models\Models\NVIDIA-Nemotron-3-Nano-4B-Q4_0.gguf"
DATA_DIR = r"E:\BRACU\26-SUMMER\440\CSE440 Lab Project_4extra models\processed_data"

print(f"=== Evaluating NVIDIA Nemotron-3 Nano (4B GGUF) ===")
print(f"Model Path: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file not found at {MODEL_PATH}")
    sys.exit(1)

# 2. Data Loading
test_df = pd.read_parquet(os.path.join(DATA_DIR, "test_subset.parquet"))
with open(os.path.join(DATA_DIR, "label_mapping.json"), "r") as f:
    mapping = json.load(f)

id2label = {int(k): v for k, v in mapping["id2label"].items()}
label2id = {k: int(v) for k, v in mapping["label2id"].items()}
target_names = [id2label[i] for i in range(9)]

print(f"Loaded test dataset: {len(test_df):,} samples")

# Select a representative stratified sample for zero-shot LLM benchmark (e.g. 200 samples to keep runtime fast)
EVAL_SAMPLES = 200
eval_df = test_df.groupby("label_id", group_keys=False).apply(lambda x: x.sample(n=min(len(x), max(10, EVAL_SAMPLES // 9)), random_state=42)).reset_index(drop=True)
print(f"Selected {len(eval_df)} stratified test samples for zero-shot prompt evaluation.")

# 3. Load llama_cpp Model
try:
    from llama_cpp import Llama
    print("Initializing Llama engine with GPU acceleration (n_gpu_layers=-1) ...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,
        n_ctx=2048,
        n_threads=8,
        verbose=False
    )
    print("Model loaded successfully into GPU memory!")
except Exception as e:
    print(f"llama-cpp-python loading error: {e}")
    sys.exit(1)

# 4. Prompting & Classification
PROMPT_TEMPLATE = """You are a financial regulatory complaint classification assistant. Classify the following consumer complaint into exactly ONE of the 9 valid categories below. Respond with ONLY the exact category name and nothing else.

Categories:
1. Bank account or service
2. Credit card / prepaid card
3. Credit reporting
4. Debt collection
5. Money transfer / virtual currency
6. Mortgage
7. Payday / title / personal loan
8. Student loan
9. Vehicle / consumer loan

Complaint:
"{text}"

Category:"""

def clean_and_match_category(output_text):
    text = output_text.strip().lower()
    for cat in target_names:
        if cat.lower() in text:
            return label2id[cat]
    # Fuzzy keyword heuristics
    if "credit report" in text or "reporting" in text or "equifax" in text or "experian" in text or "transunion" in text:
        return label2id["Credit reporting"]
    if "debt" in text or "collection" in text:
        return label2id["Debt collection"]
    if "credit card" in text or "card" in text or "prepaid" in text:
        return label2id["Credit card / prepaid card"]
    if "mortgage" in text:
        return label2id["Mortgage"]
    if "bank" in text or "checking" in text or "saving" in text:
        return label2id["Bank account or service"]
    if "student" in text:
        return label2id["Student loan"]
    if "transfer" in text or "wire" in text or "crypto" in text or "currency" in text:
        return label2id["Money transfer / virtual currency"]
    if "vehicle" in text or "auto" in text or "car" in text:
        return label2id["Vehicle / consumer loan"]
    if "payday" in text or "title loan" in text or "personal loan" in text:
        return label2id["Payday / title / personal loan"]
    return label2id["Credit reporting"] # fallback majority

preds = []
true_labels = eval_df["label_id"].tolist()
latencies = []

print("\nRunning Zero-Shot Inference on GPU ...")
start_all = time.time()

for idx, row in eval_df.iterrows():
    complaint_text = str(row["narrative_contextual"])[:1000] # truncate narrative for prompt
    prompt = PROMPT_TEMPLATE.format(text=complaint_text)
    
    t0 = time.time()
    response = llm(
        prompt,
        max_tokens=20,
        temperature=0.1,
        stop=["\n", "\n\n", "Complaint:"]
    )
    t1 = time.time()
    
    gen_text = response["choices"][0]["text"]
    pred_class = clean_and_match_category(gen_text)
    preds.append(pred_class)
    latencies.append((t1 - t0) * 1000.0) # ms
    
    if (idx + 1) % 25 == 0 or (idx + 1) == len(eval_df):
        print(f"  Processed {idx+1}/{len(eval_df)} samples | Avg Latency: {np.mean(latencies):.1f} ms")

total_eval_time = time.time() - start_all

# 5. Metrics
acc = accuracy_score(true_labels, preds)
macro_f1 = f1_score(true_labels, preds, average="macro", zero_division=0)
weighted_f1 = f1_score(true_labels, preds, average="weighted", zero_division=0)
avg_latency = np.mean(latencies)

f1_per_class = f1_score(true_labels, preds, average=None, zero_division=0)
minority_ids = [6, 8, 4, 7]
minority_f1 = np.mean([f1_per_class[i] for i in minority_ids])

print(f"\n=== NVIDIA Nemotron-3 Nano (4B GGUF) Evaluation Results ===")
print(f"Accuracy:         {acc*100:.2f}%")
print(f"Macro F1:         {macro_f1:.4f}")
print(f"Weighted F1:      {weighted_f1:.4f}")
print(f"Minority-4 F1:    {minority_f1:.4f}")
print(f"Average Latency:  {avg_latency:.2f} ms/sample")

# 6. Save Results
results = {
    "model_name": "NVIDIA Nemotron-3 (Nano 4B GGUF)",
    "paradigm": "Quantized Causal SLM",
    "parameters": "~4.0B (Q4_0)",
    "training_type": "Zero-Shot Prompting",
    "test_accuracy": round(acc * 100, 2),
    "test_macro_f1": round(macro_f1, 4),
    "test_weighted_f1": round(weighted_f1, 4),
    "minority_4_f1": round(minority_f1, 4),
    "latency_ms": round(avg_latency, 2),
    "eval_samples": len(eval_df)
}

os.makedirs(r"E:\BRACU\26-SUMMER\440\CSE440 Lab Project\_cache", exist_ok=True)
with open(r"E:\BRACU\26-SUMMER\440\CSE440 Lab Project\_cache\nemotron_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Wrote results to _cache/nemotron_results.json")
