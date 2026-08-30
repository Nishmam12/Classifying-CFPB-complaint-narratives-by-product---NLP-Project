import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout.reconfigure(encoding='utf-8')

# 1. Setup and Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
NUM_LABELS = 9
MAX_LEN = 256
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2
EPOCHS = 2
LR = 2e-4
WEIGHT_DECAY = 0.01
SEED = 42

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"=== Training Qwen2.5 (1.5B) with LoRA on {torch.cuda.get_device_name(0)} ===")

# 2. Data Loading
DATA_DIR = r"E:\BRACU\26-SUMMER\440\CSE440 Lab Project_4extra models\processed_data"
train_df = pd.read_parquet(os.path.join(DATA_DIR, "train_subset.parquet"))
val_df = pd.read_parquet(os.path.join(DATA_DIR, "val_subset.parquet"))
test_df = pd.read_parquet(os.path.join(DATA_DIR, "test_subset.parquet"))

with open(os.path.join(DATA_DIR, "label_mapping.json"), "r") as f:
    mapping = json.load(f)
id2label = {int(k): v for k, v in mapping["id2label"].items()}
label2id = {k: int(v) for k, v in mapping["label2id"].items()}
target_names = [id2label[i] for i in range(NUM_LABELS)]

print(f"Loaded datasets: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

# Compute class weights
class_counts = train_df["label_id"].value_counts().sort_index().values
total_samples = len(train_df)
class_weights = total_samples / (NUM_LABELS * class_counts)
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print(f"Class weights: {np.round(class_weights, 3)}")

# 3. Tokenizer & Dataset
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

class ComplaintDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return item

train_dataset = ComplaintDataset(train_df["narrative_contextual"], train_df["label_id"], tokenizer, MAX_LEN)
val_dataset = ComplaintDataset(val_df["narrative_contextual"], val_df["label_id"], tokenizer, MAX_LEN)
test_dataset = ComplaintDataset(test_df["narrative_contextual"], test_df["label_id"], tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, pin_memory=True)

# 4. Model & LoRA Setup
print("Loading base model Qwen2.5-1.5B ...")
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    torch_dtype=torch.bfloat16,
    id2label=id2label,
    label2id=label2id
)
base_model.config.pad_token_id = tokenizer.pad_token_id

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    modules_to_save=["score"]
)

model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()
model.to(DEVICE)

# 5. Optimizer, Scheduler, Loss
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    start_time = time.time()
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)
                
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    eval_time = time.time() - start_time
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    avg_loss = total_loss / len(loader)
    latency_ms = (eval_time / len(all_labels)) * 1000.0
    
    return avg_loss, acc, macro_f1, weighted_f1, np.array(all_preds), np.array(all_labels), latency_ms

# 6. Training Loop
print("\n=== Starting LoRA Training ===")
best_val_f1 = 0.0
train_start = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    epoch_start = time.time()
    optimizer.zero_grad()
    
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels) / GRAD_ACCUM_STEPS
            
        loss.backward()
        running_loss += loss.item() * GRAD_ACCUM_STEPS
        
        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        if (step + 1) % 200 == 0:
            print(f"  Epoch {epoch}/{EPOCHS} | Step {step+1}/{len(train_loader)} | Loss: {running_loss/(step+1):.4f}")
            
    train_duration = time.time() - epoch_start
    val_loss, val_acc, val_macro_f1, val_weighted_f1, _, _, val_lat = evaluate(model, val_loader)
    print(f"Epoch {epoch}/{EPOCHS} completed in {train_duration:.1f}s | Train Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f} | Val Macro F1: {val_macro_f1:.4f} | Val Weighted F1: {val_weighted_f1:.4f}")
    
    if val_macro_f1 > best_val_f1:
        best_val_f1 = val_macro_f1
        os.makedirs(r"E:\BRACU\26-SUMMER\440\CSE440 Lab Project\_cache\qwen_lora_best", exist_ok=True)
        model.save_pretrained(r"E:\BRACU\26-SUMMER\440\CSE440 Lab Project\_cache\qwen_lora_best")
        tokenizer.save_pretrained(r"E:\BRACU\26-SUMMER\440\CSE440 Lab Project\_cache\qwen_lora_best")

total_train_time = time.time() - train_start
print(f"\nTraining completed in {total_train_time:.1f}s. Best Val Macro F1: {best_val_f1:.4f}")

# 7. Test Set Evaluation
print("\n=== Evaluating on Held-Out Test Set (5,000 samples) ===")
test_loss, test_acc, test_macro_f1, test_weighted_f1, test_preds, test_labels, latency_ms = evaluate(model, test_loader)

# Minority-4 F1: [Payday(6), Vehicle(8), Money(4), Student(7)]
f1_per_class = f1_score(test_labels, test_preds, average=None, zero_division=0)
minority_ids = [6, 8, 4, 7]
minority_f1 = np.mean([f1_per_class[i] for i in minority_ids])

print(f"Test Accuracy:    {test_acc*100:.2f}%")
print(f"Test Macro F1:    {test_macro_f1:.4f}")
print(f"Test Weighted F1: {test_weighted_f1:.4f}")
print(f"Minority-4 F1:    {minority_f1:.4f}")
print(f"Inference Latency:{latency_ms:.2f} ms/sample")

# 8. Save Metrics & Confusion Matrix
results = {
    "model_name": "Qwen2.5 (1.5B LoRA)",
    "paradigm": "Causal Decoder SLM",
    "parameters": "1.54B (LoRA r=16)",
    "training_time_s": round(total_train_time, 2),
    "test_accuracy": round(test_acc * 100, 2),
    "test_macro_f1": round(test_macro_f1, 4),
    "test_weighted_f1": round(test_weighted_f1, 4),
    "minority_4_f1": round(minority_f1, 4),
    "latency_ms": round(latency_ms, 2)
}

os.makedirs(r"E:\BRACU\26-SUMMER\440\CSE440 Lab Project\_cache", exist_ok=True)
with open(r"E:\BRACU\26-SUMMER\440\CSE440 Lab Project\_cache\qwen_lora_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Confusion matrix plot
cm = confusion_matrix(test_labels, test_preds, normalize="true")
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.title("Normalized Confusion Matrix — Qwen2.5 (1.5B LoRA)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
os.makedirs(r"E:\BRACU\26-SUMMER\440\CSE440 Lab Project\_cache\plots", exist_ok=True)
plt.savefig(r"E:\BRACU\26-SUMMER\440\CSE440 Lab Project\_cache\plots\cm_qwen2.5_1.5b_lora.png", dpi=300)
plt.close()

print("Saved metrics to _cache/qwen_lora_results.json and confusion matrix plot.")
