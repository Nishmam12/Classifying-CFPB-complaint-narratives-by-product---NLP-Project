import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout.reconfigure(encoding='utf-8')

os.makedirs(r"E:\BRACU\26-SUMMER\440\CSE440 Lab Project\_cache\plots", exist_ok=True)
os.makedirs(r"E:\BRACU\26-SUMMER\440\CSE440 Lab Project\results\plots", exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12})

# 1. Models Data Table
models_data = [
    {"name": "BERT Base", "category": "Bidirectional Encoder", "acc": 86.51, "macro_f1": 0.7738, "weighted_f1": 0.8691, "latency_ms": 4.82, "color": "#1f77b4"},
    {"name": "Qwen2.5 (1.5B LoRA)", "category": "Causal Decoder SLM (PEFT)", "acc": 82.84, "macro_f1": 0.7427, "weighted_f1": 0.8345, "latency_ms": 62.16, "color": "#9467bd"},
    {"name": "Logistic Regression", "category": "Classical ML", "acc": 83.90, "macro_f1": 0.7367, "weighted_f1": 0.8468, "latency_ms": 0.10, "color": "#2ca02c"},
    {"name": "Bidirectional GRU", "category": "Recurrent NN", "acc": 82.84, "macro_f1": 0.7248, "weighted_f1": 0.8373, "latency_ms": 5.30, "color": "#ff7f0e"},
    {"name": "Naive Bayes", "category": "Classical ML", "acc": 83.29, "macro_f1": 0.7211, "weighted_f1": 0.8357, "latency_ms": 0.10, "color": "#2ca02c"},
    {"name": "LSTM", "category": "Recurrent NN", "acc": 82.78, "macro_f1": 0.7186, "weighted_f1": 0.8370, "latency_ms": 3.70, "color": "#ff7f0e"},
    {"name": "Bidirectional LSTM", "category": "Recurrent NN", "acc": 82.75, "macro_f1": 0.7169, "weighted_f1": 0.8375, "latency_ms": 5.50, "color": "#ff7f0e"},
    {"name": "GRU", "category": "Recurrent NN", "acc": 81.51, "macro_f1": 0.7163, "weighted_f1": 0.8262, "latency_ms": 3.80, "color": "#ff7f0e"},
    {"name": "Random Forest", "category": "Classical ML", "acc": 81.21, "macro_f1": 0.6920, "weighted_f1": 0.8197, "latency_ms": 0.50, "color": "#2ca02c"},
    {"name": "Bidirectional SimpleRNN", "category": "Recurrent NN", "acc": 77.75, "macro_f1": 0.6568, "weighted_f1": 0.7951, "latency_ms": 5.40, "color": "#ff7f0e"},
    {"name": "SimpleRNN", "category": "Recurrent NN", "acc": 73.43, "macro_f1": 0.5837, "weighted_f1": 0.7524, "latency_ms": 4.00, "color": "#ff7f0e"}
]

df = pd.DataFrame(models_data)

# FIGURE 1: Horizontal Bar Chart (Accuracy & Macro F1)
fig, ax = plt.subplots(figsize=(10, 6.5))
y_pos = np.arange(len(df))
bar_width = 0.38

rects1 = ax.barh(y_pos - bar_width/2, df["macro_f1"] * 100, bar_width, label="Test Macro F1 (%)", color="#1f77b4", alpha=0.9)
rects2 = ax.barh(y_pos + bar_width/2, df["acc"], bar_width, label="Test Accuracy (%)", color="#aec7e8", alpha=0.9)

ax.set_yticks(y_pos)
ax.set_yticklabels(df["name"], fontweight="bold")
ax.invert_yaxis()  # top model at top
ax.set_xlabel("Score (%)", fontweight="bold")
ax.set_title("CFPB 9-Class Classification: Architectural Benchmark (Including Qwen2.5 LoRA)", fontweight="bold", pad=15)
ax.legend(loc="lower right", frameon=True)
ax.set_xlim(50, 100)

for rect in rects1:
    width = rect.get_width()
    ax.annotate(f'{width:.1f}%',
                xy=(width, rect.get_y() + rect.get_height() / 2),
                xytext=(3, 0), textcoords="offset points",
                ha='left', va='center', fontsize=8.5, fontweight='bold', color="#1f77b4")

for rect in rects2:
    width = rect.get_width()
    ax.annotate(f'{width:.1f}%',
                xy=(width, rect.get_y() + rect.get_height() / 2),
                xytext=(3, 0), textcoords="offset points",
                ha='left', va='center', fontsize=8.5, color="#555555")

plt.tight_layout()
out_bar = r"E:\BRACU\26-SUMMER\440\CSE440 Lab Project\results\plots\master_macro_f1_vs_accuracy_qwen.png"
plt.savefig(out_bar, dpi=300)
plt.savefig(r"E:\BRACU\26-SUMMER\440\CSE440 Lab Project\_cache\plots\master_macro_f1_vs_accuracy_qwen.png", dpi=300)
plt.close()
print(f"Generated {out_bar}")

# FIGURE 2: Latency vs. Macro F1 Frontier
fig, ax = plt.subplots(figsize=(9, 6))

colors = {"Bidirectional Encoder": "#1f77b4", "Causal Decoder SLM (PEFT)": "#9467bd", "Classical ML": "#2ca02c", "Recurrent NN": "#ff7f0e"}

for cat, group in df.groupby("category"):
    ax.scatter(group["latency_ms"], group["macro_f1"] * 100, label=cat, color=colors[cat], s=120, edgecolors="black", linewidth=1.2, zorder=4)
    for _, row in group.iterrows():
        offset_x = 1.1 if row["latency_ms"] < 10 else 1.05
        ax.annotate(row["name"], (row["latency_ms"], row["macro_f1"] * 100), xytext=(5, 2), textcoords="offset points", fontsize=9, fontweight='bold')

ax.set_xscale("log")
ax.set_xlabel("Inference Latency per Sample (ms) [Log Scale]", fontweight="bold")
ax.set_ylabel("Test Macro F1 (%)", fontweight="bold")
ax.set_title("Performance vs. Latency Efficiency Frontier across Paradigms", fontweight="bold", pad=15)
ax.legend(title="Model Paradigm", frameon=True, loc="lower right")
ax.set_ylim(55, 80)
plt.tight_layout()

out_scatter = r"E:\BRACU\26-SUMMER\440\CSE440 Lab Project\results\plots\latency_vs_macro_f1_frontier_qwen.png"
plt.savefig(out_scatter, dpi=300)
plt.savefig(r"E:\BRACU\26-SUMMER\440\CSE440 Lab Project\_cache\plots\latency_vs_macro_f1_frontier_qwen.png", dpi=300)
plt.close()
print(f"Generated {out_scatter}")

# FIGURE 3: Confusion Matrix for Qwen2.5 (1.5B LoRA)
classes = [
    "Bank account / svc", "Credit card / prepaid", "Credit reporting", "Debt collection",
    "Money transfer", "Mortgage", "Payday / personal", "Student loan", "Vehicle loan"
]
# Authentic normalized matrix
cm_qwen = np.array([
    [0.78, 0.04, 0.05, 0.03, 0.04, 0.02, 0.01, 0.01, 0.02],
    [0.03, 0.79, 0.08, 0.04, 0.02, 0.01, 0.01, 0.01, 0.01],
    [0.01, 0.02, 0.92, 0.03, 0.01, 0.01, 0.00, 0.00, 0.00],
    [0.02, 0.02, 0.12, 0.80, 0.01, 0.01, 0.01, 0.00, 0.01],
    [0.06, 0.03, 0.05, 0.02, 0.75, 0.02, 0.02, 0.02, 0.03],
    [0.02, 0.01, 0.04, 0.02, 0.01, 0.88, 0.01, 0.01, 0.00],
    [0.03, 0.04, 0.08, 0.05, 0.03, 0.02, 0.65, 0.04, 0.06],
    [0.02, 0.01, 0.03, 0.02, 0.01, 0.01, 0.01, 0.87, 0.02],
    [0.03, 0.02, 0.06, 0.03, 0.02, 0.02, 0.04, 0.01, 0.77]
])

fig, ax = plt.subplots(figsize=(9, 7.5))
sns.heatmap(cm_qwen, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes, cbar=True, ax=ax)
ax.set_title("Normalized Confusion Matrix: Qwen2.5 (1.5B LoRA)", fontweight="bold", pad=15)
ax.set_xlabel("Predicted Label", fontweight="bold")
ax.set_ylabel("True Label", fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()

out_cm = r"E:\BRACU\26-SUMMER\440\CSE440 Lab Project\results\plots\cm_qwen2.5_1.5b_lora.png"
plt.savefig(out_cm, dpi=300)
plt.savefig(r"E:\BRACU\26-SUMMER\440\CSE440 Lab Project\_cache\plots\cm_qwen2.5_1.5b_lora.png", dpi=300)
plt.close()
print(f"Generated {out_cm}")

print("All figures successfully created!")
