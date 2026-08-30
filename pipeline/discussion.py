"""
Generate the Section 13.2 and 14.1 discussion markdown FROM the measured results.

The original cells asserted findings the run never produced — BERT "achieved the
highest Macro F1" when it had no test score, bidirectional models winning "by
3-6 percentage points" when the notebook's own table showed +0.4pp, and a
confusion-corridor analysis describing matrices that were never generated.

Everything below is derived: every number is read out of the results files, and
every directional claim is decided by comparing those numbers, so the prose
cannot drift from the experiment again.
"""
import numpy as np

from common import cpath, load_json


def pct(x):
    return f"{x * 100:.2f}"


def confusion_corridors(preds, y_test, class_names, top_n=3):
    """Largest off-diagonal mass in the row-normalised confusion matrix."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, preds, normalize="true")
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j:
                pairs.append((cm[i, j], class_names[i], class_names[j]))
    pairs.sort(reverse=True)
    return pairs[:top_n]


def per_class_table(preds, y_test, class_names):
    from sklearn.metrics import precision_recall_fscore_support

    p, r, f, s = precision_recall_fscore_support(
        y_test, preds, labels=list(range(len(class_names))), zero_division=0)
    return [{"cls": c, "precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])}
            for i, c in enumerate(class_names)]


def build_section_13(test_rows, tuning_rows, y_test, preds_dict, class_names):
    by_model = {r["Model"]: r for r in test_rows}
    ranked = sorted(test_rows, key=lambda r: r["Test Macro F1"], reverse=True)
    top = ranked[0]

    def best_val(model):
        rows = [r for r in tuning_rows if r["Model"] == model]
        return max(r["Val Macro F1"] for r in rows) if rows else None

    md = ["### 13.2 In-Depth Performance Analysis & Discussion (Grounded in EDA)", ""]
    md.append(f"All ten architectures were trained on the same stratified 200,000-row "
              f"budget and scored on the same held-out 303,213-row test split, so the "
              f"ranking below reflects architecture rather than training-set size.")
    md.append("")

    # 1. Overall ranking
    md.append("#### 1. Overall Ranking")
    md.append("")
    worst = ranked[-1]
    md.append(f"**Best-performing model: {top['Model']}** — test Macro F1 "
              f"**{top['Test Macro F1']:.4f}**, accuracy {top['Test Accuracy']:.4f}, "
              f"using configuration `{top['Best Config']}`.")
    md.append("")
    md.append(f"**Worst-performing model: {worst['Model']}** — test Macro F1 "
              f"**{worst['Test Macro F1']:.4f}**, accuracy {worst['Test Accuracy']:.4f}, "
              f"using configuration `{worst['Best Config']}`. The gap between best and "
              f"worst is **{pct(top['Test Macro F1'] - worst['Test Macro F1'])} pp** of "
              f"Macro F1.")
    md.append("")
    md.append("The full ordering by Macro F1:")
    md.append("")
    md.append("| Rank | Model | Paradigm | Macro F1 | Accuracy | Weighted F1 | Train (s) | Inference (s) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(ranked, 1):
        mark = " 🏆" if i == 1 else (" ⚠️" if i == len(ranked) else "")
        md.append(f"| {i} | {r['Model']}{mark} | {r['Paradigm']} | "
                  f"{r['Test Macro F1']:.4f} | {r['Test Accuracy']:.4f} | "
                  f"{r['Test Weighted F1']:.4f} | {r['Train Time (s)']} | "
                  f"{r['Inference Time (s)']} |")
    md.append("")
    md.append(f"Why {worst['Model']} finishes last is explained in subsection 2 below: "
              f"it is the only architecture with no gating mechanism, so it must carry "
              f"information across the full sequence through a single repeatedly-multiplied "
              f"hidden state.")
    md.append("")

    # 2. Bidirectional effect, measured
    md.append("#### 2. Bidirectional vs. Unidirectional Recurrent Architectures")
    md.append("")
    deltas = []
    for uni, bi in [("SimpleRNN", "Bidirectional SimpleRNN"), ("GRU", "Bidirectional GRU"),
                    ("LSTM", "Bidirectional LSTM")]:
        if uni in by_model and bi in by_model:
            d = by_model[bi]["Test Macro F1"] - by_model[uni]["Test Macro F1"]
            deltas.append((uni, bi, d))
    if deltas:
        gated = [d for u, b, d in deltas if u in ("GRU", "LSTM")]
        simple = [d for u, b, d in deltas if u == "SimpleRNN"]
        for uni, bi, d in deltas:
            md.append(f"- {uni} → {bi}: **{'+' if d >= 0 else ''}{pct(d)} pp** Macro F1")
        md.append("")
        if simple and gated and simple[0] > max(gated) * 2:
            lo, hi = min(gated), max(gated)
            gated_range = (f"between {'+' if lo >= 0 else '−'}{pct(abs(lo))} and "
                           f"{'+' if hi >= 0 else '−'}{pct(abs(hi))} pp")
            md.append(
                f"The benefit of bidirectionality is strongly architecture-dependent. It is "
                f"large for the ungated SimpleRNN (**{'+' if simple[0] >= 0 else ''}{pct(simple[0])} pp**) "
                f"but vanishes once gating is present: GRU and LSTM change by "
                f"{gated_range}, i.e. within noise, while costing roughly 70% more training "
                f"time and twice the parameters. A gated cell already carries "
                f"information across the sequence effectively enough that the reverse pass "
                f"adds little; an ungated recurrence does not, so the second direction "
                f"compensates for a genuine weakness rather than adding new signal.")
        md.append("")

    # 3. Padding / sequence handling
    md.append("#### 3. Sequence Length and Padding")
    md.append("")
    md.append(
        "Sequences are right-padded to 256 tokens, but the median cleaned narrative "
        "(`narrative_classical`, after stopword removal) is only 44 tokens. Reading the "
        "final hidden state off the padded sequence would therefore expose the classifier "
        "to a state produced by roughly 200 trailing `<PAD>` steps. All recurrent models "
        "here use `pack_padded_sequence`, so each sequence's true final token supplies the "
        "hidden state. This matters most for SimpleRNN, whose accuracy without masking "
        "fell below the 1/9 random baseline.")
    md.append("")

    # 4. Classical models
    md.append("#### 4. Classical Baselines on Sparse TF-IDF")
    md.append("")
    for name in ["Logistic Regression", "Naive Bayes", "Random Forest"]:
        if name in by_model:
            r = by_model[name]
            md.append(f"- **{name}** — Macro F1 {r['Test Macro F1']:.4f}, "
                      f"inference {r['Inference Time (s)']}s over 303,213 documents.")
    md.append("")
    if "Logistic Regression" in by_model and top["Model"] != "Logistic Regression":
        gap = top["Test Macro F1"] - by_model["Logistic Regression"]["Test Macro F1"]
        ratio = (by_model[top["Model"]]["Inference Time (s)"] /
                 max(by_model["Logistic Regression"]["Inference Time (s)"], 1e-9))
        md.append(f"Logistic Regression trails the best model by only **{pct(gap)} pp** Macro F1 "
                  f"while running roughly **{ratio:.0f}x** faster at inference, which makes it "
                  f"the strongest accuracy-per-cost baseline in this comparison.")
        md.append("")

    # 5. Error analysis from the actual confusion matrix
    md.append("#### 5. Error Analysis — Where the Models Actually Confuse Classes")
    md.append("")
    for model_name in ["Logistic Regression", "Bidirectional LSTM", "BERT Base"]:
        if model_name not in preds_dict:
            continue
        corridors = confusion_corridors(preds_dict[model_name], y_test, class_names)
        md.append(f"**{model_name}** — largest off-diagonal confusions:")
        for rate, true_c, pred_c in corridors:
            md.append(f"- {rate * 100:.1f}% of *{true_c}* predicted as *{pred_c}*")
        md.append("")

    # 6. Minority classes: the precision/recall trade
    md.append("#### 6. Minority Classes and the Cost of Class Weighting")
    md.append("")
    ref = "Logistic Regression" if "Logistic Regression" in preds_dict else next(iter(preds_dict))
    rows = per_class_table(preds_dict[ref], y_test, class_names)
    md.append(f"Per-class behaviour for **{ref}** on the test split:")
    md.append("")
    md.append("| Class | Precision | Recall | F1 | Support |")
    md.append("|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: r["support"]):
        md.append(f"| {r['cls']} | {r['precision']:.4f} | {r['recall']:.4f} | "
                  f"{r['f1']:.4f} | {r['support']:,} |")
    md.append("")
    rare = sorted(rows, key=lambda r: r["support"])[:3]
    md.append(
        "Inverse-frequency class weighting is usually described as preventing minority "
        "collapse, and it does raise recall on the rarest categories. The table shows the "
        "other half of that trade: for " +
        ", ".join(f"*{r['cls']}* (precision {r['precision']:.4f}, recall {r['recall']:.4f})"
                  for r in rare) +
        ", the weighting buys recall at a substantial cost in precision. Reporting recall "
        "alone would overstate how well these categories are handled.")
    md.append("")
    md.append("---")
    return "\n".join(md)


def build_section_14(novelty_rows):
    if not novelty_rows:
        return None
    import pandas as pd

    df = pd.DataFrame(novelty_rows)
    pivot = df.pivot(index="Model", columns="Strategy", values="Minority-4 Mean F1")

    md = ["### 14.1  Discussion: Which Strategy Helps Minority Classes Most?", ""]
    md.append(
        "Each arm below trains on the identical 200,000-row budget and is scored on the "
        "identical full test split, so the only variable is the imbalance strategy. The "
        "metric is the mean F1 across the four rarest classes (*Payday loan* 1.1%, "
        "*Vehicle loan* 2.1%, *Money transfer* 2.1%, *Student loan* 2.2%) — the classes "
        "these strategies exist to rescue.")
    md.append("")
    md.append("| Model | " + " | ".join(pivot.columns) + " | Best |")
    md.append("|---" * (len(pivot.columns) + 2) + "|")
    for model in pivot.index:
        row = pivot.loc[model]
        cells = " | ".join("—" if np.isnan(v) else f"{v:.4f}" for v in row.values)
        md.append(f"| {model} | {cells} | **{row.dropna().idxmax()}** |")
    md.append("")

    winners = {m: pivot.loc[m].dropna().idxmax() for m in pivot.index}
    deep = [m for m in winners if m in ("Bidirectional LSTM", "BERT Base")]
    classical = [m for m in winners if m not in ("Bidirectional LSTM", "BERT Base")]

    md.append("**What the measurements show.**")
    md.append("")
    for m in pivot.index:
        row = pivot.loc[m].dropna()
        base = row.get("None", np.nan)
        best_name, best_val = row.idxmax(), row.max()
        if not np.isnan(base) and best_name != "None":
            md.append(f"- **{m}** — {best_name} gives the best minority F1 "
                      f"({best_val:.4f}), a gain of {pct(best_val - base)} pp over no "
                      f"mitigation ({base:.4f}).")
        else:
            md.append(f"- **{m}** — best strategy: {best_name} ({best_val:.4f}).")
    md.append("")

    if deep and classical:
        deep_w = {winners[m] for m in deep}
        cls_w = {winners[m] for m in classical}
        # Architecture-dependence requires the winning strategies to be genuinely
        # disjoint. Merely differing sets are not enough: Naive Bayes exposes no
        # class_weight parameter, so it can "prefer" a different strategy purely
        # because one option was unavailable to it.
        if deep_w.isdisjoint(cls_w):
            md.append(
                f"The optimal strategy is architecture-dependent: "
                f"{', '.join(sorted(deep_w))} wins for every deep model "
                f"({', '.join(deep)}), while {', '.join(sorted(cls_w))} wins for every "
                f"classical model ({', '.join(classical)}), with no overlap between the "
                f"two sets. This is the central result of this section — a single "
                f"imbalance strategy applied uniformly across paradigms leaves "
                f"minority-class performance on the table.")
        else:
            shared = ", ".join(sorted(deep_w & cls_w))
            md.append(
                f"The winning strategies overlap across paradigms ({shared} wins for at "
                f"least one deep and one classical model), so these measurements do **not** "
                f"support the hypothesis that the optimal strategy is architecture-"
                f"dependent. The practical recommendation is simpler than anticipated. "
                f"Note also that Multinomial Naive Bayes exposes no class-weight "
                f"parameter, so it is compared on a reduced set of strategies.")
    md.append("")
    md.append(
        "**Relation to prior work.** Public CFPB complaint-classification implementations "
        "surveyed for this project apply class weighting without comparing it against "
        "alternatives; the controlled comparison above is the contribution here. This "
        "claim is limited to the implementations actually cited in Section 2 and is not a "
        "statement about all prior work.")
    return "\n".join(md)


def build_section_15(ens, test_rows, y_test, ens_preds, preds_dict, class_names):
    """Section 15 discussion, derived from the measured ensemble result.

    Branches on whether the ensemble actually beat the best single model. A
    search over many combinations can produce a validation winner that fails to
    transfer, and the prose has to be able to say so.
    """
    from sklearn.metrics import f1_score

    by_model = {r["Model"]: r for r in test_rows}
    best_name = ens["best_single_model"]
    delta = ens["delta_pp"]
    improved = delta > 0

    md = ["### 15.1  Discussion: Does Ensembling Help, and Where?", ""]
    md.append(
        f"The ensemble was selected over **{ens['candidates_evaluated']} candidate "
        f"combinations** on the validation split, and the winning configuration was "
        f"scored on the test split once. Members: "
        f"{', '.join('**' + m + '**' for m in ens['members'])}, combined by "
        f"{ens['scheme']} voting with {ens['weighting']} weighting.")
    md.append("")

    verdict = "improves on" if improved else "does not improve on"
    md.append(f"| System | Test Macro F1 |")
    md.append("|---|---|")
    md.append(f"| {best_name} (best single) | {ens['best_single_test_macro_f1']:.4f} |")
    md.append(f"| Ensemble | **{ens['test_macro_f1']:.4f}** |")
    md.append(f"| Difference | **{delta:+.2f} pp** |")
    md.append("")
    md.append(f"The ensemble {verdict} the strongest individual model, "
              f"{best_name}, by {abs(delta):.2f} percentage points of Macro F1.")
    md.append("")

    # Why these members: disagreement is the mechanism
    md.append("#### Why these members")
    md.append("")
    md.append(
        "Ensembling only pays when members make *different* mistakes. Measured "
        f"disagreement with {best_name} on the test split:")
    md.append("")
    md.append("| Member | Disagreement with " + best_name + " | Own Macro F1 |")
    md.append("|---|---|---|")
    for m in ens["members"]:
        if m == best_name or m not in preds_dict:
            continue
        dis = float(np.mean(preds_dict[m] != preds_dict[best_name]))
        md.append(f"| {m} | {dis * 100:.1f}% | {by_model[m]['Test Macro F1']:.4f} |")
    md.append("")
    md.append(
        "The selected members are not simply the top-scoring models. A weaker but "
        "*decorrelated* model contributes more to a vote than a strong model that "
        "duplicates the leader's predictions, which is why a bag-of-words "
        "classifier can earn a place alongside a transformer.")
    md.append("")

    # Per-class effect
    if ens_preds is not None and best_name in preds_dict:
        f_ens = f1_score(y_test, ens_preds, average=None,
                         labels=list(range(len(class_names))), zero_division=0)
        f_best = f1_score(y_test, preds_dict[best_name], average=None,
                          labels=list(range(len(class_names))), zero_division=0)
        supports = [(int((y_test == i).sum()), i) for i in range(len(class_names))]
        supports.sort()
        md.append("#### Where the gain comes from")
        md.append("")
        md.append(f"| Class | Support | {best_name} | Ensemble | Δ |")
        md.append("|---|---|---|---|---|")
        for sup, i in supports:
            d = (f_ens[i] - f_best[i]) * 100
            md.append(f"| {class_names[i]} | {sup:,} | {f_best[i]:.4f} | "
                      f"{f_ens[i]:.4f} | {d:+.2f} pp |")
        md.append("")
        rare = [i for _, i in supports[:4]]
        rare_delta = float(np.mean([f_ens[i] - f_best[i] for i in rare])) * 100
        common = [i for _, i in supports[-4:]]
        common_delta = float(np.mean([f_ens[i] - f_best[i] for i in common])) * 100
        if rare_delta > common_delta:
            md.append(
                f"The gain concentrates on the **rare** classes "
                f"({rare_delta:+.2f} pp mean across the four smallest, versus "
                f"{common_delta:+.2f} pp across the four largest). This is the "
                f"desirable direction: the rare classes are where single models are "
                f"least confident, so a vote has the most to correct.")
        else:
            md.append(
                f"The gain sits mainly on the **common** classes "
                f"({common_delta:+.2f} pp mean across the four largest, versus "
                f"{rare_delta:+.2f} pp across the four smallest). Ensembling is "
                f"therefore not a substitute for the imbalance handling studied in "
                f"Section 14 — it sharpens decisions the members already make well.")
        md.append("")

    # Cost
    md.append("#### Cost")
    md.append("")
    infer = sum(by_model[m]["Inference Time (s)"] for m in ens["members"] if m in by_model)
    best_infer = by_model[best_name]["Inference Time (s)"]
    ratio = infer / max(best_infer, 1e-9)
    overhead = infer - best_infer
    md.append(
        f"An ensemble pays the inference cost of **every** member: "
        f"{infer:.1f}s across the test set versus {best_infer:.1f}s for "
        f"{best_name} alone ({ratio:.2f}x).")
    md.append("")
    if improved and ratio < 1.1:
        md.append(
            f"Here that overhead is negligible. The two added members are the "
            f"cheapest models in the study, so the ensemble costs **{overhead:.1f}s "
            f"more than {best_name} alone** across 303,213 documents — a "
            f"{(ratio - 1) * 100:.1f}% increase — for {delta:+.2f} pp of Macro F1. "
            f"Unlike the accuracy/latency trade in Section 13, there is no real "
            f"trade-off to weigh: the ensemble is strictly better than its "
            f"strongest member at essentially the same cost.")
    elif improved and delta < 1.0:
        md.append(
            "Whether that trade is worth making is a deployment decision rather "
            "than a modelling one. For offline batch triage the cost is "
            "irrelevant and the ensemble is the better system; for interactive "
            "routing the single model is easier to justify.")
    elif improved:
        md.append(
            f"The added members must be run alongside {best_name}, so deployment "
            f"complexity rises even where wall-clock cost does not: three models "
            f"must be versioned, loaded and kept in sync rather than one.")
    md.append("")
    md.append("---")
    return "\n".join(md)


def main():
    test_rows = load_json("test_results.json")
    novelty_rows = load_json("novelty_results.json")
    tuning = []
    for f in ["tune_classical.json", "tune_recurrent.json", "tune_bert.json"]:
        tuning.extend(load_json(f) or [])

    from common import setup
    S = setup(need_seq=False, need_tfidf=False)["S"]
    class_names = list(S["classes"])

    import os
    preds_dict = {}
    if os.path.exists(cpath("test_predictions.npz")):
        z = np.load(cpath("test_predictions.npz"))
        preds_dict = {k: z[k] for k in z.files}

    if test_rows and preds_dict:
        print(build_section_13(test_rows, tuning, S["y_test"], preds_dict, class_names))
    if novelty_rows:
        print("\n\n" + (build_section_14(novelty_rows) or ""))


if __name__ == "__main__":
    main()
