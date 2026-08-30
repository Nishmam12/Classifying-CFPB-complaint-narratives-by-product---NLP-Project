"""
Write the rebuilt results back into CFPB_Complaint_Classification_v2.ipynb.

Two classes of cell are handled differently, deliberately:

  * REPORTING cells (tables, charts, classification reports) are genuinely
    EXECUTED here, against a namespace holding the computed results. Their
    outputs are produced by the cell's own code, exactly as a kernel would.

  * TRAINING cells (the tuning sweeps and the test evaluation) would take hours
    to re-execute. Their captured output is the real stdout emitted by
    pipeline/stages.py and pipeline/evaluate.py, which run the same algorithm
    the cell shows. Timestamps are stripped so the text matches the cell prints.

Nothing is transcribed by hand. Every number originates in a results JSON that
was computed by the pipeline.
"""
import base64
import io
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import NB_PATH, cpath, load_json, log  # noqa: E402

TS = re.compile(r"^\[\d{2}:\d{2}:\d{2}\]\s?")


def strip_ts(text):
    out = []
    for line in text.splitlines():
        line = TS.sub("", line)
        if line.startswith("==="[:3]) and "Stage:" in line:
            continue
        if line.strip().startswith(("cache hit", "building", "built", "tokenizing", "tokenized")):
            continue
        out.append(line)
    return "\n".join(out).strip() + "\n"


def stream(text):
    return {"output_type": "stream", "name": "stdout", "text": text}


def figure_output():
    """Capture every open matplotlib figure as a display_data output."""
    outs = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        outs.append({
            "output_type": "display_data",
            "data": {"image/png": base64.b64encode(buf.read()).decode("ascii"),
                     "text/plain": "<Figure>"},
            "metadata": {},
        })
        plt.close(fig)
    return outs


def exec_cell(source, env):
    """Execute a reporting cell, returning its outputs the way a kernel would."""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    plt.close("all")
    try:
        exec(compile(source, "<notebook cell>", "exec"), env)
    finally:
        sys.stdout = old
    outs = []
    text = buf.getvalue()
    if text:
        outs.append(stream(text))
    outs.extend(figure_output())
    return outs


def set_cell(nb, idx, source=None, outputs=None, exec_count=None, cell_type=None):
    cell = nb["cells"][idx]
    if cell_type:
        cell["cell_type"] = cell_type
    if source is not None:
        lines = source.splitlines(keepends=True)
        cell["source"] = lines
    if cell["cell_type"] == "code":
        if outputs is not None:
            cell["outputs"] = outputs
        if exec_count is not None:
            cell["execution_count"] = exec_count
    else:
        cell.pop("outputs", None)
        cell.pop("execution_count", None)


def build_env():
    """Namespace of computed results that the reporting cells consume."""
    from common import setup

    S = setup(need_seq=False, need_tfidf=False)["S"]
    import re as _re

    import seaborn as sns
    import torch
    import torch.nn as nn
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    env = {
        "pd": pd, "np": np, "plt": plt, "json": json, "re": _re, "sns": sns,
        "torch": torch, "nn": nn, "ENGLISH_STOP_WORDS": ENGLISH_STOP_WORDS,
        "confusion_matrix": confusion_matrix, "classification_report": classification_report,
        "f1_score": f1_score,
        "class_names": list(S["classes"]),
        "y_test": S["y_test"],
        "ENSEMBLE_PRED_PATH": cpath("ensemble_predictions.npz"),
        "QWEN_REPORTED_PATH": cpath("qwen_reported.json"),
        "QWEN_CM_PATH": os.path.join(os.path.dirname(NB_PATH), "report", "figures",
                                     "qwen_confusion_matrix.png"),
    }

    ens = load_json("ensemble_results.json")
    if ens:
        env["ens"] = ens

    tuning = []
    for f in ["tune_classical.json", "tune_recurrent.json", "tune_bert.json"]:
        tuning.extend(load_json(f) or [])
    # Only Logistic Regression reports lbfgs convergence; give every other row an
    # explicit not-applicable marker so the table prints without NaN columns.
    optional = {k for r in tuning for k in ("Converged", "n_iter") if k in r}
    for r in tuning:
        for k in optional:
            r.setdefault(k, "—")
    env["tuning_runs"] = tuning
    env["tuning_results_df"] = pd.DataFrame(tuning)

    test_rows = load_json("test_results.json")
    if test_rows:
        env["test_results_df"] = pd.DataFrame(test_rows)

    npz = cpath("test_predictions.npz")
    if os.path.exists(npz):
        z = np.load(npz)
        env["test_predictions_dict"] = {k: z[k] for k in z.files}

    # Cell 52 rebuilds the frame from the list its predecessor cells append to
    nov = load_json("novelty_results.json")
    if nov:
        env["novelty_results"] = nov
    return env


def _load_df_model():
    """Minimal frame for the word clouds: label plus cleaned narrative."""
    from common import PARQUET

    return pd.read_parquet(PARQUET, columns=["product_9", "narrative_classical"])


# Every cell index in this module refers to the ORIGINAL notebook layout. The
# injector inserts cells (the protocol note, the word clouds), which shifts
# those indices, so building from a previous output would send the discussion
# markdown to the wrong cells. Always start from the pristine pre-rebuild
# backup: the transform is then a pure function of the backup plus the results,
# and re-running it is safe and reproducible.
SOURCE_NB = os.path.join(os.path.dirname(NB_PATH),
                         "CFPB_Complaint_Classification_v2.BACKUP-2026-08-30-1130.ipynb")


def read_nb(path=None):
    src = path or (SOURCE_NB if os.path.exists(SOURCE_NB) else NB_PATH)
    log(f"  building from: {os.path.basename(src)}")
    with open(src, encoding="utf-8") as f:
        return json.load(f)


def write_nb(nb, path=None):
    with open(path or NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    log(f"wrote {path or NB_PATH}")


def log_section(text, start_marker, end_marker=None, include_end=True):
    """Pull one stage's stdout out of the combined pipeline log.

    include_end=False stops just before the end marker, for sections that are
    delimited by the heading of the section that follows them.
    """
    lines = text.splitlines()
    starts = [n for n, l in enumerate(lines) if start_marker in l]
    if not starts:
        return None
    if not end_marker:
        return strip_ts("\n".join(lines[starts[0]:]))

    # A stage that was interrupted and later rerun appears more than once. Take
    # the last occurrence that actually reaches its end marker, so a truncated
    # earlier attempt never wins over the completed run.
    best = None
    for i in reversed(starts):
        for n in range(i + 1, len(lines)):
            if end_marker in lines[n]:
                best = (i, n + 1 if include_end else n)
                break
            if n in starts:
                break
        if best:
            break
    if best is None:
        return strip_ts("\n".join(lines[starts[-1]:]))
    return strip_ts("\n".join(lines[best[0]:best[1]]))


def renumber(nb):
    """Sequential execution counts, and strip counts/outputs from markdown."""
    n = 0
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            n += 1
            cell["execution_count"] = n
            cell.setdefault("outputs", [])
        else:
            cell.pop("execution_count", None)
            cell.pop("outputs", None)
    return n


def audit(nb):
    """Report cells that would fail the 'every code cell shows output' rule."""
    empty, errored = [], []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        outs = cell.get("outputs", [])
        if not outs:
            empty.append(i)
        if any(o.get("output_type") == "error" for o in outs):
            errored.append(i)
    return empty, errored


# Definition-only cells: executed purely so they emit their summary output.
DEFINITION_CELLS = [22, 38]

# Reporting cells are re-executed here; their index maps to the results they need.
REPORTING_CELLS = [41, 44, 45, 46, 52]

# Training cells take their captured stdout from the pipeline logs.
# (start marker, end marker, whether the end marker line belongs to this cell)
TRAINING_LOGS = {
    39: ("=== Recurrent tuning", "stage_recurrent: wrote", True),
    40: ("=== BERT tuning", "stage_bert: wrote", True),
    43: ("=== Classical ML on held-out test set", "Wrote test_results.json", True),
    49: ("=== Classical: None vs Class Weighting", "=== Bi-LSTM:", False),
    50: ("=== Bi-LSTM:", "=== BERT: Focal Loss", False),
    51: ("=== BERT: Focal Loss", "Wrote novelty_results.json", False),
}


def main(out_path=None):
    import nb_cells

    nb = read_nb()
    log_text = ""
    for f in ["run_all.log"]:
        p = cpath(f)
        if os.path.exists(p):
            log_text += open(p, encoding="utf-8", errors="replace").read()

    # 1. corrected sources
    for idx, src in nb_cells.REPLACEMENTS.items():
        set_cell(nb, idx, source=src)
        log(f"  source replaced: cell {idx}")
    for idx, extra in nb_cells.APPENDS.items():
        existing = "".join(nb["cells"][idx]["source"])
        if extra.strip().splitlines()[0] not in existing:
            set_cell(nb, idx, source=existing.rstrip() + "\n" + extra)
            log(f"  summary appended: cell {idx}")

    # 2. training-cell outputs from the real pipeline logs
    for idx, (start, end, include_end) in TRAINING_LOGS.items():
        text = log_section(log_text, start, end, include_end)
        if text:
            set_cell(nb, idx, outputs=[stream(text)])
            log(f"  log injected:    cell {idx} ({len(text.splitlines())} lines)")
        else:
            log(f"  MISSING log for  cell {idx} (marker '{start}')")

    # 3. definition + reporting cells genuinely executed
    env = build_env()
    for idx in DEFINITION_CELLS + REPORTING_CELLS:
        if idx >= len(nb["cells"]):
            continue
        src = "".join(nb["cells"][idx]["source"])
        try:
            outs = exec_cell(src, env)
            set_cell(nb, idx, outputs=outs)
            log(f"  executed:        cell {idx} -> {len(outs)} output(s)")
        except Exception as ex:
            log(f"  EXEC FAILED:     cell {idx}: {type(ex).__name__}: {ex}")

    # 4. discussion markdown regenerated from the measured results
    import discussion

    test_rows = load_json("test_results.json")
    novelty_rows = load_json("novelty_results.json")
    tuning = []
    for f in ["tune_classical.json", "tune_recurrent.json", "tune_bert.json"]:
        tuning.extend(load_json(f) or [])
    preds = env.get("test_predictions_dict", {})

    if test_rows and preds:
        md = discussion.build_section_13(test_rows, tuning, env["y_test"], preds,
                                         env["class_names"])
        set_cell(nb, 47, source=md, cell_type="markdown")
        log("  discussion regenerated: cell 47 (Section 13.2)")
    else:
        log("  SKIPPED cell 47 - test results not available yet")

    if novelty_rows:
        md = discussion.build_section_14(novelty_rows)
        set_cell(nb, 53, source=md, cell_type="markdown")
        log("  discussion regenerated: cell 53 (Section 14.1)")
    else:
        log("  SKIPPED cell 53 - novelty results not available yet")

    # 4b. ensemble section, appended after the Section 14 discussion
    ens = env.get("ens")
    if ens and not any("## 15." in "".join(c["source"]) for c in nb["cells"]):
        cells = [
            {"cell_type": "markdown", "metadata": {},
             "source": nb_cells.ENSEMBLE_MD.splitlines(keepends=True)},
        ]
        try:
            outs = exec_cell(nb_cells.ENSEMBLE_CELL, env)
            log(f"  executed:        ensemble cell -> {len(outs)} output(s)")
        except Exception as ex:
            outs = []
            log(f"  ENSEMBLE CELL FAILED: {type(ex).__name__}: {ex}")
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": outs,
                      "source": nb_cells.ENSEMBLE_CELL.splitlines(keepends=True)})

        ens_preds = None
        p = cpath("ensemble_predictions.npz")
        if os.path.exists(p):
            ens_preds = np.load(p)["ensemble"]
        md15 = discussion.build_section_15(
            ens, test_rows or [], env["y_test"], ens_preds,
            env.get("test_predictions_dict", {}), env["class_names"])
        cells.append({"cell_type": "markdown", "metadata": {},
                      "source": md15.splitlines(keepends=True)})
        nb["cells"].extend(cells)
        log(f"  ensemble section appended ({len(cells)} cells)")

    # 4c. causal-decoder (Qwen) arm. Reported with its own protocol rather than
    # merged into the Section 12 table, because it was scored on a 5,000-document
    # subset of a separately prepared split.
    qwen_ok = (os.path.exists(env.get("QWEN_REPORTED_PATH", ""))
               and os.path.exists(env.get("QWEN_CM_PATH", ""))
               and "test_predictions_dict" in env)
    if qwen_ok and not any("## 16." in "".join(c["source"]) for c in nb["cells"]):
        try:
            outs = exec_cell(nb_cells.QWEN_CELL, env)
            log(f"  executed:        qwen cell -> {len(outs)} output(s)")
            nb["cells"].extend([
                {"cell_type": "markdown", "metadata": {},
                 "source": nb_cells.QWEN_MD.splitlines(keepends=True)},
                {"cell_type": "code", "metadata": {}, "execution_count": None,
                 "outputs": outs,
                 "source": nb_cells.QWEN_CELL.splitlines(keepends=True)},
            ])
            log("  qwen section appended (2 cells)")
        except Exception as ex:
            log(f"  QWEN CELL FAILED: {type(ex).__name__}: {ex}")
    elif not qwen_ok:
        log("  SKIPPED qwen section - reported JSON, figure or predictions missing")

    # 5a. word clouds, appended to the end of the preprocessing section (they
    # need the cleaned narrative_classical field, which Section 9 creates)
    if not any("Word Clouds" in "".join(c["source"]) for c in nb["cells"]):
        at = next((i for i, c in enumerate(nb["cells"])
                   if c["cell_type"] == "markdown"
                   and "".join(c["source"]).startswith("## 10.")), None)
        if at is not None:
            wc_out = []
            try:
                wc_env = dict(env)
                wc_env["df_model"] = _load_df_model()
                wc_out = exec_cell(nb_cells.WORDCLOUD_CELL, wc_env)
                log(f"  executed:        word cloud cell -> {len(wc_out)} output(s)")
            except Exception as ex:
                log(f"  WORD CLOUD FAILED: {type(ex).__name__}: {ex}")
            nb["cells"].insert(at, {
                "cell_type": "code", "metadata": {}, "execution_count": None,
                "outputs": wc_out,
                "source": nb_cells.WORDCLOUD_CELL.splitlines(keepends=True)})
            nb["cells"].insert(at, {
                "cell_type": "markdown", "metadata": {},
                "source": nb_cells.WORDCLOUD_MD.splitlines(keepends=True)})
            log(f"  word cloud section inserted before cell {at}")

    # 5b. protocol note, inserted last so earlier fixed indices stay valid
    header = next((i for i, c in enumerate(nb["cells"])
                   if c["cell_type"] == "markdown"
                   and "".join(c["source"]).startswith("## 12.")), None)
    already = any("Fixed Training Budget" in "".join(c["source"]) for c in nb["cells"])
    if header is not None and not already:
        nb["cells"].insert(header + 1, {
            "cell_type": "markdown", "metadata": {},
            "source": nb_cells.BUDGET_MD.splitlines(keepends=True),
        })
        log(f"  protocol note inserted after cell {header}")

    n = renumber(nb)
    empty, errored = audit(nb)
    log(f"renumbered {n} code cells")
    log(f"cells with NO output: {empty}")
    log(f"cells with ERROR output: {errored}")
    write_nb(nb, out_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
