"""Export every notebook figure to report/figures/ for inclusion in the LaTeX paper."""
import base64
import json
import os
import re

from common import NB_PATH, log

OUT = os.path.join(os.path.dirname(NB_PATH), "report", "figures")

# Maps a distinctive phrase in the producing cell to the filename the paper uses.
NAMES = [
    ("Number of raw classes", "raw_class_distribution"),
    ("narrative_word_count", "narrative_length"),
    ("category_map", "class_distribution_9"),
    ("before vs after classical", "preprocessing_effect"),
    ("word clouds over the cleaned", "wordclouds"),
    ("class distributions across all three splits", "split_stratification"),
    ("unified Section 3.8 deliverable table", "tuning_overview"),
    ("horizontal bar chart", "model_comparison"),
    ("Normalized Confusion Matrices", "confusion_matrices"),
    ("Master Imbalance Strategy Comparison", "imbalance_comparison"),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    nb = json.load(open(NB_PATH, encoding="utf-8"))
    written, used = [], set()

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        figs = [o for o in cell.get("outputs", [])
                if o.get("output_type") == "display_data" and "image/png" in o.get("data", {})]
        if not figs:
            continue
        src = "".join(cell["source"])
        name = None
        for phrase, fname in NAMES:
            if phrase in src and fname not in used:
                name, _ = fname, used.add(fname)
                break
        if name is None:
            name = f"figure_cell{i:02d}"

        for n, fig in enumerate(figs):
            suffix = "" if len(figs) == 1 else f"_{n + 1}"
            path = os.path.join(OUT, f"{name}{suffix}.png")
            with open(path, "wb") as f:
                f.write(base64.b64decode(fig["data"]["image/png"]))
            kb = os.path.getsize(path) // 1024
            written.append((f"{name}{suffix}.png", i, kb))

    log(f"exported {len(written)} figures to report/figures/")
    for fname, cell_idx, kb in written:
        print(f"  {fname:34s} (cell {cell_idx:2d})  {kb:4d} KB")


if __name__ == "__main__":
    main()
