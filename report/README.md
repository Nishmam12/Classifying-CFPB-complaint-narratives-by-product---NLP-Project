# Building the ACL-format report

## What's here

| File | Purpose |
|---|---|
| `acl_report.tex` | The paper. ACL format, ~8 pages with figures. |
| `custom.bib` | 24 BibTeX entries, all cited in the paper. |
| `figures/` | 10 PNGs exported directly from the notebook outputs. |

## What you still need

The ACL style files are **not** included here (they're distributed by the ACL,
not something to vendor blindly). Download two files from
<https://github.com/acl-org/acl-style-files> and drop them beside `acl_report.tex`:

- `acl.sty`
- `acl_natbib.bst`

## Build

**Overleaf (easiest, no local install):** create a project, upload
`acl_report.tex`, `custom.bib`, the `figures/` folder, and the two ACL style
files. Set the compiler to pdfLaTeX. Overleaf runs BibTeX automatically.

**Locally**, with TeX Live or MiKTeX installed:

```bash
pdflatex acl_report
bibtex   acl_report
pdflatex acl_report
pdflatex acl_report
```

The two trailing `pdflatex` passes are needed to resolve citations and
cross-references — a single pass leaves `??` markers in the text.

## Regenerating the figures

The figures are exported straight from the executed notebook, so they always
match the reported numbers. If the notebook is re-run, refresh them with:

```bash
cd ../pipeline
python export_figs.py
```

`raw_class_distribution.png` is exported but not currently referenced by the
paper — it shows the 21 raw categories before consolidation, and is available
if you want it as an appendix figure.

## Before you submit

Two things in the paper need your attention rather than mine:

1. **Author block** — currently `Nabil / BRAC University`. Add co-authors,
   student IDs, or whatever your course requires.
2. **The Related Work positioning claim** (Section 2, final paragraph) says
   public CFPB implementations "we surveyed" use class weighting without
   comparison. It is deliberately hedged, but it is only honest if you have
   actually looked at some. Cite two or three specific Kaggle/GitHub
   implementations in `custom.bib` and reference them there, or delete the
   sentence. An examiner is likely to ask which ones.
