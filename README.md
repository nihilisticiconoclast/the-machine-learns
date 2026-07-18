# the-machine-learns

A statistically rigorous machine-learning-with-Python curriculum for a finance-focused analyst, delivered as a series of Jupyter notebooks. Written for a reader who is R-fluent with graduate-level statistics: every method is introduced as an estimator — loss function, assumptions, properties, failure modes — and demonstrated against a synthetic data-generating process before it touches real data. The terminal orientation is model risk and independent validation (PRA SS1/23), not leaderboards.

## Contents

Thirteen modules in four blocks. [`SYLLABUS.md`](SYLLABUS.md) is the source of truth for content; [`CLAUDE.md`](CLAUDE.md) governs how modules are authored.

| Block | Modules | Status |
|---|---|---|
| **A — Foundations** | 00 Python for the R-native statistician · 01 The statistical learning frame · 02 Linear models as estimators | 00 ✅ |
| **B — Supervised learning and honest evaluation** | 03 Classification, probability, and calibration · 04 Evaluation protocols · 05 Trees and bagging · 06 Gradient boosting | — |
| **C — The validator's toolkit** | 07 Interpretability and model validation · 08 Uncertainty quantification · 09 Unsupervised structure, with skepticism · 10 Temporal ML and drift | — |
| **D — Depth and synthesis** | 11 Neural networks: foundations · 12 Capstone: champion–challenger with a validation pack | — |

Each notebook contains prose exposition, derivations for everything the syllabus marks **[derive]**, a simulation study against a known DGP, a real-data application, tiered exercises (A comprehension / B derivation / C investigation), and further reading. Solutions live in parallel notebooks under [`solutions/`](solutions/).

## Getting started

The project is managed with [uv](https://docs.astral.sh/uv/):

```sh
uv sync                 # create .venv with all dependencies
uv run jupyter lab      # open the notebooks
```

Notebooks are authored as jupytext `py:percent` scripts and committed as pairs — the `.py` for diffs, the executed `.ipynb` so outputs render on GitHub. To re-execute a module from scratch:

```sh
uv run jupytext --to ipynb --execute notebooks/00-python-for-r-users.py
```

## Layout

```
├── SYLLABUS.md        # content spec, source of truth
├── CLAUDE.md          # authoring conventions
├── notebooks/         # one jupytext pair per module
├── solutions/         # parallel solutions notebooks, full workings
├── src/               # shared helpers (plot theme, data loaders)
└── data/              # local cache only; get_data.py fetches, nothing large committed
```

## Data

No data is committed. `data/get_data.py` documents acquisition (kagglehub for the Kaggle sources); notebooks load through `src/data.py` helpers. Module 00 is self-contained and needs no external data.
