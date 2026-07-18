# CLAUDE.md — authoring conventions for this repository

This repository is a machine-learning-with-Python curriculum delivered as Jupyter notebooks. `SYLLABUS.md` is the source of truth for content. This file governs *how* modules are written. Read both before authoring anything; read only the syllabus section for the module being commissioned, not the whole document.

## Workflow

- **One module per session.** Author exactly the commissioned module and its solutions notebook. Do not begin the next module.
- **Author in jupytext, not raw ipynb.** Write each notebook as a `py:percent` script, then convert and execute:
  `jupytext --to ipynb --execute notebooks/NN-title.py`
  This is both token-efficient and diff-friendly. Commit the `.py`, the executed `.ipynb` (outputs retained — they render on GitHub), and the solutions pair.
- **Must run clean.** Full top-to-bottom execution in a fresh kernel with no errors is the gate for committing. Target under ~5 minutes runtime per notebook (capstone and NN modules may take up to ~15 with cached data).
- **Register lock.** Notebook 01 is the register exemplar once reviewed. For modules 02+, match the prose register of `notebooks/01-*.ipynb`.
- **Commit convention.** One commit per module: `module NN: <title>` plus a body listing any deviations from the syllabus spec. Deviations are permitted only with a stated reason.

## Notebook structure

Every notebook, in order:

1. **Header cell** — module number and title, learning objectives (3–5), prerequisites (which prior modules and which assumed background), estimated working time, and the module seed constant.
2. **Sections** of prose exposition. Full paragraphs. No bullet-point lecturing, no slide-deck prose. Bullets are permitted only for genuine enumerations (e.g. a taxonomy).
3. **Mathematics** in LaTeX. Number equations that are referenced later. Everything marked **[derive]** in the syllabus gets a genuine derivation with steps, not an assertion with a citation. Everything *not* marked may be stated with a reference.
4. **Code cells** short and single-purpose. No hidden state across distant cells; a reader executing top-to-bottom must never be surprised. Seed all randomness via `rng = np.random.default_rng(MODULE_SEED)` — never the legacy global API.
5. **Simulation before real data**, per the syllabus design principles. The simulation section states what claim is being tested and whether the evidence supported it.
6. **Exercises** at section ends, tiered: **A** (comprehension), **B** (derivation/extension), **C** (investigation, usually a small simulation study). No solutions inline.
7. **Further reading** — the syllabus citations plus at most two additions.

Solutions notebooks (`solutions/NN-solutions.ipynb`, same jupytext pairing) restate each exercise, then give full workings: derivations written out, code solutions executed, investigation exercises with brief written conclusions.

## Style

- British English throughout.
- Graduate register. No filler enthusiasm, no "simply", no "it's easy to see". If it were easy to see, derive it in one line.
- Do not re-teach assumed background (GLMs, likelihood, Bayes/shrinkage, survival, Brier/Murphy, SPC, R). Reference and connect instead.
- Epistemic tags where the literature is unsettled: prefix the paragraph with **Contested:** or **Speculative:** (the syllabus marks the known cases — double descent, SHAP value semantics, conditional conformal coverage).
- R comparisons are permitted in module 00 only. Thereafter the course is Python-native.
- Plots go through `src/tunnel_mpl.py` if present (`from src.tunnel_mpl import apply_theme`); otherwise restrained matplotlib defaults. Every figure has axis labels and a one-sentence caption in the following markdown cell.

## Repository layout

```
.
├── CLAUDE.md
├── SYLLABUS.md
├── pyproject.toml            # uv-managed; add deps via `uv add`
├── jupytext.toml             # pairing: ipynb <-> py:percent
├── notebooks/
│   ├── 00-python-for-r-users.py / .ipynb
│   └── ...
├── solutions/
│   ├── 00-solutions.py / .ipynb
│   └── ...
├── src/
│   ├── data.py               # cached loaders used by all notebooks
│   └── tunnel_mpl.py         # plot theme (optional)
├── data/
│   ├── get_data.py           # kagglehub / documented downloads; nothing large committed
│   └── .gitignore            # ignores everything but get_data.py
└── .github/workflows/
    └── execute.yml           # optional: smoke-execute changed notebooks on push
```

## Libraries

numpy, pandas, matplotlib, scipy, scikit-learn, statsmodels, xgboost, lightgbm; torch from module 11. Add anything else via `uv add` with a one-line justification in the commit body.

## Data

Nothing large is committed. `data/get_data.py` downloads and caches (kagglehub for Kaggle sources; document the auth requirement in its docstring). Notebooks load exclusively through `src/data.py` helpers so path handling lives in one place.

## Definition of done (per module)

- [ ] Executes clean, fresh kernel, within runtime target
- [ ] Content matches the syllabus section; all **[derive]** items genuinely derived
- [ ] Simulation section tests a stated claim against a known DGP
- [ ] Exercises tiered A/B/C; solutions notebook complete, executed, with workings
- [ ] Register matches notebook 01; British English; epistemic tags where specified
- [ ] jupytext pair, executed ipynb, and solutions pair committed in one commit

## Commissioning template

To commission a module, the prompt is:

> Author module NN following CLAUDE.md and SYLLABUS.md §NN. Match the register of notebooks/01. Verify clean execution, then commit per the convention.
