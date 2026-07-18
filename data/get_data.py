"""Download and cache the datasets used by the curriculum.

Nothing large is committed to the repository; this script fetches each
dataset on first use and caches it under ``data/``. Kaggle sources are
retrieved with ``kagglehub``, which requires a Kaggle account and an API
token in ``~/.kaggle/kaggle.json`` (see kaggle.com/settings). Notebooks
never call this module directly: they load through the helpers in
``src/data.py`` so that path handling lives in one place.

Datasets (introduced from module 01 onwards):

- Lending Club loans (Kaggle) — default classification and interest-rate
  regression, Blocks A–C.
- ULB credit card fraud (Kaggle) — class imbalance, anomaly detection,
  temporal splits.
- Capstone (module 12): Home Credit Default Risk or Freddie Mac
  Single-Family Loan-Level, decided before commissioning.

Module 00 uses no external data; loaders are added here as the modules
that need them are authored.
"""

if __name__ == "__main__":
    print(__doc__)
