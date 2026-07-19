"""Download and cache the datasets used by the curriculum.

Nothing large is committed to the repository; this script fetches each
dataset on first use and caches it under ``data/``. Kaggle sources are
retrieved with ``kagglehub``. Public datasets download anonymously — no
account is needed; if you hit a rate limit or a private resource, place a
Kaggle API token in ``~/.kaggle/kaggle.json`` (kaggle.com → Settings →
Create New Token). Notebooks never call this module directly: they load
through the helpers in ``src/data.py`` so that path handling lives in
one place.

Datasets:

- **Lending Club accepted loans** (Kaggle, ``wordsforthewise/lending-club``,
  2007–2018Q4, ~2.26M rows) — running example for default classification
  and interest-rate regression, Blocks A–C. ``fetch_lending_club`` keeps
  the modelling-relevant columns and caches them as
  ``data/lending_club.parquet`` (~150 MB on disk; the one-off download is
  ~370 MB and parsing takes a few minutes).
- **ULB credit card fraud** (Kaggle) — added when module 03 is authored.
- **Capstone** (module 12): Home Credit Default Risk or Freddie Mac
  Single-Family Loan-Level, decided before commissioning.
"""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
LENDING_CLUB_PARQUET = DATA_DIR / "lending_club.parquet"

# The column subset used across Blocks A-C: loan terms, borrower
# characteristics known at origination, and the outcome fields.
LENDING_CLUB_COLUMNS = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "issue_d", "loan_status", "purpose", "dti", "delinq_2yrs",
    "earliest_cr_line", "fico_range_low", "fico_range_high", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc", "addr_state",
]


def fetch_lending_club(force: bool = False) -> Path:
    """Download the accepted-loans file and cache the column subset as parquet.

    Idempotent: returns immediately if the parquet cache exists (unless
    ``force``). The raw ``.csv.gz`` stays in kagglehub's own cache and is
    read once.
    """
    if LENDING_CLUB_PARQUET.exists() and not force:
        return LENDING_CLUB_PARQUET

    import kagglehub
    import pandas as pd

    raw = kagglehub.dataset_download(
        "wordsforthewise/lending-club", path="accepted_2007_to_2018Q4.csv.gz"
    )
    df = pd.read_csv(raw, usecols=LENDING_CLUB_COLUMNS, low_memory=False)
    # Rows with no loan_amnt are the file's trailing summary artefacts.
    df = df.dropna(subset=["loan_amnt"])
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")
    df["term_months"] = df["term"].str.strip().str.slice(0, 2).astype("int16")
    df = df.drop(columns=["term"])
    df.to_parquet(LENDING_CLUB_PARQUET, index=False)
    return LENDING_CLUB_PARQUET


if __name__ == "__main__":
    print(f"Lending Club cache: {fetch_lending_club()}")
