"""Cached data loaders used by all notebooks.

Path handling and acquisition live here (and in ``data/get_data.py``);
notebooks call these helpers and never touch file paths themselves.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "data"


def load_lending_club(
    n_rows: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Load the Lending Club accepted-loans table (2007-2018Q4).

    Downloads and caches on first call (see ``data/get_data.py``; the
    one-off download is ~370 MB). Returns the full ~2.26M-row table with
    the standard column subset, or a deterministic random subsample of
    ``n_rows`` when requested — pass a ``seed`` so the subsample is
    reproducible across sessions.
    """
    if str(_DATA_DIR) not in sys.path:
        sys.path.insert(0, str(_DATA_DIR))
    from get_data import fetch_lending_club

    df = pd.read_parquet(fetch_lending_club())
    if n_rows is not None and n_rows < len(df):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df), size=n_rows, replace=False)
        df = df.iloc[np.sort(idx)].reset_index(drop=True)
    return df
