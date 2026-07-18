# %% [markdown]
# # Module 00 · Solutions
#
# Full workings for the exercises in `notebooks/00-python-for-r-users.ipynb`. Each exercise is restated before its solution. The synthetic loan book is regenerated here from the same data-generating process and module seed; because this notebook's draw sequence differs from the teaching notebook's, individual values differ while every conclusion is unchanged.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_SEED = 0
rng = np.random.default_rng(MODULE_SEED)

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tunnel_mpl import apply_theme

apply_theme()

# %%
n = 200
grade = rng.choice(list("ABCDE"), size=n, p=[0.30, 0.30, 0.20, 0.15, 0.05])
base_rate = pd.Series({"A": 6.0, "B": 9.0, "C": 12.0, "D": 16.0, "E": 21.0})
loans = pd.DataFrame(
    {
        "loan_id": [f"L{i:04d}" for i in range(n)],
        "grade": grade,
        "purpose": rng.choice(["car", "house", "consolidation"], size=n),
        "term": rng.choice([36, 60], size=n),
        "loan_amnt": rng.integers(1_000, 40_001, size=n),
        "int_rate": base_rate[grade].to_numpy() + rng.normal(0.0, 1.0, size=n).round(2),
    }
).set_index("loan_id")

# %% [markdown]
# ## Exercise 2.1 (A — views and copies)
#
# *For `x = np.arange(12).reshape(3, 4)`, predict for each of the following whether it returns a view or a copy, then verify with `np.shares_memory`: (a) `x[1]`; (b) `x[:, ::2]`; (c) `x[x > 5]`; (d) `x[[0, 2]]`; (e) `x.T`; (f) `x.reshape(4, 3)`. For the views, state what a write through the result does to `x`.*
#
# **Solution.** The rule from §2.2: anything expressible as new strides over the same buffer is a view — basic slicing, transposition, and reshaping of contiguous data. Anything requiring element gathering is a copy — integer-array and boolean indexing.
#
# - (a) `x[1]` — **view**. A row is a basic slice; writing to it overwrites row 1 of `x`.
# - (b) `x[:, ::2]` — **view**. Strided slicing in both axes; writes land in columns 0 and 2 of `x`.
# - (c) `x[x > 5]` — **copy**. Boolean indexing gathers an unpredictable subset; writes to it never reach `x`.
# - (d) `x[[0, 2]]` — **copy**. Integer-array (advanced) indexing always copies.
# - (e) `x.T` — **view**. The transpose swaps shape and strides over the same buffer; writes appear in `x` at transposed positions.
# - (f) `x.reshape(4, 3)` — **view**, because `x` is C-contiguous, so the same buffer reread with new shape suffices. (On non-contiguous data, `reshape` silently copies — which is why code that *relies* on reshape aliasing is fragile.)

# %%
x = np.arange(12).reshape(3, 4)
cases = {
    "x[1]": x[1],
    "x[:, ::2]": x[:, ::2],
    "x[x > 5]": x[x > 5],
    "x[[0, 2]]": x[[0, 2]],
    "x.T": x.T,
    "x.reshape(4, 3)": x.reshape(4, 3),
}
for name, y in cases.items():
    print(f"{name:16s} view: {np.shares_memory(x, y)}")

# %% [markdown]
# ## Exercise 2.2 (B — broadcasting shape puzzles)
#
# *Predict the result shape — or that an error is raised — for elementwise addition of arrays with shapes: (a) `(3, 4)` and `(4,)`; (b) `(3, 4)` and `(3,)`; (c) `(3, 1)` and `(1, 4)`; (d) `(5,)` and `(5, 1)`; (e) `(2, 1, 4)` and `(3, 1)`; (f) `(8, 1, 6, 1)` and `(7, 1, 5)`. Verify with `np.broadcast_shapes`. Then write a single expression that standardises the columns of an `(n, p)` matrix, and explain why the row-wise analogue requires `keepdims=True` while the column-wise version does not.*
#
# **Solution.** Applying equation (2.1) — right-align, pad with 1s, each aligned pair must be equal or contain a 1:
#
# - (a) `(3,4)` + `(4,)` → pad to `(1,4)`; pairs (3,1),(4,4) → **`(3, 4)`**.
# - (b) `(3,4)` + `(3,)` → pad to `(1,3)`; pair (4,3) incompatible → **error**.
# - (c) `(3,1)` + `(1,4)` → pairs (3,1),(1,4) → **`(3, 4)`**.
# - (d) `(5,)` + `(5,1)` → pad to `(1,5)` vs `(5,1)` → **`(5, 5)`** — the silent outer-product trap: valid, and almost never what was meant.
# - (e) `(2,1,4)` + `(3,1)` → pad to `(1,3,1)`; pairs (2,1),(1,3),(4,1) → **`(2, 3, 4)`**.
# - (f) `(8,1,6,1)` + `(7,1,5)` → pad to `(1,7,1,5)` → **`(8, 7, 6, 5)`**.

# %%
pairs = [
    ((3, 4), (4,)), ((3, 4), (3,)), ((3, 1), (1, 4)),
    ((5,), (5, 1)), ((2, 1, 4), (3, 1)), ((8, 1, 6, 1), (7, 1, 5)),
]
for s, t in pairs:
    try:
        print(f"{s!s:14s} + {t!s:12s} -> {np.broadcast_shapes(s, t)}")
    except ValueError:
        print(f"{s!s:14s} + {t!s:12s} -> error (incompatible)")

# %% [markdown]
# Column standardisation in one expression:

# %%
X = rng.normal(size=(6, 3))
Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
Z.mean(axis=0).round(12), Z.std(axis=0, ddof=1).round(12)

# %% [markdown]
# Why the asymmetry: `X.mean(axis=0)` has shape `(p,)`, which right-aligns against the trailing (column) axis of `(n, p)` — exactly the intended pairing, so no reshaping is needed. `X.mean(axis=1)` has shape `(n,)`, which would right-align against the *columns* too: incompatible when `p ≠ n` (an error), and a silent `(n, n)` broadcast when `p = n` (worse). `keepdims=True` returns `(n, 1)`, which broadcasts across columns as intended.
#
# ## Exercise 3.1 (A — diagnose and fix)
#
# *Explain (i) what `loans[loans["term"] == 60]["int_rate"] = loans["int_rate"] + 0.5` does under pandas 3's copy-on-write semantics and why no data changes; (ii) why its behaviour under pandas 1.5 was undefined rather than correct; (iii) the idiomatic fix.*
#
# **Solution.** (i) The statement chains two indexing operations. `loans[loans["term"] == 60]` evaluates first and returns a new object that behaves as a copy; the item assignment `[...]["int_rate"] = ...` then writes into that temporary, which no name holds, and it is discarded. Under copy-on-write this is guaranteed dead code: pandas 3 emits a `ChainedAssignmentError` warning and `loans` is untouched.
#
# (ii) Before copy-on-write, whether the first `[]` returned a view or a copy of the underlying data depended on internal block layout — dtype homogeneity, prior operations, pandas version. When it happened to be a view the write reached `loans`; when a copy, it vanished. The `SettingWithCopyWarning` existed precisely because pandas could not promise which. Code that "worked yesterday" was relying on an accident of memory layout, not on defined behaviour.
#
# (iii) Select rows and column in a single `.loc` call so pandas performs one labelled write on the original:

# %%
fixed = loans.copy()
mask = fixed["term"] == 60
fixed.loc[mask, "int_rate"] = fixed["int_rate"] + 0.5

comparison = pd.DataFrame({"before": loans["int_rate"], "after": fixed["int_rate"]})
comparison.groupby(loans["term"]).mean()

# %% [markdown]
# Only the 60-month loans moved, by exactly 0.5 — the write reached the original, and the 36-month rows are untouched.
#
# ## Exercise 3.2 (B — three tidyverse translations)
#
# *Translate each pipeline into idiomatic pandas against the `loans` frame (methods, not loops; one chain each).*
#
# **(a)** `filter` on two conditions, `mutate`, grouped `summarise`. `query` handles the filter; named aggregations in `agg` give `summarise`'s column naming; `n()` is the `"size"` aggregation:

# %%
(
    loans
    .query("grade in ['A', 'B'] and loan_amnt > 5000")
    .assign(rate=lambda d: d["int_rate"] / 100)
    .groupby("grade")
    .agg(mean_rate=("rate", "mean"), n=("rate", "size"))
)

# %% [markdown]
# **(b)** A grouped `mutate` is `groupby(...).transform(...)` inside `assign` — the per-group statistic is broadcast back to the original shape, so no `ungroup()` step exists or is needed. pandas' `.std()` defaults to `ddof=1`, matching R's `sd()`. The filter on a derived column uses `.loc` with a callable (`query` also works, but the callable form keeps the expression in ordinary Python):

# %%
(
    loans
    .assign(
        amnt_z=lambda d: d["loan_amnt"]
        .sub(d.groupby("purpose")["loan_amnt"].transform("mean"))
        .div(d.groupby("purpose")["loan_amnt"].transform("std"))
    )
    .loc[lambda d: d["amnt_z"].abs() > 2]
    .sort_values("amnt_z", ascending=False)
)

# %% [markdown]
# **(c)** `count` + `pivot_wider` is a grouped size followed by `unstack`, which rotates one index level into columns; `fill_value=0` plays the role of `values_fill`:

# %%
loans.groupby(["grade", "term"]).size().unstack("term", fill_value=0)

# %% [markdown]
# `pd.crosstab(loans["grade"], loans["term"])` is an equivalent one-liner for this contingency-table special case; the `groupby`–`unstack` pattern is the general tool once the aggregation is anything other than a count.
#
# ## Exercise 3.3 (C — a chained cleaning function, with alignment tests)
#
# *Write `clean_loans(df)` as a single method chain using `.assign` and `.pipe`, adding `rate`, a per-grade z-score of `loan_amnt`, and a flag for loans more than 2 percentage points above their grade's mean rate. Then write assertion tests demonstrating order-invariance, and state which pandas semantic the tests exercised.*

# %%
def add_grade_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-grade standardisation and rate premium, broadcast to loan level."""
    grp = df.groupby("grade")
    return df.assign(
        amnt_z=(df["loan_amnt"] - grp["loan_amnt"].transform("mean"))
        / grp["loan_amnt"].transform("std"),
        rate_premium=df["int_rate"] - grp["int_rate"].transform("mean"),
    )


def clean_loans(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df
        .assign(rate=lambda d: d["int_rate"] / 100)
        .pipe(add_grade_stats)
        .assign(overpriced=lambda d: d["rate_premium"] > 2.0)
    )


clean = clean_loans(loans)
clean[["rate", "amnt_z", "rate_premium", "overpriced"]].head()

# %% [markdown]
# The tests: apply the same function to a row-shuffled copy and require label-by-label agreement of every derived column; then multiply by a weights Series delivered in yet another order and require the product to pair labels correctly.

# %%
derived = ["rate", "amnt_z", "rate_premium", "overpriced"]

shuffled = loans.sample(frac=1.0, random_state=rng)
assert not shuffled.index.equals(loans.index)  # the shuffle actually shuffled

pd.testing.assert_frame_equal(
    clean_loans(shuffled)[derived].sort_index(),
    clean[derived].sort_index(),
)

weights = pd.Series(
    rng.uniform(0.5, 1.5, size=len(loans)), index=loans.index
).sample(frac=1.0, random_state=rng)

weighted = clean["rate"] * weights           # aligned despite different orders
check_id = loans.index[7]
assert np.isclose(weighted.loc[check_id], clean.loc[check_id, "rate"] * weights.loc[check_id])
assert not weighted.isna().any()

print("all alignment tests passed")

# %% [markdown]
# The semantic exercised is **index alignment**: every derived quantity is keyed on the loan identifier, so grouped transforms and cross-Series arithmetic pair rows by label, making the entire pipeline invariant to row order — the property the §6 simulation showed positional arithmetic lacks.
#
# ## Exercise 4.1 (A — failures of global seeding)
#
# *Give two concrete ways a script using `np.random.seed(42)` with the global API can produce different results on two runs even though the seed is fixed, and explain how the `Generator` design eliminates each.*
#
# **Solution.**
#
# 1. **An intervening consumer of the global stream.** Any call between the seed and your draws that itself uses the global stream — a library utility, a plotting jitter, a colleague's import-time code — advances the shared state, shifting every subsequent draw. Upgrading a dependency can therefore change your "seeded" results. With a `Generator`, your stream is a local object passed explicitly; no third party can advance it.
#
# 2. **A change in execution order or a conditional draw.** Reordering two analysis blocks, or a branch that draws only when triggered (a debug path, a retry), changes how many variates have been consumed before each block, so every block's results depend on total program history. With one spawned child per component, each component owns an independent stream and its results depend only on its own code.
#
# The demonstration below is the only appearance of the legacy API in this course, retained to exhibit the failure:

# %%
np.random.seed(42)
clean_draws = np.random.normal(size=3)

np.random.seed(42)
_ = np.random.uniform()          # an "innocent" intervening call
shifted_draws = np.random.normal(size=3)

print("undisturbed:", clean_draws.round(4))
print("after 1 extra call:", shifted_draws.round(4))

# %%
mine = np.random.default_rng(42)
other = np.random.default_rng(123)
_ = other.uniform(size=1_000)    # someone else's randomness, any amount
print("my stream, regardless:", mine.normal(size=3).round(4))
print("fresh stream, same seed:", np.random.default_rng(42).normal(size=3).round(4))

# %% [markdown]
# The two `Generator` lines agree: nothing `other` did could touch `mine`.
#
# ## Exercise 4.2 (C — an order-dependence investigation)
#
# *Simulate `est_mean` (mean of 10 000 standard normals) and `est_tail` ($\hat P(Z > 2)$ from a further 10 000 draws) sharing one generator, in both execution orders; repeat with spawned child streams. Report the four pairs and state which design makes each estimator order-invariant, and why that matters.*

# %%
def est_mean(g: np.random.Generator) -> float:
    return float(g.normal(size=10_000).mean())


def est_tail(g: np.random.Generator) -> float:
    return float((g.normal(size=10_000) > 2).mean())


# Design 1: one shared generator
g = np.random.default_rng(7)
m_first, t_second = est_mean(g), est_tail(g)
g = np.random.default_rng(7)
t_first, m_second = est_tail(g), est_mean(g)

# Design 2: one spawned child per estimator
g = np.random.default_rng(7)
g_mean, g_tail = g.spawn(2)
m_a, t_a = est_mean(g_mean), est_tail(g_tail)
g = np.random.default_rng(7)
g_mean, g_tail = g.spawn(2)
t_b, m_b = est_tail(g_tail), est_mean(g_mean)

print("shared generator:")
print(f"  mean run first: {m_first:+.5f}   mean run second: {m_second:+.5f}")
print(f"  tail run second: {t_second:.5f}  tail run first:  {t_first:.5f}")
print("spawned streams:")
print(f"  mean either order: {m_a:+.5f} vs {m_b:+.5f}  (equal: {m_a == m_b})")
print(f"  tail either order: {t_a:.5f} vs {t_b:.5f}  (equal: {t_a == t_b})")

# %% [markdown]
# **Conclusion.** Under the shared generator, each estimator's value depends on *when* it ran: whichever runs second consumes draws 10 001–20 000 instead of 1–10 000, so both estimators change value when the order flips. Under spawned streams, each estimator owns its stream and returns bit-identical values in either order. The property matters for debugging and refactoring large simulations: with per-component streams, moving, disabling, or adding components leaves every other component's output unchanged, so a changed result isolates the component actually responsible — while under a shared stream any structural edit perturbs everything downstream, and "the numbers changed" carries no diagnostic information.
#
# ## Exercise 6.1 (A — restoring safety at the NumPy boundary)
#
# *The exposure system emits plain NumPy arrays with a separate array of obligor IDs. State the discipline that restores the safety of the §6 simulation before positional arithmetic, and give the one-line pandas idiom.*
#
# **Solution.** Reattach the labels and align to the risk table's index *before* any arithmetic — then verify the alignment found every obligor. In one line:
#
# ```python
# ead = pd.Series(ead_arr, index=id_arr).reindex(risk.index)
# ```
#
# followed by `assert not ead.isna().any()`, which catches unknown or missing IDs (`reindex` inserts `NaN` for any label absent from the source, so a silent mismatch becomes a loud one). After this, `ead` shares the risk table's row order by construction, and extracting `.to_numpy()` for positional computation is safe — the arrays now come from a single, already-aligned object, which is exactly the discipline §3.6 prescribed.

# %%
ead_arr = rng.lognormal(10.0, 1.0, size=5)
id_arr = np.array(["OBL4", "OBL1", "OBL3", "OBL0", "OBL2"])
risk_index = pd.Index([f"OBL{i}" for i in range(5)], name="obligor")

ead = pd.Series(ead_arr, index=id_arr).reindex(risk_index)
assert not ead.isna().any()
ead
