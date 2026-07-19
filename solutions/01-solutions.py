# %% [markdown]
# # Module 01 · Solutions
#
# Full workings for the exercises in `notebooks/01-statistical-learning-frame.ipynb`. Each exercise is restated before its solution. Simulation components (the sine DGP, Chebyshev designs) are rebuilt here from the module seed; this notebook's draw sequence differs from the teaching notebook's, so individual Monte Carlo values differ while every conclusion is unchanged.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev

MODULE_SEED = 1
rng = np.random.default_rng(MODULE_SEED)

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tunnel_mpl import apply_theme

apply_theme()

SIGMA = 0.3


def true_f(x):
    return np.sin(2 * np.pi * x)


def design(x, degree):
    return chebyshev.chebvander(2.0 * x - 1.0, degree)

# %% [markdown]
# ## Exercise 1.1 (A — losses and their optimal predictors)
#
# *For each loss, state the Bayes-optimal prediction as a functional of the conditional distribution of $Y \mid X = x$, with one line of justification each: (a) squared loss; (b) absolute loss; (c) 0–1 loss on $K$ classes; (d) the pinball loss $\rho_\tau$. Then explain why $\widehat{R}_n(\hat f)$ is downward-biased for $R(\hat f)$ when $\hat f$ was ERM-selected, while $\widehat{R}_n(g)$ is unbiased for $R(g)$ when $g$ was chosen without reference to the sample.*
#
# **Solution.**
#
# - (a) **Conditional mean** $\mathbb{E}[Y \mid X = x]$: by equation (2.1) of the module, the conditional expected loss is the conditional variance plus $(m(x) - a)^2$, minimised at $a = m(x)$.
# - (b) **Conditional median**: the derivative of the conditional expected loss is $2F(a) - 1$, zero exactly where $F(a) = \tfrac12$ (worked in full in exercise 2.1).
# - (c) **Posterior mode** $\arg\max_k \Pr(Y = k \mid X = x)$: the expected loss of action $k$ is $1 - \Pr(Y = k \mid x)$, minimised by the most probable class.
# - (d) **Conditional $\tau$-quantile** $F^{-1}(\tau)$: the derivative of the expected pinball loss is $F(a) - \tau$ (worked in exercise 2.1); note $\tau = \tfrac12$ recovers (b).
#
# On the bias: for a *fixed* function $g$, each $L(y_i, g(x_i))$ is an i.i.d. draw of the random variable $L(Y, g(X))$, so the empirical average is unbiased for $R(g)$. ERM breaks this by using the same sample twice — first to *choose* $\hat f$ as the minimiser of $\widehat{R}_n$ over $\mathcal{F}$, then to *report* $\widehat{R}_n(\hat f)$. The minimum of many correlated-with-the-data quantities is systematically below its population counterpart: $\mathbb{E}[\min_f \widehat{R}_n(f)] \le \min_f \mathbb{E}[\widehat{R}_n(f)] = \min_f R(f) \le R(\hat f)$ in expectation terms — the selected model looks best partly because the noise in this sample happened to flatter it.
#
# ## Exercise 2.1 (B — the conditional median and the pinball generalisation)
#
# *Derive the conditional-median result from first principles, treating carefully the step where the derivative passes the expectation. Then show that the minimiser of $\mathbb{E}[\rho_\tau(Y - a) \mid X = x]$ is the conditional $\tau$-quantile, and confirm $\tau = \tfrac12$ recovers (half) the absolute loss and the median.*
#
# **Solution.** Fix $x$; write $F$ and $p$ for the conditional distribution function and density of $Y$ given $X = x$, and $g(a) = \mathbb{E}[\,|Y - a|\,]$ (conditioning suppressed). The integrand $y \mapsto |y - a|$ is differentiable in $a$ except at $a = y$ (a $p$-null set), with $\partial_a |y - a| = \operatorname{sign}(a - y)$ where it exists, and the difference quotients are uniformly bounded: $\big|\,|y - a'| - |y - a|\,\big| \le |a' - a|$ by the triangle inequality, so the quotients are dominated by the constant $1$, which is integrable whenever $\mathbb{E}|Y| < \infty$. The dominated convergence theorem therefore licenses differentiation under the expectation:
#
# $$
# g'(a) = \mathbb{E}\big[\operatorname{sign}(a - Y)\big]
# = \Pr(Y < a) - \Pr(Y > a)
# = 2F(a) - 1
# $$
#
# (using $\Pr(Y = a) = 0$ under the density assumption). Each function $a \mapsto |y - a|$ is convex, expectations of convex functions are convex, so $g$ is convex and the stationary point $F(a) = \tfrac12$ is a global minimum. Without a density, $g$ is still convex; its one-sided derivatives are $g'(a^-) = 2F(a^-) - 1$ and $g'(a^+) = 2F(a) - 1$, and $a$ is a minimiser iff $g'(a^-) \le 0 \le g'(a^+)$, i.e. $F(a^-) \le \tfrac12 \le F(a)$ — precisely the definition of a median, possibly an interval of them.
#
# For the pinball loss, split the expectation at $a$. With $\rho_\tau(u) = u(\tau - \mathbf{1}\{u < 0\})$,
#
# $$
# g_\tau(a) = \mathbb{E}\big[\rho_\tau(Y - a)\big]
# = \tau \int_{y > a} (y - a)\, p(y)\, dy \;+\; (1 - \tau) \int_{y < a} (a - y)\, p(y)\, dy.
# $$
#
# The same domination argument (difference quotients bounded by $\max(\tau, 1-\tau) \le 1$) gives
#
# $$
# g_\tau'(a) = -\tau \Pr(Y > a) + (1 - \tau) \Pr(Y < a)
# = -\tau\big(1 - F(a)\big) + (1 - \tau) F(a)
# = F(a) - \tau,
# $$
#
# and convexity again upgrades the stationary condition to a global minimum: $F(a) = \tau$, the conditional $\tau$-quantile. At $\tau = \tfrac12$, $\rho_{1/2}(u) = \tfrac12 |u|$ — half the absolute loss, hence the same minimiser — and $F(a) = \tfrac12$ is the median condition, as required. A numerical spot check on a skewed distribution:

# %%
y_sim = rng.lognormal(0.0, 1.0, size=400_000)
tau = 0.9


def pinball(u, tau):
    return u * (tau - (u < 0))


a_grid = np.linspace(0.5, 6.0, 400)
risks = np.array([pinball(y_sim - a, tau).mean() for a in a_grid])
print(f"argmin of MC pinball risk (τ=0.9): {a_grid[risks.argmin()]:.3f}")
print(f"0.9-quantile of the sample:        {np.quantile(y_sim, tau):.3f}")
print(f"theoretical LogNormal quantile:    {np.exp(1.2816):.3f}")

# %% [markdown]
# The minimiser of the Monte Carlo pinball risk sits on the 0.9-quantile, as derived.
#
# ## Exercise 3.1 (B — the decomposition, and optimal shrinkage)
#
# *(i) Derive the decomposition (3.1), stating exactly where each independence and zero-mean assumption is used. (ii) For $\hat\theta$ unbiased for $\theta \ne 0$ with variance $v$ and the shrunken family $\hat\theta_\alpha = \alpha \hat\theta$, derive $\mathrm{MSE}(\alpha)$ and the minimising $\alpha^\ast$, show $\alpha^\ast < 1$ strictly, and connect to shrinkage estimation.*
#
# **Solution (i).** Setup: $Y_0 = f(x_0) + \varepsilon_0$ with $\mathbb{E}[\varepsilon_0] = 0$, $\operatorname{Var}(\varepsilon_0) = \sigma^2$; the fitted $\hat f(x_0)$ is a function of the training sample $\mathcal{D}$ alone; $\varepsilon_0 \perp \mathcal{D}$. Expand around $f(x_0)$:
#
# $$
# \mathbb{E}\big[(Y_0 - \hat f(x_0))^2\big]
# = \mathbb{E}[\varepsilon_0^2]
# + 2\,\mathbb{E}\big[\varepsilon_0 \big(f(x_0) - \hat f(x_0)\big)\big]
# + \mathbb{E}\big[(f(x_0) - \hat f(x_0))^2\big].
# $$
#
# The cross term factorises as $2\,\mathbb{E}[\varepsilon_0]\,\mathbb{E}[f(x_0) - \hat f(x_0)]$ — this is where **independence** of $\varepsilon_0$ and $\mathcal{D}$ is used, and it then dies by the **zero-mean** property of $\varepsilon_0$. (Independence is not decorative: if the "test" point were one of the training points, $\varepsilon_0$ and $\hat f$ would be correlated and the cross term would survive — that surviving term is exactly the optimism of §5.) Next expand the third term around $\bar f(x_0) = \mathbb{E}_{\mathcal{D}}[\hat f(x_0)]$:
#
# $$
# \mathbb{E}\big[(f(x_0) - \hat f(x_0))^2\big]
# = \big(f(x_0) - \bar f(x_0)\big)^2
# + 2\big(f(x_0) - \bar f(x_0)\big)\,\mathbb{E}\big[\bar f(x_0) - \hat f(x_0)\big]
# + \mathbb{E}\big[(\bar f(x_0) - \hat f(x_0))^2\big],
# $$
#
# where the middle term vanishes **by the definition of $\bar f(x_0)$** as the mean of $\hat f(x_0)$ — no assumption needed, only that the mean exists. Collecting terms gives $\sigma^2 + \text{bias}^2 + \text{variance}$, equation (3.1).
#
# **Solution (ii).** Bias and variance of $\hat\theta_\alpha$: $\mathbb{E}[\alpha\hat\theta] - \theta = (\alpha - 1)\theta$ and $\operatorname{Var}(\alpha\hat\theta) = \alpha^2 v$, so
#
# $$
# \mathrm{MSE}(\alpha) = (\alpha - 1)^2 \theta^2 + \alpha^2 v,
# \qquad
# \frac{d\,\mathrm{MSE}}{d\alpha} = 2(\alpha - 1)\theta^2 + 2\alpha v = 0
# \;\Longrightarrow\;
# \alpha^\ast = \frac{\theta^2}{\theta^2 + v}.
# $$
#
# Since $v > 0$ and $\theta^2 < \infty$, $\alpha^\ast < 1$ strictly: *some* shrinkage towards zero always beats the unbiased estimator in MSE, with the optimal amount governed by the noise-to-signal ratio $v/\theta^2$. This is the one-parameter kernel of the shrinkage phenomenon familiar from the Bayesian and empirical-Bayes background: the posterior mean under a zero-centred prior is exactly a data-weighted shrinkage of this form, and James–Stein estimates the shrinkage factor from the data across coordinates. The catch is that $\alpha^\ast$ depends on the unknown $\theta$, so it is infeasible as written; practical methods substitute an estimate of the signal-to-noise ratio — via a prior (Bayes), via pooling across coordinates (empirical Bayes / James–Stein), or via a penalty parameter chosen by cross-validation, which is precisely what ridge regression does in module 02. A quick Monte Carlo, with $\theta = 1$, $v = 0.5$:

# %%
theta, v = 1.0, 0.5
alpha_star = theta**2 / (theta**2 + v)
theta_hat = rng.normal(theta, np.sqrt(v), size=400_000)
mse = {"α = 1 (unbiased)": ((theta_hat - theta) ** 2).mean(),
       f"α* = {alpha_star:.3f}": ((alpha_star * theta_hat - theta) ** 2).mean()}
print(pd.Series(mse).round(4).to_string())
print(f"theoretical MSE(α*): {alpha_star * v:.4f}")

# %% [markdown]
# The shrunken estimator beats the unbiased one by the predicted margin ($\mathrm{MSE}(\alpha^\ast) = \alpha^\ast v$), accepting bias to buy a larger variance reduction — the decomposition of (i) used as an instrument rather than an observation.
#
# ## Exercise 4.1 (A — what is being estimated)
#
# *For each scenario, state whether the quantity estimated is the conditional error $\mathrm{Err}_{\mathcal{D}}$, the expected error $\mathrm{Err}$, or neither, and name the flaw if any.*
#
# **Solution.**
#
# - **(a)** A temporal holdout scored once after all decisions froze estimates $\mathrm{Err}_{\mathcal{D}}$ — the error of *this* fitted model — with respect to the future-period distribution. This is the cleanest of the three; its remaining caveats are distributional (the future may differ from the holdout period), not procedural.
# - **(b)** The CV comparison itself is legitimate procedure-level inference, but *reporting the winner's CV score as its performance estimate* is **neither**: selection has biased the reported number optimistically (the winner won partly on noise — the "winner's curse"). The repair is to estimate the error of the *entire pipeline including selection*, which is nested cross-validation, module 04.
# - **(c)** **Neither.** A validation set consulted repeatedly during feature engineering has been absorbed into training in the information-flow sense; its final score inherits selection optimism of unknown size and is no longer an estimate of out-of-sample error at all. Only data that influenced no decision can play the test role.
#
# ## Exercise 4.2 (C — the variability of cross-validation as $k$ varies)
#
# *Design and run a simulation showing how the variability of the $k$-fold CV estimate depends on $k$, using a known DGP, a fixed model, $k \in \{2, 5, 10, n\}$, many independent training sets, and the true expected error by Monte Carlo; exploit the LOO shortcut for linear smoothers; state which sources of variability the design does and does not average over.*
#
# **Design.** DGP as in the module's §6: $Y = \sin(2\pi X) + \varepsilon$, $\sigma = 0.3$, $n = 100$, fixed model a cubic (Chebyshev basis, degree 3). For each of $R = 300$ independent training sets we compute $\widehat{\mathrm{CV}}_k$ for $k = 2, 5, 10$ by explicit fold rotation (one random fold assignment per dataset) and $\widehat{\mathrm{CV}}_{(n)}$ by the leave-one-out identity $\frac{1}{n}\sum_i \big(\frac{y_i - \hat y_i}{1 - h_{ii}}\big)^2$, valid for any linear smoother because deleting observation $i$ and refitting changes $\hat y_i$ by exactly the factor $1/(1 - h_{ii})$. The benchmark $\mathrm{Err}$ — the expected prediction error of the procedure at training size $n$ — is estimated by fitting on each full dataset and scoring on a fresh 2 000-point test sample, averaged over the $R$ replicates.

# %%
n, R, DEGREE = 100, 300, 3
ks = [2, 5, 10]

cv_scores = {k: np.empty(R) for k in ks}
cv_scores["LOO"] = np.empty(R)
full_test_err = np.empty(R)

for r in range(R):
    x = rng.uniform(0.0, 1.0, size=n)
    y = true_f(x) + rng.normal(0.0, SIGMA, size=n)

    # explicit k-fold
    fold_order = rng.permutation(n)
    for k in ks:
        folds = np.array_split(fold_order, k)
        sq_errs = np.empty(n)
        for fold in folds:
            mask = np.ones(n, dtype=bool)
            mask[fold] = False
            coef, *_ = np.linalg.lstsq(design(x[mask], DEGREE), y[mask], rcond=None)
            sq_errs[fold] = (y[fold] - design(x[fold], DEGREE) @ coef) ** 2
        cv_scores[k][r] = sq_errs.mean()

    # LOO via the hat-matrix identity
    B = design(x, DEGREE)
    Q, _ = np.linalg.qr(B)
    h = (Q**2).sum(axis=1)
    coef, *_ = np.linalg.lstsq(B, y, rcond=None)
    resid = y - B @ coef
    cv_scores["LOO"][r] = ((resid / (1.0 - h)) ** 2).mean()

    # truth: fit on all n, score on fresh data
    x_test = rng.uniform(0.0, 1.0, size=2_000)
    y_test = true_f(x_test) + rng.normal(0.0, SIGMA, size=2_000)
    full_test_err[r] = ((y_test - design(x_test, DEGREE) @ coef) ** 2).mean()

err_n = full_test_err.mean()
table = pd.DataFrame(
    {label: {"mean CV": s.mean(), "SD of CV": s.std(ddof=1),
             "bias vs Err": s.mean() - err_n}
     for label, s in cv_scores.items()}
).T
print(f"Err (procedure, training size n=100): {err_n:.4f}")
table.round(4)

# %% [markdown]
# **Conclusions.** Both bias and variability move in the expected directions. The $k = 2$ estimate is visibly biased upwards — each fold's model trains on only $n/2$ points, so CV honestly reports the error of a *smaller-sample procedure* — and it is also the most variable. Increasing $k$ shrinks the training-size bias (LOO's is negligible by construction) and, in this smooth squared-loss setting, the standard deviation falls from $k=2$ towards $k=10$ and stabilises, with LOO comparable to $k=10$. Two design notes qualify the picture. First, our SD averages over *datasets*; for $k < n$ there is an additional within-dataset variance from the random fold assignment that a single assignment per dataset leaves confounded with sampling variance (repeated CV would separate them; LOO has no such randomness). Second, none of these SDs is estimable from a single dataset's fold scores in the way naive standard errors pretend — the folds share training data — which is the Bengio–Grandvalet impossibility taken up in module 04. General claims that LOO dominates or is dominated in variance should be resisted; the ranking is setting-dependent, and module 04 revisits it with sharper tools.
#
# ## Exercise 5.1 (A — reading the optimism formula)
#
# *(a) Why is $\omega > 0$ for any sensible smoother with $\operatorname{tr}(\mathbf{H}) > 0$, and what does df mean operationally? (b) Compute df for the constant-$\bar y$ predictor and for interpolation. (c) 25 parameters, $n = 50$, training MSE 0.71, $\hat\sigma^2 \approx 1$: what does (5.2) predict for in-sample test error, and why is even that an understatement out of sample?*
#
# **Solution.**
#
# **(a)** From (5.1), $\omega = \frac{2}{n}\sum_i \operatorname{Cov}(y_i, \hat y_i)$; for a linear smoother this is $\frac{2\sigma^2}{n}\sum_i h_{ii}$. A sensible smoother responds non-negatively to its own response ($h_{ii} \ge 0$; for projections $h_{ii} = \sum_j H_{ij}^2 \ge 0$ automatically), so every term is non-negative and the total is positive whenever the trace is. Operationally, $h_{ii} = \partial \hat y_i / \partial y_i$: df totals the model's *self-sensitivity* — how much of each observation's own noise it absorbs into that observation's prediction — which is why it, and not the raw parameter count, is the honest complexity measure for shrinking and smoothing methods.
#
# **(b)** The constant predictor $\hat y_i = \bar y$ has $\mathbf{H} = \frac{1}{n}\mathbf{1}\mathbf{1}^{\top}$, so $\mathrm{df} = \operatorname{tr}(\mathbf{H}) = n \cdot \frac1n = 1$: one parameter's worth of flexibility, as intuition demands. Interpolation has $\mathbf{H} = \mathbf{I}$, $\mathrm{df} = n$, and optimism $2\sigma^2$: training error (zero) has become completely uninformative, overstating quality by twice the noise variance.
#
# **(c)** $\omega = 2\hat\sigma^2\,\mathrm{df}/n = 2 \cdot 1 \cdot 25/50 = 1.0$, so predicted in-sample test error $\approx 0.71 + 1.00 = 1.71$ — the training number undersells the error by a factor of ~2.4. It is still an understatement of out-of-sample error because $\mathrm{Err}_{\mathrm{in}}$ scores fresh *responses at the training inputs*: a new sample also draws new inputs, adding error wherever the fit is poor in regions the training design undersampled — and, in production, wherever the input distribution has drifted (module 10). ($\hat\sigma^2$ itself is typically estimated from the optimistic fit, compounding the understatement.)
#
# ## Exercise 6.1 (C — optimism under heteroscedasticity)
#
# *Show that with $\operatorname{Var}(\varepsilon_i) = \sigma_i^2$ the linear-smoother optimism becomes $\omega = \frac{2}{n}\sum_i \sigma_i^2 h_{ii}$. Simulate with $\sigma_i$ increasing in $x_i$, compare empirical optimism against the correct and the homoscedastic-average formulas, identify the leverage pattern that maximises their divergence, and draw the implication for $C_p$-style corrections.*
#
# **Derivation.** Equation (5.1), $\omega = \frac{2}{n}\sum_i \operatorname{Cov}(y_i, \hat y_i)$, made no homoscedasticity assumption. For $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$ with $\operatorname{Var}(\mathbf{y}) = \boldsymbol\Sigma = \operatorname{diag}(\sigma_1^2, \dots, \sigma_n^2)$,
#
# $$
# \sum_i \operatorname{Cov}(y_i, \hat y_i) = \operatorname{tr}\big(\operatorname{Cov}(\hat{\mathbf{y}}, \mathbf{y})\big)
# = \operatorname{tr}(\mathbf{H}\boldsymbol\Sigma)
# = \sum_i h_{ii}\,\sigma_i^2,
# $$
#
# since $(\mathbf{H}\boldsymbol\Sigma)_{ii} = h_{ii}\sigma_i^2$ for diagonal $\boldsymbol\Sigma$. Hence $\omega = \frac{2}{n}\sum_i \sigma_i^2 h_{ii}$: a *leverage-weighted* average of the noise variances. The homoscedastic formula with the average variance, $\frac{2\bar\sigma^2}{n}\operatorname{tr}(\mathbf{H})$, coincides with it exactly when leverage and noise are uncorrelated across observations — e.g. flat leverage — and diverges most when the noisiest observations sit at the highest-leverage design points.

# %%
n_het = 40
x_het = np.linspace(0.0, 1.0, n_het)
sig_i = 0.1 + 0.6 * x_het                       # noise grows along x
DEGREE_HET = 3

Q, _ = np.linalg.qr(design(x_het, DEGREE_HET))
H = Q @ Q.T
h_diag = np.diag(H)
f_het = true_f(x_het)

M = 40_000
eps = rng.normal(0.0, 1.0, size=(M, n_het)) * sig_i
eps_star = rng.normal(0.0, 1.0, size=(M, n_het)) * sig_i
y_hat = (f_het + eps) @ H.T
emp = (((f_het + eps_star - y_hat) ** 2).mean(axis=1)
       - ((f_het + eps - y_hat) ** 2).mean(axis=1)).mean()

correct = 2.0 / n_het * (sig_i**2 * h_diag).sum()
homosced = 2.0 * (sig_i**2).mean() * np.trace(H) / n_het

print(f"empirical optimism:            {emp:.4f}")
print(f"correct  2/n Σ σ²ᵢ hᵢᵢ:        {correct:.4f}")
print(f"homoscedastic  2σ̄² tr(H)/n:    {homosced:.4f}")
print(f"corr(hᵢᵢ, σ²ᵢ): {np.corrcoef(h_diag, sig_i**2)[0, 1]:.2f}")

# %% [markdown]
# The Monte Carlo optimism lands on the leverage-weighted formula; the homoscedastic approximation misses it, despite using the *correct average* noise level, because polynomial leverage is largest at the interval edges and the right-hand edge is where we put the noise. That is the general mechanism: the divergence is $\frac{2}{n}\sum_i (\sigma_i^2 - \bar\sigma^2)(h_{ii} - \bar h)$, a covariance between leverage and noise variance, maximised when high-variance observations occupy high-leverage positions (edges, sparse regions, extreme covariate values). The implication for practice: a $C_p$/AIC-style correction built on a single pooled $\hat\sigma^2$ *understates* optimism — and therefore over-selects complexity — precisely when the design's influential points are also its noisiest, which in financial data (heavy-tailed, volatility-clustered, extreme obligors most influential) is the rule rather than the exception. Under heteroscedasticity, resampling-based estimates of generalisation error inherit no such fragility, one more reason module 04's protocols carry the load in this course.
#
# ## Exercise 7.1 (A — what the applied protocol still owes us)
#
# *List three specific respects in which §7's protocol falls short of an estimate you would sign your name to, and for each name the module supplying the repair; at least one should concern the data.*
#
# **Solution.** More than three are available; the following would head a reviewer's list.
#
# 1. **The completed-loans filter conditions on the outcome** (a data problem). Keeping only loans whose terminal status is known truncates the most recent vintages asymmetrically: a loan issued in 2017 appears in a 2018-snapshot file as "completed" mainly if it defaulted early or prepaid quickly, so the sample over-represents fast resolutions and ties the default rate to vintage age. An honest design defines the target as default *within a fixed horizon* of origination and includes only vintages old enough for that horizon to have elapsed — point-in-time discipline, module 10.
# 2. **The split is random where the world is temporal.** Loans co-move with the credit cycle, and a random split lets the model train on 2015 loans and be "tested" on 2012 ones; performance on next year's book is the question actually asked of a credit model. Temporal holdouts and walk-forward evaluation arrive in module 04 and become the whole subject of module 10.
# 3. **No selection accounting and no uncertainty.** We compared five specifications and (implicitly) admired the best held-out score without adjusting for having selected it, and reported all scores as bare point estimates from a single split. Nested CV and the quantified winner's curse are module 04; bootstrap and DeLong-style uncertainty on score differences arrive with the champion–challenger comparison of module 06.
#
# (Also creditable: no leakage audit of the feature set beyond excluding the lender's own risk grades — module 04 builds the taxonomy; and log-loss reported without a calibration analysis — module 03.)
