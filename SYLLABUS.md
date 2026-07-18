# Machine Learning with Python

A statistically rigorous ML curriculum for a finance-focused analyst, delivered as a series of Jupyter notebooks. This document is the source of truth: each notebook is authored against its section here, under the conventions in `CLAUDE.md`.

---

## Design principles

1. **Estimators, not recipes.** Every method is introduced with its loss function, the assumptions under which it works, its properties as an estimator, and its characteristic failure modes. `model.fit()` appears only after the reader knows what quantity is being estimated.
2. **Simulation first.** Each method is demonstrated against a synthetic data-generating process where the truth is known, so that claims (unbiasedness, coverage, optimism, variance reduction) can be verified empirically before the method touches real data. Real finance data follows as the second act of every module.
3. **Evaluation is the spine.** Proper scoring rules, calibration, leakage discipline, selection optimism, and temporal validity recur throughout. The curriculum is written from the validator's chair as much as the developer's: the terminal orientation is model risk and independent validation (PRA SS1/23 register), not Kaggle leaderboards.
4. **Finance as substrate.** Credit default, loan pricing, card fraud, and drift monitoring supply the applied examples. Capstone artefacts are written to double as portfolio evidence for probabilistic-modelling and model-validation roles.
5. **Python as a second statistical language.** The reader is R-fluent with graduate-level statistics. Python semantics are taught explicitly where they diverge from R (module 00), after which the course is Python-native and stops translating.

## Assumed background

Linear models and GLMs (theory and practice), likelihood methods, Bayesian fundamentals and shrinkage/empirical Bayes, survival analysis, proper-scoring-rule literacy (Brier score and its Murphy decomposition), statistical process control, and working R. Notebooks are brisk where this base carries the weight and slow where the machinery is ML-specific: boosting internals, conformal guarantees, Shapley computation, backpropagation.

## Structure and format

Thirteen notebooks in four blocks. Every notebook contains: prose exposition in full paragraphs (not bullet decks); mathematics in LaTeX with derivations for the results marked **[derive]** below; short, single-purpose code cells; a simulation study; a real-data application; tiered exercises at section ends (**A** comprehension, **B** derivation or extension, **C** small investigation, usually a simulation study); and a further-reading list. Solutions live in parallel notebooks under `solutions/`, with full workings, never inline.

## Datasets

- **Synthetic DGPs** — first-class citizens, specified per module.
- **Lending Club loans** (Kaggle) — running example for default classification and interest-rate regression, Blocks A–C.
- **ULB credit card fraud** (Kaggle) — class imbalance, anomaly detection, and the temporal-split demonstration. Its features being PCA outputs is itself used as a teaching point in module 09.
- **Capstone** — Home Credit Default Risk (Kaggle, relational, richer feature engineering) *or* Freddie Mac Single-Family Loan-Level dataset (registration required, real mortgage performance, closer to industry). Choose one before commissioning module 12.

No data is committed to the repository; `data/get_data.py` handles acquisition and caching.

---

## Block A — Foundations

### 00 · Python for the R-native statistician

**Purpose.** Convert R fluency into idiomatic Python without the subtle bugs that R muscle memory produces. This is the only module where R comparisons are permitted.

**Content.** The NumPy array model versus R vectors: memory layout, dtypes, formal broadcasting rules, views versus copies, ufuncs and vectorisation. pandas semantics that genuinely differ from the tidyverse: the index as a first-class object, alignment on index during arithmetic (a real semantic difference, not a syntax difference), `SettingWithCopy` and the copy-on-write model, groupby–apply versus `dplyr` verbs, method chaining with `.pipe`/`.assign`, and when to drop to NumPy. Modern RNG (`np.random.default_rng`, never legacy `np.random.seed`). Environment discipline: `uv`, `pyproject.toml`, lockfiles, kernels. The matplotlib object model (Figure/Axes), with the project plot theme in `src/tunnel_mpl.py`.

**Exercises.** Translate three tidyverse pipelines into idiomatic pandas; diagnose and fix a `SettingWithCopy` bug; broadcasting shape puzzles; write a `.pipe`-chained cleaning function with tests of index-alignment behaviour.

**Reading.** McKinney, *Python for Data Analysis* (reference, not cover-to-cover); pandas copy-on-write docs.

### 01 · The statistical learning frame

**Purpose.** Establish the decision-theoretic frame everything else hangs on.

**Content.** Loss, risk, empirical risk minimisation; the Bayes classifier and Bayes risk; the regression function as conditional expectation under squared loss and conditional median under absolute loss **[derive]**. The bias–variance decomposition for squared loss **[derive]**, with an honest note that the classification analogue is messier and contested. Overfitting framed as estimator variance. Training/validation/test roles; what cross-validation actually estimates — expected prediction error over training sets rather than the conditional error of *this* fitted model (Bates–Hastie–Tibshirani). Optimism of in-sample error for linear smoothers via effective degrees of freedom, df = tr(**H**) **[derive]**. One flagged aside on double descent as a modern complication (**Contested:** tag), with pointers, not a treatment.

**Simulation.** A known nonlinear DGP: fit polynomial families of increasing degree; empirically decompose bias² and variance across resampled training sets; verify the in-sample optimism formula against Monte Carlo truth.

**Applied.** Lending Club: naive in-sample versus held-out error for a first default model; the gap quantified.

**Exercises.** B: derive the decomposition and the conditional-median result. C: design a simulation showing CV's variance as k varies.

**Reading.** ESL ch. 2, 7; CASI ch. 12; Breiman (2001) "Two Cultures"; Bates, Hastie & Tibshirani (2023).

### 02 · Linear models as estimators

**Purpose.** Rebuild familiar territory in Python at full rigour, as the baseline everything later must beat.

**Content.** OLS geometry: projection, the hat matrix, leverage. Ridge via the SVD: shrinkage factors d²ⱼ/(d²ⱼ+λ) and effective degrees of freedom **[derive]**; ridge as MAP under a Gaussian prior, with a paragraph connecting shrinkage to James–Stein (assumed background, referenced not re-taught). Lasso: soft-thresholding in the orthogonal design **[derive]**, coordinate descent, and selection instability demonstrated by bootstrapping the active set. Elastic net briefly. The statsmodels/scikit-learn split as the inference-versus-prediction cultural divide, and when each is the right tool.

**Simulation.** Sparse and dense true coefficient regimes; ridge and lasso risk versus OLS as a function of λ; active-set instability under correlated designs.

**Applied.** Lending Club interest-rate regression: penalised linear baseline with honest uncertainty about which features "matter".

**Exercises.** B: derive the ridge SVD form and soft-thresholding. C: bootstrap lasso selection; report selection frequencies and discuss what they do and do not license.

**Reading.** ESL ch. 3; CASI ch. 7, 16; Efron et al. (2004) LARS (skim).

---

## Block B — Supervised learning and honest evaluation

### 03 · Classification, probability, and calibration

**Purpose.** Treat classifiers as probability estimators first, and make calibration a first-class property.

**Content.** Logistic regression as a GLM; IRLS **[derive]**; separation and penalisation. Proper scoring rules: definition, Brier and log score, propriety of the Brier score **[derive]**; the Murphy decomposition (assumed background — connected, not re-taught) as reliability/resolution/uncertainty. Calibration curves; ECE and the estimator pathologies of binning; Platt scaling versus isotonic regression; calibration on held-out data only. Class imbalance treated honestly: resampling (SMOTE et al.) distorts base rates and therefore calibration **[demonstrate in simulation]**; the principled alternative is cost-sensitive thresholding on calibrated probabilities. ROC versus precision–recall at low base rates.

**Simulation.** Known Bernoulli DGP: show SMOTE breaks calibration and that post-hoc recalibration partially repairs it; binning bias in ECE estimates.

**Applied.** ULB fraud: calibrated logistic baseline; threshold chosen by explicit cost matrix, not by F1 reflex.

**Exercises.** B: prove Brier propriety; derive IRLS updates. C: simulation comparing Platt vs isotonic across sample sizes.

**Reading.** Gneiting & Raftery (2007); Niculescu-Mizil & Caruana (2005); ESL ch. 4.

### 04 · Evaluation protocols: resampling, selection, and leakage

**Purpose.** The module the whole course pivots on: how to produce an error estimate you would sign your name to.

**Content.** Cross-validation variants (k-fold, repeated, stratified, grouped); the variance of CV and the Bengio–Grandvalet result that no unbiased estimator of that variance exists. Nested CV as the estimate of the *pipeline's* error; selection optimism ("winner's curse" of picking the best of m models) quantified by simulation. Hyperparameter search: grid versus random (Bergstra–Bengio argument), successive halving, a brief honest note on Bayesian optimisation. A leakage taxonomy: target leakage, group leakage, temporal leakage, and preprocessing leakage — with scikit-learn `Pipeline`/`ColumnTransformer` as the enforcement mechanism, not a convenience. Temporal data: walk-forward evaluation, and purged k-fold with embargo (López de Prado) for overlapping-label settings.

**Simulation.** Null-feature dataset: select the best of 50 models by CV and watch the selected score exceed chance; nested CV repairs the estimate. Preprocessing leakage: fitting a scaler outside the folds shifts the estimate measurably.

**Applied.** Lending Club with a temporal holdout; random-split versus temporal-split performance compared as a preview of module 10.

**Deliverable exercise.** Write a one-page *evaluation protocol* memo for the Block B applied problem — split design, metrics, selection rules — fixed before any model is fitted. This artefact is reused in modules 06 and 12.

**Reading.** ESL ch. 7; Bengio & Grandvalet (2004); Bergstra & Bengio (2012); López de Prado (2018) ch. 7.

### 05 · Trees and bagging

**Purpose.** From instability to variance reduction: the first genuinely nonparametric machinery.

**Content.** CART: greedy recursive partitioning, impurity measures, cost-complexity pruning, and instability as the defining pathology. Bagging: variance of an average of correlated predictors, ρσ² + (1−ρ)σ²/B **[derive]**, and why decorrelation is the whole game. Random forests: feature subsampling, out-of-bag error as free cross-validation, and two honest caveats — RF probability estimates are poorly calibrated out of the box, and trees cannot extrapolate beyond the convex hull of training targets (which matters for trended financial quantities).

**Simulation.** Verify the correlated-average variance formula empirically; demonstrate OOB error tracking k-fold CV; show extrapolation failure on a trending target.

**Applied.** Lending Club default: RF versus the module 03 penalised logistic baseline, evaluated under the module 04 protocol; recalibrate RF probabilities and re-compare.

**Exercises.** B: the variance derivation; why OOB approximates leave-one-out behaviour. C: calibration of RF probabilities before/after isotonic correction.

**Reading.** ESL ch. 9, 15; Breiman (2001) "Random Forests".

### 06 · Gradient boosting

**Purpose.** The workhorse of tabular finance ML, from first principles to production caveats.

**Content.** Boosting as functional gradient descent: fitting trees to the negative gradient of the loss **[derive]**; shrinkage and subsampling as regularisation. Second-order (Newton) boosting: the XGBoost objective, optimal leaf weight w*ⱼ = −Gⱼ/(Hⱼ+λ) and the split gain formula **[derive]**; LightGBM's histogram and leaf-wise growth. Monotonic constraints (and why they matter for credit models facing regulatory and conduct scrutiny) and interaction constraints. Early stopping on a validation fold as the primary regulariser. Custom objectives: the pinball loss for quantile boosting, foreshadowing module 08.

**Simulation.** Recover a known additive-plus-interaction DGP; show monotone constraints trading a little loss for guaranteed shape; early-stopping curves.

**Applied.** Champion–challenger under the pre-registered module 04 protocol: gradient boosting versus penalised logistic on Lending Club. Is the lift real? Bootstrap the AUC/Brier difference; DeLong test for correlated AUCs; report with uncertainty, not a leaderboard delta.

**Exercises.** B: derive the leaf-weight and gain formulas. C: implement a custom pinball objective and validate against library quantile mode.

**Reading.** Friedman (2001); Chen & Guestrin (2016); ESL ch. 10.

---

## Block C — The validator's toolkit

### 07 · Interpretability and model validation

**Purpose.** What can honestly be said about why a model predicts what it does — and the discipline of independent validation.

**Content.** Permutation importance and its failure under correlated features; conditional variants. Partial dependence and ICE; the H-statistic for interactions. Shapley values: the axioms, why exact computation is intractable, TreeSHAP; the interventional-versus-observational (conditional) expectation dispute presented as live (**Contested:** tag), with the practical consequences of each choice. What interpretability does not deliver: causal claims or recourse guarantees. Then validation as a discipline: benchmarking against simpler challengers, input sensitivity analysis, population and characteristic stability (PSI/CSI), stress-testing inputs, and documentation. One section maps this to the SS1/23 principle set — model inventory and tiering, development standards, independent validation and effective challenge, ongoing monitoring — as orientation, not legal exegesis.

**Simulation.** Correlated-features DGP where permutation importance actively misleads and conditional approaches partially recover; SHAP interventional vs observational divergence on the same model.

**Applied / deliverable.** A miniature validation report (3–4 pages, prose) for the module 06 champion model: benchmarking, stability, sensitivity, limitations, and a recommendation. This is the first full dress rehearsal of the target-role artefact.

**Reading.** Lundberg & Lee (2017); Lundberg et al. (2020); Janzing et al. (2020) and Sundararajan & Najmi (2020) for the dispute; PRA SS1/23 (the document itself — it is short).

### 08 · Uncertainty quantification

**Purpose.** From point predictions to honest intervals and distributions.

**Content.** Prediction intervals versus confidence intervals, and why naive bootstrap-of-fit undercovers for prediction. Quantile regression via the pinball loss, including gradient-boosted quantiles and the crossing problem. Conformal prediction: exchangeability, the split-conformal marginal coverage guarantee **[derive — it is a short, satisfying proof]**, conformalised quantile regression (Romano–Patterson–Candès), Mondrian/group-conditional conformal; the honest limits — marginal is not conditional coverage, and the impossibility results that make that gap fundamental (**flag, cite Vovk/Lei**). Scoring probabilistic forecasts: CRPS and pinball, connected back to Brier/Murphy from module 03 and to live forecasting practice (Dead Reckoning uses exactly this machinery).

**Simulation.** Coverage audits: split conformal hits nominal marginal coverage on exchangeable data; break exchangeability with covariate shift and watch coverage degrade; CQR versus plain conformal interval widths under heteroscedasticity.

**Applied.** Prediction intervals for Lending Club interest rates via CQR on the boosted model; group-conditional coverage checked across borrower segments.

**Exercises.** B: the split-conformal proof. C: a coverage study under shift, written up with an explicit statement of what guarantee survived.

**Reading.** Angelopoulos & Bates (2023) tutorial; Romano et al. (2019); Koenker for quantile regression background.

### 09 · Unsupervised structure, with skepticism

**Purpose.** Dimension reduction and clustering as hypothesis-generating tools whose outputs demand validation, not belief.

**Content.** PCA via the SVD: variance maximisation as an eigenproblem **[derive]**; the scaling decision; randomised SVD for scale; PCA inside a supervised pipeline as a leakage risk (fit within folds). Mixture models and EM: the E and M steps for a Gaussian mixture **[derive]**, monotone ascent of the likelihood, k-means as the small-variance limit. Cluster validity done honestly: silhouette's limits, stability-based validation (bootstrap ARI), and the trap that clusters exist because algorithms output them. Anomaly detection for fraud: Isolation Forest, LOF, one-class approaches, PCA reconstruction error — and the structural problem of evaluation without labels.

**Simulation.** Well-separated versus overlapping mixtures: watch every method "find" clusters in pure noise; stability analysis correctly refuses them.

**Applied.** ULB fraud as anomaly detection (its features being PCA components makes the meta-point about preprocessing provenance); comparison against the supervised benchmark from module 03.

**Reading.** ESL ch. 14; CASI ch. 13 (EM); Liu et al. (2008) Isolation Forest.

### 10 · Temporal ML and drift

**Purpose.** Everything about evaluation changes when time enters. Not a forecasting-theory course (M343 owns the stochastic-process theory); this is ML-on-temporal-tabular-data discipline.

**Content.** Feature construction without leakage: lags, rolling and expanding windows, groupby–shift patterns, and the point-in-time discipline of "what was knowable when". Walk-forward evaluation with gaps and purging; hyperparameter selection under temporal CV. Covariate shift versus concept drift, precisely distinguished. Monitoring in production terms: PSI over time, residual control charts (SPC assumed background — connected directly), and drift-triggered review as an SS1/23 ongoing-monitoring obligation.

**Simulation.** A DGP with an engineered regime change: random-split CV reports fantasy performance; walk-forward tells the truth; monitoring statistics detect the break with quantified delay.

**Applied.** ULB fraud with temporal splits — the canonical demonstration that random splits inflate fraud-model metrics; drift dashboard statistics computed across the file's time index.

**Exercises.** C: build a leak-free feature pipeline for a panel dataset and prove point-in-time correctness with an adversarial test.

**Reading.** López de Prado (2018) ch. 7–8 (selectively); Gama et al. (2014) drift survey (skim).

---

## Block D — Depth and synthesis

### 11 · Neural networks: foundations

**Purpose.** Mechanical understanding sufficient to make TM358 consolidation rather than first contact — and an honest account of when networks earn their keep on tabular data.

**Content.** An MLP from scratch in NumPy: forward pass, backpropagation derived cleanly via the chain rule with explicit shapes **[derive]**, gradient checking as a ritual. Then PyTorch: autograd, `nn.Module`, an honest training loop, early stopping, weight decay (and the AdamW decoupling caveat), dropout. The tabular reality check: gradient boosting usually wins (Grinsztajn et al.), and the cases where networks pay — learned embeddings for high-cardinality categoricals, sequence structure. Deliberately stops before architectures: CNNs, RNNs, and transformers are left for TM358.

**Simulation.** Overfit a tiny dataset on purpose as the standard sanity ritual; gradient-check a hand-rolled layer against autograd.

**Applied.** Entity embeddings for a high-cardinality categorical in Lending Club; benchmark against the module 06 champion under the standing protocol.

**Reading.** Goodfellow et al. ch. 6 (selectively); Grinsztajn, Oyallon & Varoquaux (2022); Loshchilov & Hutter (2019).

### 12 · Capstone: champion–challenger with a validation pack

**Purpose.** The synthesis artefact, built to industry shape.

**Content.** On the chosen capstone dataset (Home Credit or Freddie Mac), end to end: a pre-registered evaluation protocol (module 04 pattern, written first and committed before modelling); feature engineering with point-in-time discipline; a penalised GLM baseline; a boosted challenger with monotone constraints where economically sensible; calibration; conformal intervals; interpretability analysis; stability and sensitivity testing; temporal holdout. Two written deliverables: a **model development document** and an **independent-style validation memo** (limitations, effective challenge, recommendation with conditions). Together these form the portfolio centrepiece for probabilistic-modelling / model-validation applications — and a natural candidate for the real-name public writing track.

**Definition of done.** Both documents complete; every claim in them traceable to a notebook cell; the protocol memo unchanged since pre-registration (deviations logged, not silently absorbed).

---

## Cadence

Authoring cost is near zero once handed to Claude Code; the binding constraint is working-through time. At roughly 3–5 hours per module including exercises, the full course is ~50–65 hours. Suggested shape: modules 00–07 before the October OU start (about two per week through August–September, front-loading while capacity exists), then 08–12 at fortnightly cadence alongside TM358 + M343, finishing around year end. Module 11 lands most usefully just before or alongside TM358's opening weeks. This is a suggestion, not a schedule; the 60-credit October load remains the fragile element and this course should flex around it, not compete with it.

## Reference texts (standing)

*The Elements of Statistical Learning* (Hastie, Tibshirani & Friedman) as the primary spine; *Computer Age Statistical Inference* (Efron & Hastie) for the inferential register; *An Introduction to Statistical Learning with Python* as the gentler parallel where useful. Papers cited per module above.
