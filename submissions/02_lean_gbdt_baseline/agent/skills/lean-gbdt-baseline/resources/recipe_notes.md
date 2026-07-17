# Recipe 1: Lean GBDT Baseline — full notes

The trimmed-down common core shared across several heavier community solutions to
similar tabular competitions (dozens-to-hundreds of models, GPU, hours of iteration),
stripped to what a 60-minute CPU-only session can actually run. Treat this as the safe
first submission, not the ceiling — see later recipes for ensembling and shift-aware
validation once this baseline has a real score to beat.

## When to reach for this

First, always. Fastest to implement and validate; gives a real public-score baseline
before spending budget on anything fancier.

## Budget profile

- Core pipeline: a few minutes, CPU only, leaving most of the 60-minute session free.
- Dependencies: pandas, numpy, scikit-learn, one GBDT library. No internet.

## What `train_baseline.py` implements

- **Exploration checklist**: target balance, categorical/numeric split with cardinality,
  missing-value pattern, train/test duplicate-row check.
- **Target/frequency encoding of categoricals**, refit inside each CV fold only — never
  on the full train set, to avoid leakage.
- **Ratio/interaction features** between the numeric columns most correlated with the
  target, capped to a handful of pairs rather than an exhaustive sweep.
- **Polynomial (square) features** on the same top numeric columns.
- **Category dtype casting** for native GBDT categorical support (xgboost/lightgbm/
  catboost), in addition to the encoded numeric versions.
- **Decimal-digit / fractional-residue features** on every numeric column — digit at
  each decimal position, an "is this suspiciously round" flag, and residuals from common
  fractions (1/2, 1/4, 1/5, 1/10). Independently converged on across solutions to three
  different competitions without copying each other — strong evidence this is a real,
  structural property of this data family (synthetic generators leave detectable
  precision/rounding fingerprints), not a one-off trick. Treated as core.
- **Group-level binary-flag aggregation**: when several columns are variants of the same
  yes/no concept, sum them into a count plus has-any/has-all flags. Detected generically
  (3+ binary-valued numeric columns), not by name — silently skipped if none exist.
- **Multi-scale binning before encoding**: quantile bins, fixed-width bins,
  round-then-integer-divide, and plain truncation on high-cardinality numeric columns,
  each binning separately target/frequency-encoded. Kept to modest bin counts (tens).
- **Cardinality-based categorical/continuous split**: numeric columns with fewer than
  ~10 unique values are also treated as categorical-like for encoding purposes.

## Model & CV strategy

- 5-fold stratified K-fold (configurable via `--n-folds`).
- One GBDT — auto-selects xgboost → lightgbm → catboost → sklearn
  `HistGradientBoostingClassifier`, whichever is importable in the sandbox. No
  hyperparameter search; uses commonly-converged-on defaults rather than a tuning pass,
  since every heavy solution that ran serious search had far more time than this session
  has.
- **Shallow trees** (depth 3-4, default 4) with strong regularization
  (`subsample=0.5, colsample_bytree=0.2, reg_lambda=15, reg_alpha=10`) — two independent
  solutions in the reviewed set explicitly found shallow trees outperform deeper ones on
  this data family. These are a reasonable starting range, not values to copy blindly.

## Ensembling

None by default — a single well-validated GBDT is the point of this recipe. Use
`--second-model` for a 50/50 blend once the single-model baseline is submitted and
scored; that's Recipe 2 territory.

## Time-permitting extras implemented via CLI flags

- `--second-model {lightgbm,xgboost,catboost,hist_gbm}`: a second GBDT library averaged
  50/50 with the first, using the same CV split for a fair OOF blend estimate.
- `--features`: toggle feature groups on/off and compare `VALIDATION_SCORE` — add
  gradually and check each addition's own contribution rather than batch-adding, which
  can hide features that hurt as much as others help. **Confirmed on a local dataset**:
  `--features none` scored *higher* than `--features all` on `train_05` (0.6749 vs.
  0.6494) — do not assume more engineered features is strictly better.

## Time-permitting extras NOT implemented (documented here for a future recipe)

- **Benford's-Law leading-digit deviation** and a **TF-IDF-over-character-n-grams**
  feature on the string form of float columns — same "synthetic artifact detection"
  logic as the digit/fractional features above, more involved to implement correctly and
  cheaply, so kept as a documented follow-up rather than core.
- **Offline pseudo-target encoding**: target-encode a categorical using an auxiliary
  informative column already in the training data (one that correlates with the outcome
  but isn't the target itself) instead of the real label. Not implemented because it
  needs a specific auxiliary-column candidate identified per dataset via EDA first —
  worth trying by hand if the baseline underperforms and EDA turns up a promising
  candidate column, rather than something a generic script can detect on its own.
