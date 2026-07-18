---
name: lean-gbdt-baseline
description: >-
  Runs the full "lean GBDT baseline" recipe in one CV-safe script: EDA
  summary, capped feature engineering, frequency/target encoding refit per
  fold (leak-free), and one GBDT (auto-selects whichever of
  xgboost/lightgbm/catboost/sklearn is available).
---

# Lean GBDT Baseline Skill

The trimmed-down common core of several heavier community solutions to similar tabular
competitions, stripped to what a 60-minute CPU-only session can run. Always try this
first — it's the fastest way to a real, leak-free public-score baseline before spending
budget on anything fancier.

## Available Scripts

### 1. `train_baseline.py`

Loads train/test, prints a compact EDA summary, engineers a capped set of features, fits
one GBDT inside stratified K-fold CV with per-fold-refit frequency + smoothed target
encoding (never fit on the full train set — the recipe's own leakage warning), and writes
a submission-ready CSV from a model refit on the full training set.

**Usage via `run_skill_script`** (defaults are the recommended first call):
```python
run_skill_script(
    skill_name="lean-gbdt-baseline",
    script_name="train_baseline.py",
    args="--train /work/train.csv --test /work/test.csv --sample-submission /work/sample_submission.csv --target target --output /work/submission.csv",
)
```

**IMPORTANT — always use absolute `/work/...` paths for every file argument
(`--train`/`--test`/`--sample-submission`/`--output`), never relative ones.**
`run_skill_script` materializes the skill's own files into a fresh temporary directory
and runs the script there — that directory does NOT contain your problem's
`train.csv`/`test.csv`, only this skill's bundled scripts, and it is deleted the moment
the tool call returns. A relative `--train` path will fail with "not found" even though
the file exists in `/work`; and a relative `--output` path will write the submission CSV
into that temporary directory, where it vanishes before you can `submit_predictions` it.

**Arguments**:
- `--train` / `--test` / `--sample-submission`: input CSV paths (defaults: `train.csv`,
  `test.csv`, `sample_submission.csv` — always override with the absolute `/work/...`
  path).
- `--target`: target column name (default: `target`).
- `--id-col`: row identifier column, passed through untouched (default: `row_id`).
- `--output`: submission CSV path to write (default: `submission.csv` — always override
  with an absolute `/work/...` path, e.g. `/work/submission.csv`).
- `--n-folds`: CV folds (default: 5).
- `--depth`: GBDT max tree depth (default: 4 — shallow, per the recipe's evidence that
  shallow trees generalize better than deeper ones on this data family).
- `--model {auto,xgboost,lightgbm,catboost,hist_gbm}`: GBDT backend. `auto` (default)
  tries them in that order and uses whichever is importable.
- `--second-model {none,xgboost,lightgbm,catboost,hist_gbm}`: if set, trains a second
  backend with the same CV split and averages test predictions 50/50 with the first —
  the cheapest next step after the single-model baseline is submitted.
- `--features`: comma-separated subset of `{ratio_poly,digit_frac,group_flag,
  multiscale_bin}`, or `all` (default) / `none`. **Try `none` and `all` and compare —
  more engineered features is not guaranteed to score higher**; add groups incrementally
  and check each one's own contribution rather than trusting `all` blindly.

**Output**: prints `VALIDATION_SCORE: <OOF ROC-AUC>` and `VALIDATION_STD: <fold-to-fold
std>` on the last two lines — use these (not the public score) to decide which
submission(s) to select at the end. Writes the submission CSV with the same columns and
row count as `sample_submission.csv`.

**Feature groups** (see `resources/recipe_notes.md` for the full rationale and evidence
behind each one):
- `ratio_poly`: ratio and interaction features between the numeric columns most
  correlated with the target, plus their squares. Capped to a handful of pairs, not an
  exhaustive sweep.
- `digit_frac`: decimal-digit and fractional-residue features on every numeric column
  (digit at each decimal position, "suspiciously round" flag, residuals from common
  fractions). Independently converged on across multiple unrelated solutions to
  different competitions in this family — treat as core, not a thin-evidence bet.
- `group_flag`: if 3+ numeric columns are binary `{0,1}`-valued, adds a sum/has-any/
  has-all aggregate over that family. Silently skipped if no such family exists.
- `multiscale_bin`: bins high-cardinality numeric columns four different ways (quantile,
  fixed-width, round-then-divide, truncation); each binning gets its own frequency/target
  encoding.

**Categorical/ordinal handling**: schema-agnostic, like the `feature-engineer` skill in
`submissions/01_fe_categorical_encoding` — column type is inferred from values, not name
or position, since the dataset family does not keep a fixed feature-to-type mapping.
Numeric columns with fewer than ~10 unique values are also treated as categorical-like
(cardinality-based split).

---

## Domain Knowledge Resources

### `recipe_notes.md`
The full "Recipe 1" writeup this skill implements, including the techniques *not*
implemented in `train_baseline.py` (kept as documented follow-ups rather than core, since
they are more involved and less evidenced). Read it via `load_skill_resource`:
```python
load_skill_resource(
    skill_name="lean-gbdt-baseline",
    resource_name="recipe_notes.md",
)
```
