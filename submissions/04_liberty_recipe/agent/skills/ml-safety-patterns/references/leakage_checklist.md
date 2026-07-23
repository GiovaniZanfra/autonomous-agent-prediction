# Data Leakage Prevention Checklist

Data leakage occurs when information from outside the training dataset — or from the future,
or from the validation/test split itself — is used to build a feature or fit a model, leading
to overly optimistic local scores and a real performance gap on the private leaderboard.

## Target Leakage
**Rule**: no feature should be directly derived from, or a proxy for, the target column in a
way that wouldn't actually be available at inference time. Any encoding that uses target
statistics (frequency/target encoding, category means) must be fit per-CV-fold on the training
split only — see `leak_safe_encoding.md` for the exact pattern and why fitting on the full
training set before splitting leaks target information into validation.

## Train/Test Distribution Leakage
**Rule**: don't fit any statistic — imputers, scalers, encoders, bin edges — on train and test
combined. Fit only on train (or, inside CV, only on that fold's training split), then transform
test/validation using the already-fitted statistic. This includes the quantile/width bin edges
in `feature_engineering_patterns.md`'s multi-scale binning — those are computed from
`train_df` only and then applied to `test_df`, never recomputed on the combined data.

## Row-Level Leakage
**Rule**: check whether any row in train also appears (identical feature values) in test — if
duplicate rows exist across the split, a memorization-prone model can look artificially strong
in CV without actually generalizing. Cheap to check directly (`merge` on feature columns and
count matches) as part of your EDA pass.
