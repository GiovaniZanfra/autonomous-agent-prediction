# Leak-Safe Frequency/Target Encoding

**The one non-negotiable pattern in this skill, regardless of how extensive your feature set
gets.** Frequency encoding (how common is this category?) and target encoding (what's the mean
target for this category?) are both extremely useful for categorical and binned/high-cardinality
columns — but if you fit either on the full training set *before* splitting into CV folds, the
validation fold's own target information leaks into its own encoded features through the
category statistics. Your CV score will look great and be meaningless; the private score will
disappoint. Fit only on each fold's training split, apply to that fold's validation split, and
refit fresh on the full training set only once, at the very end, to encode the test set.

```python
def encode_categoricals(train_fit_df, y_fit, apply_dfs, cat_like_cols, smoothing=20):
    """Fit frequency + smoothed target encoding on train_fit_df/y_fit; apply
    the fitted mapping to every dataframe in apply_dfs (including train_fit_df
    itself). Returns the new encoded column names."""
    global_mean = y_fit.mean()
    new_cols = []
    for c in cat_like_cols:
        freq_map = train_fit_df[c].value_counts(normalize=True)
        stats = y_fit.groupby(train_fit_df[c].values).agg(["mean", "count"])
        te_map = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)

        freq_name, te_name = f"{c}__freq", f"{c}__te"
        for df in apply_dfs:
            df[freq_name] = df[c].map(freq_map).fillna(0.0)
            df[te_name] = df[c].map(te_map).fillna(global_mean)
        new_cols += [freq_name, te_name]
    return new_cols
```

**How to use it correctly across a CV loop:**

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    fold_train = X.iloc[tr_idx].copy()
    fold_val = X.iloc[val_idx].copy()
    y_tr = y.iloc[tr_idx]

    # Fit ONLY on this fold's training split, apply to both train and val.
    enc_cols = encode_categoricals(fold_train, y_tr, [fold_train, fold_val], cat_like_cols)

    # ... train and evaluate this fold using fold_train/fold_val ...

# After CV is done, refit once on the FULL training set to encode the test set
# for your final model (this is the only time you fit on the full train set).
enc_cols = encode_categoricals(X, y, [X, X_test], cat_like_cols)
```

The `te_map` uses additive smoothing (`smoothing=20` above) so low-count categories pull toward
the global mean instead of overfitting to a handful of rows — worth keeping regardless of how
you adapt the rest of the function. `.fillna(...)` on the `.map(...)` calls handles categories
present in validation/test but never seen in the fitting split (a normal occurrence, not an
error) — smoothed toward the global mean rather than left as NaN.
