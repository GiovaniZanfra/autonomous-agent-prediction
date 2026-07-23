# Category Dtype Safety

A real, confirmed crash on this dataset family: casting training and validation columns to
pandas `"category"` dtype **independently** produces two different category sets whenever
validation contains a category value train didn't. XGBoost, LightGBM, and scikit-learn's
`HistGradientBoostingClassifier` all crash at predict time with something like "Found a
category not in the training set" when this happens — and it happens on real datasets in this
family, not just hypothetically.

**The fix**: cast validation/test to train's *exact* dtype (same category set), not an
independent cast. A category seen only in validation/test then becomes `NaN`, which every one
of these libraries already treats as an ordinary missing value — no crash, no special handling
needed.

```python
for c in cat_cols:
    X_tr[c] = X_tr[c].astype("category")
    # Reuse train's exact category set (not an independent cast) so a
    # test-only category becomes NaN (a normal missing value to the
    # model) instead of crashing at predict time on an unseen level.
    X_val[c] = X_val[c].astype(X_tr[c].dtype)
```

Apply this same pattern for XGBoost (`enable_categorical=True`), LightGBM (native
`category` dtype support), and `HistGradientBoostingClassifier` (`categorical_features=...`).

**CatBoost is the exception** — it doesn't need this pattern. Cast categorical columns to
plain strings (`.astype(str)`) for both train and validation/test instead; CatBoost handles
unseen categories natively without needing the shared-dtype trick.

This applies regardless of which specific feature engineering you do — any categorical or
binned column (see `feature_engineering_patterns.md`) that ends up going into the model as a
native categorical feature (as opposed to only through frequency/target encoding) needs this.
