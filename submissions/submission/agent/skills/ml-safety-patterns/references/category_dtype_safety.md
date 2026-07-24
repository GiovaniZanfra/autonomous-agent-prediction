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

## Don't `LabelEncoder` a nominal column and feed it as if numeric/ordinal

This is a real, confirmed mistake, not a hypothetical: a prior run loaded this exact
reference, then wrote its own code that `LabelEncoder`'d every nominal categorical column
into an arbitrary integer and fed the result straight to the GBDTs as a plain numeric
column — on a dataset that was 100% categorical, meaning this was the *entire* feature set.

A nominal category (e.g. `cat_0` .. `cat_7` with no real order) has no meaningful "greater
than" relationship. Mapping it to an arbitrary integer and handing it to a tree model as a
numeric feature forces the tree to build splits like "is this category's arbitrary code
greater than 3" — which is meaningless and throws away the actual category structure. This
is a *different* problem from the one the fix above solves (that fix is about safely casting
to `"category"` dtype without crashing on unseen levels; it assumes you're already using
native categorical support). Skipping native categorical support entirely and using
label-encoded integers instead isn't safer, it's just a worse representation.

**The native categorical pattern above (cast to `"category"` dtype for XGBoost/LightGBM/
HistGBM, raw strings for CatBoost) is the *primary* representation for any categorical
column fed to a GBDT — not `LabelEncoder`, not any other integer-mapping scheme.**
Frequency/target encoding (`leak_safe_encoding.md`) is a *supplementary* feature computed
alongside the native categorical column, not a replacement for it, and stacking it on top of
an already label-encoded column just adds noise to a bad representation rather than fixing
it (this is exactly what happened in the prior run: target/frequency encoding "didn't help"
because the underlying representation it was built on top of was already wrong).
