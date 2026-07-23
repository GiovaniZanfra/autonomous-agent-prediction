# Diverse Model Ensemble (Recipe 2 pattern)

The highest-consensus finding across the wider research this project draws from: train
several genuinely diverse models, blend their out-of-fold (OOF) predictions. Verified
directly (not just documented) against two local datasets, including a zero-categorical one
— the pattern below runs cleanly on both.

**Only build this after a solid single-model baseline is already submitted** (see
`system.md`'s workflow) — reuse that same feature set here rather than expanding it further.
The recipe's own evidence is explicit: the source solutions' advantage came from model
diversity, not feature volume.

**Honest expectation, not oversold**: on both datasets tested here, the blend beat 3 of 4
solo models but not the single best one (e.g. one dataset: meta-blend 0.9054 vs. solo
CatBoost 0.9065). This matches the source research's own nuance — a lean ensemble is
*competitive with*, not automatically superior to, the best single model on any one dataset.
The real value is consistency across many different datasets/draws, which a single "best"
model choice can't guarantee in advance.

## 1. Shared folds across every model — the one non-negotiable part

Use the **same** `StratifiedKFold` split and seed for every model in the blend. Mixing
different fold/seed setups per model is a cheap, easy-to-avoid mistake that widens the
CV-to-public gap (a specific, named caution from the source research, not a hypothetical).

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
```

## 2. Three GBDT models + one linear model — diversity is the point

Three tree-based models plus a `LogisticRegression` — the linear leg is the actual diversity
lever (a genuinely different model family), not a 4th tree variant. Reuse the category-dtype
pattern from `category_dtype_safety.md` for XGBoost/LightGBM; CatBoost takes raw string
categoricals directly (see that same reference for why).

**The linear model needs different feature prep than the GBDTs** — this is the concrete
detail the source research doesn't spell out: it can't take raw categorical-dtype columns the
way the GBDTs can, and its numeric inputs should be scaled. Feed it the native numeric
columns plus the frequency/target-encoded columns (already numeric, from
`leak_safe_encoding.md`) — not the raw categorical columns — scaled with `StandardScaler` fit
on the fold's training split only (same leak-safe discipline as encoding).

```python
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def fit_predict_gbdt(model_name, X_tr, y_tr, X_val, cat_cols):
    X_tr, X_val = X_tr.copy(), X_val.copy()
    if model_name == "catboost":
        for c in cat_cols:
            X_tr[c] = X_tr[c].astype(str)
            X_val[c] = X_val[c].astype(str)
        model = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05,
                                    random_seed=0, verbose=False, cat_features=cat_cols)
        model.fit(X_tr, y_tr)
        return model.predict_proba(X_val)[:, 1], model
    for c in cat_cols:
        X_tr[c] = X_tr[c].astype("category")
        X_val[c] = X_val[c].astype(X_tr[c].dtype)  # category_dtype_safety.md pattern
    if model_name == "xgboost":
        model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                   enable_categorical=True, random_state=0, n_jobs=1,
                                   eval_metric="auc")
    elif model_name == "lightgbm":
        model = lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                    random_state=0, n_jobs=1, verbosity=-1)
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_val)[:, 1], model


def fit_predict_linear(X_tr, y_tr, X_val):
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)  # fit on this fold's training split only
    X_val_scaled = scaler.transform(X_val)
    model = LogisticRegression(max_iter=1000, random_state=0)
    model.fit(X_tr_scaled, y_tr)
    return model.predict_proba(X_val_scaled)[:, 1], model, scaler
```

## 3. Collect OOF predictions over the shared folds

```python
MODEL_NAMES = ["xgboost", "lightgbm", "catboost", "linear"]
oof = {m: np.zeros(len(train_df)) for m in MODEL_NAMES}

for tr_idx, val_idx in skf.split(train_df, y):
    fold_train, fold_val = train_df.iloc[tr_idx].copy(), train_df.iloc[val_idx].copy()
    y_tr = y.iloc[tr_idx]

    # leak_safe_encoding.md pattern: fit only on this fold's training split
    enc_cols = encode_categoricals(fold_train, y_tr, [fold_train, fold_val], cat_like_cols)

    X_tr_gbdt = fold_train[native_num_cols + enc_cols + obj_cols]
    X_val_gbdt = fold_val[native_num_cols + enc_cols + obj_cols]
    for m in ["xgboost", "lightgbm", "catboost"]:
        preds, _ = fit_predict_gbdt(m, X_tr_gbdt, y_tr, X_val_gbdt, obj_cols)
        oof[m][val_idx] = preds

    linear_cols = native_num_cols + enc_cols  # numeric-only, no raw categoricals
    preds, _, _ = fit_predict_linear(fold_train[linear_cols], y_tr, fold_val[linear_cols])
    oof["linear"][val_idx] = preds
```

## 4. Correlation-based dedup — cheap pool-cutting, not a full search

If you end up with more candidate models than 3-4 (e.g. after trying variants per model
type), drop a candidate whose OOF predictions correlate above ~0.999 with one you're already
keeping — it isn't adding diversity, just cost. In both datasets this was verified against,
no pair exceeded ~0.98, so this rarely triggers with the base 4 — it matters more if you add
further variants.

```python
oof_df = pd.DataFrame(oof)
corr = oof_df.corr()
to_drop = [
    b for i, a in enumerate(MODEL_NAMES) for b in MODEL_NAMES[i + 1:]
    if corr.loc[a, b] > 0.999
]
```

## 5. Rank-transform, then try both blend methods — keep whichever scores better

Evidence on linear-meta-learner vs. plain average is genuinely mixed in the source research.
Both are cheap once you have OOF predictions, so compute both and keep whichever actually
scores better on your own data rather than assuming one wins. **Don't use a nonlinear
meta-learner** (GBDT/NN as the blender) — that was clearly evidenced as worse than linear
methods in the source research.

```python
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ranked_oof = pd.DataFrame({m: rankdata(oof[m]) / len(oof[m]) for m in MODEL_NAMES})

avg_blend_auc = roc_auc_score(y, ranked_oof.mean(axis=1))

meta = LogisticRegression(max_iter=1000, random_state=0)
meta.fit(ranked_oof, y)
meta_blend_auc = roc_auc_score(y, meta.predict_proba(ranked_oof)[:, 1])

# keep whichever of avg_blend_auc / meta_blend_auc is higher
```

## 6. Full-train refit for test predictions

Refit every model once on the full training set (same one-time-refit discipline as
`leak_safe_encoding.md`), predict on test, rank-transform each model's test predictions
independently, then apply whichever blend method (average or the fitted meta-learner) won in
step 5.

```python
enc_cols_full = encode_categoricals(train_df, y, [train_df, test_df], cat_like_cols)
test_preds = {}
X_full_gbdt = train_df[native_num_cols + enc_cols_full + obj_cols]
X_test_gbdt = test_df[native_num_cols + enc_cols_full + obj_cols]
for m in ["xgboost", "lightgbm", "catboost"]:
    preds, _ = fit_predict_gbdt(m, X_full_gbdt, y, X_test_gbdt, obj_cols)
    test_preds[m] = preds
linear_cols_full = native_num_cols + enc_cols_full
preds, _, _ = fit_predict_linear(train_df[linear_cols_full], y, test_df[linear_cols_full])
test_preds["linear"] = preds

ranked_test = pd.DataFrame({m: rankdata(test_preds[m]) / len(test_preds[m]) for m in MODEL_NAMES})
final_blend = ranked_test.mean(axis=1) if avg_blend_auc >= meta_blend_auc else meta.predict_proba(ranked_test)[:, 1]
```

## Stopping rule specific to this pattern

If adding another model to the blend improves CV but the resulting submission's public score
doesn't follow, stop adding models — this is the same leak/overfitting signal from the
selection rule in `system.md`, applied specifically to ensemble size rather than to a single
model's feature set.
