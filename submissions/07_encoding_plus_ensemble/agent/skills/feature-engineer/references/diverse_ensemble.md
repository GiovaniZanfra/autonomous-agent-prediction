# Diverse Model Ensemble (Recipe 2 pattern), on top of `generate_features.py` output

The highest-consensus finding across the wider research this project draws from: train
several genuinely diverse models, blend their out-of-fold (OOF) predictions.

**Only build this after a solid single-model submission using `generate_features.py`'s
output already exists.** Reuse that exact engineered feature set for every model in the
blend — don't add further feature engineering here. The recipe's own evidence is that model
diversity is the lever, not feature volume.

**Honest expectation, not oversold**: a lean ensemble is typically *competitive with*, not
automatically superior to, the single best model on any one dataset. Its real value is
consistency across draws from the family, which a single "best model" pick can't guarantee
in advance — submit it and compare against your best solo model rather than assuming it wins.

## Why this is simpler than a from-scratch ensemble

`generate_features.py` already imputes and label/ordinal-encodes every non-numeric column
into `train_engineered.csv`/`test_engineered.csv` — the output is fully numeric (except the
id column). That encoding never looks at the target (pure value-pattern based), so unlike
target/frequency encoding it does **not** need to be refit per-fold — it was already fit once,
safely, when you ran the script. That means every model below, including the linear leg, can
read the same engineered CSV directly: no native-categorical handling, no per-fold re-encoding.

## 1. Shared folds across every model — the one non-negotiable part

```python
import pandas as pd
from sklearn.model_selection import StratifiedKFold

train_df = pd.read_csv("/work/train_engineered.csv")
test_df = pd.read_csv("/work/test_engineered.csv")
y = train_df.pop("target")
id_col = "row_id"
feature_cols = [c for c in train_df.columns if c != id_col]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
```

Use the **same** split and seed for every model in the blend — mixing per-model fold/seed
setups is a cheap, easy-to-avoid mistake that widens the CV-to-public gap.

## 2. Three GBDT models + one linear model — diversity is the point

The linear leg is the actual diversity lever (a genuinely different model family), not a 4th
tree variant. Because the input is already fully numeric, only the linear leg needs extra
prep (`StandardScaler`, fit on the fold's training split only).

```python
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def fit_predict(model_name, X_tr, y_tr, X_val):
    if model_name == "xgboost":
        model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                   random_state=0, n_jobs=1, eval_metric="auc")
        model.fit(X_tr, y_tr)
        return model.predict_proba(X_val)[:, 1], model
    if model_name == "lightgbm":
        model = lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                    random_state=0, n_jobs=1, verbosity=-1)
        model.fit(X_tr, y_tr)
        return model.predict_proba(X_val)[:, 1], model
    if model_name == "catboost":
        model = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05,
                                    random_seed=0, verbose=False)
        model.fit(X_tr, y_tr)
        return model.predict_proba(X_val)[:, 1], model
    if model_name == "linear":
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)  # fit on this fold's training split only
        X_val_s = scaler.transform(X_val)
        model = LogisticRegression(max_iter=1000, random_state=0)
        model.fit(X_tr_s, y_tr)
        return model.predict_proba(X_val_s)[:, 1], (model, scaler)
```

## 3. Collect OOF predictions over the shared folds

```python
MODEL_NAMES = ["xgboost", "lightgbm", "catboost", "linear"]
oof = {m: np.zeros(len(train_df)) for m in MODEL_NAMES}

for tr_idx, val_idx in skf.split(train_df, y):
    X_tr, X_val = train_df.iloc[tr_idx][feature_cols], train_df.iloc[val_idx][feature_cols]
    y_tr = y.iloc[tr_idx]
    for m in MODEL_NAMES:
        preds, _ = fit_predict(m, X_tr, y_tr, X_val)
        oof[m][val_idx] = preds
```

## 4. Correlation-based dedup — cheap pool-cutting, not a full search

If you try further model variants and end up with more than 3-4 candidates, drop one whose
OOF predictions correlate above ~0.999 with a candidate you're already keeping — it isn't
adding diversity, just cost.

```python
oof_df = pd.DataFrame(oof)
corr = oof_df.corr()
to_drop = [
    b for i, a in enumerate(MODEL_NAMES) for b in MODEL_NAMES[i + 1:]
    if corr.loc[a, b] > 0.999
]
```

## 5. Rank-transform, then try both blend methods — keep whichever scores better

Evidence on linear-meta-learner vs. plain average is genuinely mixed. Both are cheap once
you have OOF predictions, so compute both and keep whichever actually scores better on your
own data. **Don't use a nonlinear meta-learner** (GBDT/NN as the blender).

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

```python
test_preds = {}
for m in MODEL_NAMES:
    preds, _ = fit_predict(m, train_df[feature_cols], y, test_df[feature_cols])
    test_preds[m] = preds

ranked_test = pd.DataFrame({m: rankdata(test_preds[m]) / len(test_preds[m]) for m in MODEL_NAMES})
final_blend = (ranked_test.mean(axis=1) if avg_blend_auc >= meta_blend_auc
               else meta.predict_proba(ranked_test)[:, 1])
```

## Stopping rule specific to this pattern

If adding another model to the blend improves CV but the resulting submission's public score
doesn't follow, stop adding models — the same leak/overfitting signal as the selection rule
in `system.md`, applied to ensemble size specifically.
