#!/usr/bin/env python3
"""Lean GBDT baseline: fast, CPU-only, single-model, CV-safe target encoding.

Loads train/test, engineers a capped set of cheap features (ratio/interaction
and polynomial terms on the columns most correlated with the target, decimal
digit / fractional-residue features on all numeric columns, multi-scale
binning on high-cardinality numeric columns, and binary-flag-family
aggregation when detected), then fits one GBDT (whichever of
xgboost/lightgbm/catboost is importable, else sklearn's
HistGradientBoostingClassifier) inside stratified K-fold CV with frequency +
smoothed target encoding refit per fold to avoid leakage. Prints an EDA
summary, the out-of-fold ROC-AUC as VALIDATION_SCORE, and writes a
submission-ready CSV from a model refit on the full training set.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Column-by-column feature engineering deliberately fragments these frames;
# the resulting PerformanceWarning spam would otherwise burn through the
# agent's max_stdout_chars budget and could bury the VALIDATION_SCORE line.
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

SEED = 0


# --------------------------------------------------------------------------- #
# GBDT backend selection
# --------------------------------------------------------------------------- #

def get_backend(name="auto"):
    """Return (backend_name, fit_predict_fn). fit_predict_fn(X_tr, y_tr, X_val,
    cat_cols, params) -> val_proba (used for both OOF and final test predict)."""

    def try_xgboost():
        import xgboost as xgb

        def fit_predict(X_tr, y_tr, X_val, cat_cols, depth):
            for c in cat_cols:
                X_tr[c] = X_tr[c].astype("category")
                # Reuse train's exact category set (not an independent cast) so a
                # test-only category becomes NaN (a normal missing value to the
                # model) instead of crashing at predict time on an unseen level.
                X_val[c] = X_val[c].astype(X_tr[c].dtype)
            model = xgb.XGBClassifier(
                n_estimators=400,
                max_depth=depth,
                learning_rate=0.05,
                subsample=0.5,
                colsample_bytree=0.2,
                reg_lambda=15,
                reg_alpha=10,
                enable_categorical=True,
                tree_method="hist",
                eval_metric="auc",
                random_state=SEED,
                n_jobs=1,
            )
            model.fit(X_tr, y_tr)
            return model.predict_proba(X_val)[:, 1]

        return fit_predict

    def try_lightgbm():
        import lightgbm as lgb

        def fit_predict(X_tr, y_tr, X_val, cat_cols, depth):
            for c in cat_cols:
                X_tr[c] = X_tr[c].astype("category")
                # Reuse train's exact category set (not an independent cast) so a
                # test-only category becomes NaN (a normal missing value to the
                # model) instead of crashing at predict time on an unseen level.
                X_val[c] = X_val[c].astype(X_tr[c].dtype)
            model = lgb.LGBMClassifier(
                n_estimators=400,
                max_depth=depth,
                num_leaves=max(2 ** depth // 2, 4),
                learning_rate=0.05,
                subsample=0.5,
                colsample_bytree=0.2,
                reg_lambda=15,
                reg_alpha=10,
                random_state=SEED,
                n_jobs=1,
                verbosity=-1,
            )
            model.fit(X_tr, y_tr, categorical_feature=cat_cols if cat_cols else "auto")
            return model.predict_proba(X_val)[:, 1]

        return fit_predict

    def try_catboost():
        from catboost import CatBoostClassifier

        def fit_predict(X_tr, y_tr, X_val, cat_cols, depth):
            for c in cat_cols:
                X_tr[c] = X_tr[c].astype(str)
                X_val[c] = X_val[c].astype(str)
            model = CatBoostClassifier(
                iterations=400,
                depth=depth,
                learning_rate=0.05,
                subsample=0.5,
                colsample_bylevel=0.2,
                l2_leaf_reg=15,
                random_state=SEED,
                thread_count=1,
                verbose=False,
            )
            model.fit(X_tr, y_tr, cat_features=cat_cols if cat_cols else None)
            return model.predict_proba(X_val)[:, 1]

        return fit_predict

    def sklearn_fallback():
        from sklearn.ensemble import HistGradientBoostingClassifier

        def fit_predict(X_tr, y_tr, X_val, cat_cols, depth):
            for c in cat_cols:
                X_tr[c] = X_tr[c].astype("category")
                # Reuse train's exact category set (not an independent cast) so a
                # test-only category becomes NaN (a normal missing value to the
                # model) instead of crashing at predict time on an unseen level.
                X_val[c] = X_val[c].astype(X_tr[c].dtype)
            model = HistGradientBoostingClassifier(
                max_depth=depth,
                learning_rate=0.05,
                l2_regularization=15.0,
                categorical_features=cat_cols if cat_cols else None,
                random_state=SEED,
            )
            model.fit(X_tr, y_tr)
            return model.predict_proba(X_val)[:, 1]

        return fit_predict

    candidates = {
        "xgboost": try_xgboost,
        "lightgbm": try_lightgbm,
        "catboost": try_catboost,
        "hist_gbm": sklearn_fallback,
    }

    if name != "auto":
        return name, candidates[name]()

    order = ["xgboost", "lightgbm", "catboost", "hist_gbm"]
    for candidate_name in order:
        try:
            fn = candidates[candidate_name]()
            return candidate_name, fn
        except ImportError:
            continue
    raise RuntimeError("No GBDT backend available (tried xgboost, lightgbm, catboost, hist_gbm)")


def read_csv_safe(path):
    """pandas' default C/python parsers have been observed to *segfault*
    (not raise) on some files in this dataset family -- a crash that can't be
    caught from Python and kills the whole session. The pyarrow engine reads
    the same files correctly (verified byte-for-byte equivalent on a normal
    file), so use it whenever available."""
    try:
        return pd.read_csv(path, engine="pyarrow")
    except ImportError:
        return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# EDA summary (cheap, printed only)
# --------------------------------------------------------------------------- #

def print_eda_summary(train_df, test_df, target_col, id_col):
    print("=== EDA summary ===")
    y = train_df[target_col]
    print(f"Target balance: {y.value_counts(normalize=True).round(4).to_dict()}")

    feat_cols = [c for c in train_df.columns if c not in (target_col, id_col)]
    obj_cols = train_df[feat_cols].select_dtypes(exclude=[np.number]).columns.tolist()
    num_cols = train_df[feat_cols].select_dtypes(include=[np.number]).columns.tolist()
    print(f"Numeric columns: {len(num_cols)}, categorical (object) columns: {len(obj_cols)}")
    for c in obj_cols:
        print(f"  {c}: cardinality={train_df[c].nunique()}")

    miss = train_df[feat_cols].isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if len(miss):
        print("Missing value fraction (train, nonzero only):")
        for c, frac in miss.items():
            print(f"  {c}: {frac:.3f}")
    else:
        print("No missing values in train.")

    common_feat = [c for c in feat_cols if c in test_df.columns]
    train_keys = train_df[common_feat].astype(str).agg("|".join, axis=1)
    test_keys = test_df[common_feat].astype(str).agg("|".join, axis=1)
    dup_count = train_keys.isin(set(test_keys)).sum()
    print(f"Train rows with an identical feature-row also present in test: {dup_count} / {len(train_df)}")
    print()


# --------------------------------------------------------------------------- #
# Feature engineering (leak-free: no per-row target usage outside the CV loop)
# --------------------------------------------------------------------------- #

def add_ratio_and_poly_features(train_df, test_df, num_cols, y, top_k=5, max_pairs=6):
    if len(num_cols) < 2:
        return [], []
    corr = train_df[num_cols].apply(lambda col: col.corr(y)).abs().sort_values(ascending=False)
    top_cols = corr.head(top_k).index.tolist()

    poly_names = []
    for c in top_cols:
        name = f"{c}__sq"
        train_df[name] = train_df[c] ** 2
        test_df[name] = test_df[c] ** 2
        poly_names.append(name)

    ratio_names = []
    pair_count = 0
    eps = 1e-6
    for i, a in enumerate(top_cols):
        for b in top_cols[i + 1:]:
            if pair_count >= max_pairs:
                break
            ratio_name = f"{a}__div__{b}"
            inter_name = f"{a}__x__{b}"
            train_df[ratio_name] = train_df[a] / (train_df[b].abs() + eps)
            test_df[ratio_name] = test_df[a] / (test_df[b].abs() + eps)
            train_df[inter_name] = train_df[a] * train_df[b]
            test_df[inter_name] = test_df[a] * test_df[b]
            ratio_names += [ratio_name, inter_name]
            pair_count += 1
        if pair_count >= max_pairs:
            break

    return poly_names, ratio_names


def add_digit_fraction_features(train_df, test_df, num_cols):
    """Decimal-digit and fractional-residue features — treated as core per the
    recipe (independently converged on across multiple unrelated solutions,
    a likely structural artifact of the synthetic data generator)."""
    new_cols = []
    fractions = {"half": 0.5, "quarter": 0.25, "fifth": 0.2, "tenth": 0.1}
    for c in num_cols:
        frac = train_df[c] % 1
        frac_test = test_df[c] % 1

        d1 = f"{c}__digit1"
        train_df[d1] = (frac * 10 % 10).apply(np.floor)
        test_df[d1] = (frac_test * 10 % 10).apply(np.floor)
        new_cols.append(d1)

        d2 = f"{c}__digit2"
        train_df[d2] = (frac * 100 % 10).apply(np.floor)
        test_df[d2] = (frac_test * 100 % 10).apply(np.floor)
        new_cols.append(d2)

        round_flag = f"{c}__is_round"
        train_df[round_flag] = (frac.abs() < 1e-6).astype(int)
        test_df[round_flag] = (frac_test.abs() < 1e-6).astype(int)
        new_cols.append(round_flag)

        for fname, fval in fractions.items():
            resid_name = f"{c}__resid_{fname}"
            train_df[resid_name] = (frac % fval).clip(upper=fval - (frac % fval))
            test_df[resid_name] = (frac_test % fval).clip(upper=fval - (frac_test % fval))
            new_cols.append(resid_name)

    return new_cols


def add_group_flag_aggregation(train_df, test_df, num_cols):
    """If >=3 numeric columns are binary {0,1}-valued, treat them as a family
    of yes/no flags and add count / has-any / has-all aggregates."""
    binary_cols = [
        c for c in num_cols
        if train_df[c].dropna().isin([0, 1]).all() and train_df[c].nunique() <= 2
    ]
    if len(binary_cols) < 3:
        return []
    for df in (train_df, test_df):
        df["flag_group__sum"] = df[binary_cols].sum(axis=1)
        df["flag_group__has_any"] = (df[binary_cols].sum(axis=1) > 0).astype(int)
        df["flag_group__has_all"] = (df[binary_cols].sum(axis=1) == len(binary_cols)).astype(int)
    return ["flag_group__sum", "flag_group__has_any", "flag_group__has_all"]


def add_multiscale_bins(train_df, test_df, num_cols, high_card_threshold=20, n_bins=10):
    """Bin high-cardinality numeric columns multiple ways; the bin labels are
    later target/frequency-encoded like any other categorical-like column.
    Kept to modest bin counts (tens, not thousands) per the recipe."""
    bin_cols = []
    for c in num_cols:
        if train_df[c].nunique() < high_card_threshold:
            continue

        q_name = f"{c}__qbin"
        try:
            _, edges = pd.qcut(train_df[c], q=n_bins, retbins=True, duplicates="drop")
        except ValueError:
            continue
        train_df[q_name] = pd.cut(train_df[c], bins=edges, include_lowest=True).astype(str)
        test_df[q_name] = pd.cut(test_df[c], bins=edges, include_lowest=True).astype(str)
        bin_cols.append(q_name)

        w_name = f"{c}__wbin"
        lo, hi = train_df[c].min(), train_df[c].max()
        edges_w = np.linspace(lo, hi, n_bins + 1)
        train_df[w_name] = pd.cut(train_df[c], bins=edges_w, include_lowest=True).astype(str)
        test_df[w_name] = pd.cut(test_df[c], bins=edges_w, include_lowest=True).astype(str)
        bin_cols.append(w_name)

        scale = (hi - lo) / n_bins if hi > lo else 1.0
        r_name = f"{c}__roundbin"
        train_df[r_name] = (train_df[c] / scale).round().astype(int) if scale else 0
        test_df[r_name] = (test_df[c] / scale).round().astype(int) if scale else 0
        bin_cols.append(r_name)

        t_name = f"{c}__truncbin"
        train_df[t_name] = np.floor(train_df[c]).astype(int)
        test_df[t_name] = np.floor(test_df[c]).astype(int)
        bin_cols.append(t_name)

    return bin_cols


def cardinality_based_split(train_df, cat_like_candidates, low_card_threshold=10):
    """Numeric columns with few unique values are treated as categorical-like
    for encoding purposes, alongside true object-dtype columns."""
    return [c for c in cat_like_candidates if train_df[c].nunique() < low_card_threshold]


# --------------------------------------------------------------------------- #
# CV-safe frequency + smoothed target encoding
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Lean GBDT baseline (Recipe 1).")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--sample-submission", default="sample_submission.csv")
    parser.add_argument("--target", default="target")
    parser.add_argument("--id-col", default="row_id")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument(
        "--model", default="auto",
        choices=["auto", "xgboost", "lightgbm", "catboost", "hist_gbm"],
    )
    parser.add_argument(
        "--second-model", default="none",
        choices=["none", "xgboost", "lightgbm", "catboost", "hist_gbm"],
        help="If set, trains a second backend and averages test predictions 50/50.",
    )
    parser.add_argument(
        "--features", default="all",
        help="Comma-separated subset of {ratio_poly,digit_frac,group_flag,multiscale_bin} or 'all'/'none'.",
    )
    args = parser.parse_args()

    for path in (args.train, args.test):
        if not os.path.exists(path):
            print(f"Error: '{path}' not found.")
            sys.exit(1)

    train_df = read_csv_safe(args.train)
    test_df = read_csv_safe(args.test)
    y = train_df[args.target]
    test_ids = test_df[args.id_col].copy() if args.id_col in test_df.columns else None

    print_eda_summary(train_df, test_df, args.target, args.id_col)

    feat_cols = [c for c in train_df.columns if c not in (args.target, args.id_col)]
    feat_cols = [c for c in feat_cols if c in test_df.columns]
    train_df = train_df[feat_cols].copy()
    test_df = test_df[feat_cols].copy()

    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    obj_cols = train_df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Impute before any feature engineering so downstream math never sees NaN.
    for c in num_cols:
        median = train_df[c].median()
        train_df[c] = train_df[c].fillna(median)
        test_df[c] = test_df[c].fillna(median)
    for c in obj_cols:
        mode = train_df[c].mode().iloc[0] if not train_df[c].mode().empty else "missing"
        train_df[c] = train_df[c].fillna(mode)
        test_df[c] = test_df[c].fillna(mode)

    active = set(args.features.split(",")) if args.features != "none" else set()
    if args.features == "all":
        active = {"ratio_poly", "digit_frac", "group_flag", "multiscale_bin"}

    engineered_num_cols = []
    if "ratio_poly" in active:
        poly_cols, ratio_cols = add_ratio_and_poly_features(train_df, test_df, num_cols, y)
        engineered_num_cols += poly_cols + ratio_cols
        print(f"Added {len(poly_cols)} polynomial + {len(ratio_cols)} ratio/interaction features.")
    if "digit_frac" in active:
        digit_cols = add_digit_fraction_features(train_df, test_df, num_cols)
        engineered_num_cols += digit_cols
        print(f"Added {len(digit_cols)} decimal-digit/fractional-residue features.")
    if "group_flag" in active:
        flag_cols = add_group_flag_aggregation(train_df, test_df, num_cols)
        engineered_num_cols += flag_cols
        if flag_cols:
            print(f"Added {len(flag_cols)} group-level binary-flag aggregation features.")

    bin_cols = []
    if "multiscale_bin" in active:
        bin_cols = add_multiscale_bins(train_df, test_df, num_cols)
        print(f"Added {len(bin_cols)} multi-scale binned columns (to be target/freq-encoded).")

    low_card_numeric = cardinality_based_split(train_df, num_cols)
    cat_like_cols = obj_cols + low_card_numeric + bin_cols
    cat_like_cols = list(dict.fromkeys(cat_like_cols))  # de-dup, keep order
    print(f"Categorical-like columns for encoding: {len(cat_like_cols)} "
          f"({len(obj_cols)} object + {len(low_card_numeric)} low-cardinality numeric + {len(bin_cols)} binned)")

    native_num_cols = [c for c in num_cols if c not in low_card_numeric] + engineered_num_cols

    def run_backend(model_name):
        backend_name, fit_predict = get_backend(model_name)
        print(f"Using GBDT backend: {backend_name}")

        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
        oof = np.zeros(len(train_df))
        fold_scores = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df, y)):
            fold_train = train_df.iloc[tr_idx].copy()
            fold_val = train_df.iloc[val_idx].copy()
            y_tr = y.iloc[tr_idx]

            enc_cols = encode_categoricals(fold_train, y_tr, [fold_train, fold_val], cat_like_cols)

            X_tr = fold_train[native_num_cols + enc_cols + obj_cols].copy()
            X_val = fold_val[native_num_cols + enc_cols + obj_cols].copy()
            oof[val_idx] = fit_predict(X_tr, y_tr, X_val, obj_cols, args.depth)
            fold_auc = roc_auc_score(y.iloc[val_idx], oof[val_idx])
            fold_scores.append(fold_auc)
            print(f"  fold {fold}: AUC={fold_auc:.4f}")

        cv_score = roc_auc_score(y, oof)
        cv_std = float(np.std(fold_scores))
        print(f"Backend {backend_name} OOF ROC-AUC: {cv_score:.4f} (fold mean {np.mean(fold_scores):.4f} +/- {cv_std:.4f})")

        # Refit on full train for the actual test predictions.
        full_train = train_df.copy()
        full_test = test_df.copy()
        enc_cols = encode_categoricals(full_train, y, [full_train, full_test], cat_like_cols)
        X_full = full_train[native_num_cols + enc_cols + obj_cols].copy()
        X_test = full_test[native_num_cols + enc_cols + obj_cols].copy()
        test_proba = fit_predict(X_full, y, X_test, obj_cols, args.depth)

        return cv_score, cv_std, oof, test_proba

    cv_score_1, cv_std_1, oof_1, test_proba_1 = run_backend(args.model)

    if args.second_model != "none":
        cv_score_2, cv_std_2, oof_2, test_proba_2 = run_backend(args.second_model)
        blend_oof = (oof_1 + oof_2) / 2
        blend_cv = roc_auc_score(y, blend_oof)
        # Recompute fold-wise std for the blend using the same deterministic
        # fold split (same n_splits/seed/y => identical assignment as above).
        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
        blend_fold_scores = [
            roc_auc_score(y.iloc[val_idx], blend_oof[val_idx])
            for _, val_idx in skf.split(train_df, y)
        ]
        blend_std = float(np.std(blend_fold_scores))
        print(f"Second backend OOF ROC-AUC: {cv_score_2:.4f} (+/- {cv_std_2:.4f})")
        print(f"50/50 blend OOF ROC-AUC: {blend_cv:.4f} (+/- {blend_std:.4f})")
        final_cv, final_cv_std = blend_cv, blend_std
        final_test_proba = (test_proba_1 + test_proba_2) / 2
    else:
        final_cv, final_cv_std = cv_score_1, cv_std_1
        final_test_proba = test_proba_1

    sample_sub = read_csv_safe(args.sample_submission)
    id_col_in_sub = [c for c in sample_sub.columns if c != args.target][0]
    if test_ids is None:
        test_ids = sample_sub[id_col_in_sub]
    submission = pd.DataFrame({id_col_in_sub: test_ids, args.target: final_test_proba})
    submission = submission[sample_sub.columns]
    submission.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(submission)} rows)")

    print(f"VALIDATION_SCORE: {final_cv:.6f}")
    print(f"VALIDATION_STD: {final_cv_std:.6f}")


if __name__ == "__main__":
    main()
