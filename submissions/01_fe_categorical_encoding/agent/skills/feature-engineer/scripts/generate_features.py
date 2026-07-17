#!/usr/bin/env python3
"""Robust CLI script for automated feature generation.

Imputes missing values and encodes every non-numeric column to numeric so the
output is directly model-ready, then adds a row-mean feature over the original
numeric columns. Categorical/ordinal typing is inferred from each column's
values (an `ord_<int>` pattern means ordinal, preserving order; anything else
is nominal) rather than from column name or position, because the dataset
family does not keep a fixed feature-to-type mapping across datasets.
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

ORDINAL_RE = re.compile(r"^ord_(\d+)$")


def fit_ordinal_map(train_col: pd.Series) -> dict:
    """Map each observed value to its trailing integer, preserving order."""
    return {v: int(ORDINAL_RE.match(v).group(1)) for v in train_col.unique()}


def fit_nominal_map(train_col: pd.Series) -> dict:
    """Map each observed value to a deterministic integer code."""
    return {v: i for i, v in enumerate(sorted(train_col.unique()))}


def main():
    parser = argparse.ArgumentParser(description="Generate automated ML features.")
    parser.add_argument("--train", type=str, default="train.csv", help="Path to train CSV")
    parser.add_argument("--test", type=str, default="test.csv", help="Path to test CSV")
    parser.add_argument("--target", type=str, default="target", help="Target column name")
    parser.add_argument(
        "--id-col",
        type=str,
        default="row_id",
        help="Identifier column to pass through untouched (default: row_id)",
    )
    args = parser.parse_args()

    print(f"Loading datasets: {args.train}, {args.test}...")
    if not os.path.exists(args.train):
        print(f"Error: Train file '{args.train}' not found.")
        sys.exit(1)
    if not os.path.exists(args.test):
        print(f"Error: Test file '{args.test}' not found.")
        sys.exit(1)

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    target_series = None
    if args.target in train_df.columns:
        target_series = train_df[args.target]
        train_df = train_df.drop(columns=[args.target])
    else:
        print(f"Warning: Target column '{args.target}' not found in train_df.")

    # Align columns
    common_cols = [c for c in train_df.columns if c in test_df.columns]
    train_df = train_df[common_cols].copy()
    test_df = test_df[common_cols].copy()

    print(f"Initial shape: train={train_df.shape}, test={test_df.shape}")

    # Set the identifier column aside — never impute/encode/aggregate it.
    id_train = train_df.pop(args.id_col) if args.id_col in train_df.columns else None
    id_test = test_df.pop(args.id_col) if args.id_col in test_df.columns else None

    # Identify column types
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = train_df.select_dtypes(exclude=[np.number]).columns.tolist()

    # 1. Impute missing values (fit on train, transform test)
    if num_cols:
        print(f"Imputing missing values for {len(num_cols)} numeric columns...")
        num_imputer = SimpleImputer(strategy="median")
        train_df[num_cols] = num_imputer.fit_transform(train_df[num_cols])
        test_df[num_cols] = num_imputer.transform(test_df[num_cols])

    if cat_cols:
        print(f"Imputing missing values for {len(cat_cols)} categorical columns...")
        cat_imputer = SimpleImputer(strategy="most_frequent")
        train_df[cat_cols] = cat_imputer.fit_transform(train_df[cat_cols])
        test_df[cat_cols] = cat_imputer.transform(test_df[cat_cols])

    # 2. Encode categorical/ordinal columns to numeric (schema-agnostic: typed
    # by value pattern, not column name/position — see module docstring).
    for col in cat_cols:
        train_values = train_df[col].astype(str)
        is_ordinal = train_values.map(lambda v: bool(ORDINAL_RE.match(v))).all()
        if is_ordinal:
            mapping = fit_ordinal_map(train_values)
            print(f"  {col}: ordinal, {len(mapping)} levels")
        else:
            mapping = fit_nominal_map(train_values)
            print(f"  {col}: nominal, {len(mapping)} categories")

        train_df[col] = train_values.map(mapping)
        test_values = test_df[col].astype(str)
        test_df[col] = test_values.map(mapping)

        if is_ordinal:
            # Unseen test category: fall back to the median train code.
            fallback = float(np.median(list(mapping.values())))
        else:
            # Unseen test category: reserved sentinel below the observed range.
            fallback = -1
        test_df[col] = test_df[col].fillna(fallback)

    # 3. Aggregation Features (over the *original* numeric columns only)
    if len(num_cols) > 0:
        print("Calculating row-wise mean feature...")
        train_df["row_mean"] = train_df[num_cols].mean(axis=1)
        test_df["row_mean"] = test_df[num_cols].mean(axis=1)

    # Re-attach identifier and target
    if id_train is not None:
        train_df.insert(0, args.id_col, id_train)
        test_df.insert(0, args.id_col, id_test)
    if target_series is not None:
        train_df[args.target] = target_series

    print(f"Engineered shape: train={train_df.shape}, test={test_df.shape}")
    train_df.to_csv("train_engineered.csv", index=False)
    test_df.to_csv("test_engineered.csv", index=False)
    print("Saved train_engineered.csv and test_engineered.csv successfully.")


if __name__ == "__main__":
    main()
