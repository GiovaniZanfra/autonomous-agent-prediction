---
name: feature-engineer
description: >-
  Provides a robust Python script for automated feature generation.
  Handles missing value imputation, schema-agnostic ordinal/nominal
  categorical encoding, and numeric aggregations.
---

# Feature Engineer Skill

This skill equips the agent with a pre-packaged Python CLI script for automated feature engineering.

## Available Scripts

### 1. `generate_features.py`
Automatically identifies column types, imputes missing values, encodes
categorical/ordinal columns to numeric, and calculates row mean over the
original numeric columns.

**Column typing is schema-agnostic**: the dataset family does not keep a fixed
feature-to-type mapping (e.g. `feature_1` is `ordinal` in one dataset and
`count` in another). Each non-numeric column is typed by inspecting its
*values*, not its name or position:
- If every observed value matches `ord_<int>`, it's treated as **ordinal** and
  encoded to the trailing integer (preserves order).
- Otherwise it's treated as **nominal** and label-encoded via a fitted
  train-derived mapping (unseen test categories map to `-1`).

**Usage via `run_skill_script`**:
```python
run_skill_script(
    skill_name="feature_engineer",
    script_name="generate_features.py",
    args="--train train.csv --test test.csv --target target",
)
```
**Arguments**:
- `--train`: Path to training CSV (default: `train.csv`).
- `--test`: Path to test CSV (default: `test.csv`).
- `--target`: Name of the target column (default: `target`).
- `--id-col`: Name of the row identifier column to pass through untouched,
  excluded from imputation/encoding/row_mean (default: `row_id`).

**Outputs**: Creates `train_engineered.csv` and `test_engineered.csv`, both
fully numeric except for the identifier column — ready to feed directly into
an sklearn estimator.

---

## Domain Knowledge Resources

### `leakage_checklist.md`
A concise guide on preventing data leakage during feature engineering. You can read it using the `load_skill_resource` tool:
```python
load_skill_resource(
    skill_name="feature_engineer",
    resource_name="leakage_checklist.md",
)
```
