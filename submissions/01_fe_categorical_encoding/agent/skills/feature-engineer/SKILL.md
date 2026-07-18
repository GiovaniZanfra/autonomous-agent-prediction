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
    skill_name="feature-engineer",
    script_name="generate_features.py",
    args="--train /work/train.csv --test /work/test.csv --target target --output-dir /work",
)
```
**IMPORTANT — always use absolute `/work/...` paths for `--train`/`--test`/
`--output-dir`, never relative ones.** `run_skill_script` materializes the skill's own
files into a fresh temporary directory and runs the script there — that directory does
NOT contain your problem's `train.csv`/`test.csv`, only this skill's bundled scripts, and
it is deleted the moment the tool call returns. A relative `--train` path will fail with
"not found" even though the file exists in `/work`; and if you omit `--output-dir /work`,
the engineered CSVs will be written into that temporary directory and vanish before you
can read them back with `run_command` or feed them to a model.

**Arguments**:
- `--train`: Path to training CSV (default: `train.csv` — override with `/work/train.csv`).
- `--test`: Path to test CSV (default: `test.csv` — override with `/work/test.csv`).
- `--target`: Name of the target column (default: `target`).
- `--id-col`: Name of the row identifier column to pass through untouched,
  excluded from imputation/encoding/row_mean (default: `row_id`).
- `--output-dir`: Directory to write `train_engineered.csv`/`test_engineered.csv` to
  (default: `.` — always override with `/work` so the files persist after the tool call
  returns).

**Outputs**: Creates `train_engineered.csv` and `test_engineered.csv` in `--output-dir`,
both fully numeric except for the identifier column — ready to feed directly into
an sklearn estimator.

---

## Domain Knowledge Resources

### `leakage_checklist.md`
A concise guide on preventing data leakage during feature engineering. You can read it using the `load_skill_resource` tool:
```python
load_skill_resource(
    skill_name="feature-engineer",
    resource_name="leakage_checklist.md",
)
```
