---
name: ml-safety-patterns
description: >-
  Reference code patterns (not a script to run) for extensive, leak-safe
  feature engineering and category-handling on this dataset family. Read
  these before writing your own modeling code, then adapt them directly.
---

# ML Safety Patterns Skill

**This skill has no scripts.** Unlike other skills you may have seen, there is nothing to
invoke via `run_skill_script`. Instead, this skill gives you vetted, working code *patterns*
to read via `load_skill_resource` and adapt directly into your own hand-written script (which
you create with `write_file` and run with `run_command`, same as any other code you write).

The patterns here come from a script (`train_baseline.py`, from a prior submission) that was
heavily tested across all 16 local practice datasets, plus a 256-run deterministic sweep of
every feature-technique combination against real held-out ground truth. Some of these
patterns exist specifically because they caused real, confirmed bugs when done naively —
reuse them rather than reimplementing from scratch. The feature-engineering guidance below is
evidence-based, not "apply everything" — read it before assuming more is better.

## Available Resources

### `feature_engineering_patterns.md`
Four core feature-engineering technique families, each measured directly (not assumed)
against all 16 local practice datasets, plus four further additions (missing-indicator flags,
cardinality-based categorical split, cardinality-based frequency encoding, categorical
cross-features) that are standard/low-risk but not part of that specific measurement.
**`ratio_poly` is the empirically best core default; `digit_frac` and `multiscale_bin` are
not defaults** — applying all four core techniques together had the *worst* average rank of
every combination tested, and `multiscale_bin` specifically hurts both accuracy and
cross-dataset consistency when stacked onto one model (though see the file's note on using it
for *representation diversity across ensemble members* instead — a different, untested
mechanism). See the file for the full ranking and when the non-default two are still worth
trying (concrete EDA justification + your own CV verification).

### `leak_safe_encoding.md`
The per-fold-refit frequency/target encoding pattern. **Non-negotiable regardless of how
extensive your feature set gets**: any categorical or high-cardinality-numeric column you
encode against the target must be refit on each CV fold's training split only, never on the
full training set before splitting — fitting on the full set leaks target information into
validation.

### `category_dtype_safety.md`
The exact fix for a real crash found this project: casting train/validation categorical
columns to `"category"` dtype independently can crash GBDT libraries on categories only seen
in validation/test. Read this before handling any categorical column with XGBoost, LightGBM,
or scikit-learn's `HistGradientBoostingClassifier`.

### `leakage_checklist.md`
General data leakage checklist — target leakage and train/test-distribution leakage.

### `diverse_ensemble.md`
The Recipe 2 pattern: 3-6 genuinely diverse models (default: whichever GBDT backends a
probe-at-the-start finds actually importable — up to XGBoost/LightGBM/CatBoost/
HistGradientBoostingClassifier — plus one linear model) blended via OOF rank-transform, with
the same-folds discipline, correlation-based dedup (>0.9999, only near-exact duplicates), a
multi-seed OOF comparison for deciding between blend methods/hyperparameters (so small
differences aren't mistaken for signal), and a blend-specific stopping rule. Build this
*after* a single-model baseline is already submitted — see `system.md`'s workflow for when.

Read all five via
`load_skill_resource(skill_name="ml-safety-patterns", file_path="references/<name>.md")`
(e.g. `file_path="references/feature_engineering_patterns.md"`) before writing your first
modeling script.
