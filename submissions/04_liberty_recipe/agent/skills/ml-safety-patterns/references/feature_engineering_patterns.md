# Feature Engineering Patterns

Four technique families, each independently tested against all 16 local practice datasets
(256 total runs, every combination of the four, scored against real held-out ground truth —
not a guess). **The result was not "apply generously" — measured directly, throwing all four
on together produced the worst average rank of every combination tested (11.5th of 16
possible combos), and one of these four (`multiscale_bin`) actively hurts both mean AUC and
cross-dataset consistency every time it's included.** Use this as your default starting point
instead:

- **`ratio_poly`: apply by default.** Best average rank across all 16 datasets (3.6/16),
  outright best on 7 of 16. Cheap, capped to a handful of columns, never made things worse
  by more than a rounding error.
- **`group_flag`: apply by default, but it's usually a no-op.** None of the 16 practice
  datasets had the ≥3 binary-column family it requires, so it fell back to doing nothing
  every time it was tested — harmless to always check for, not something to rely on.
- **`digit_frac`: not a default.** Called "core" in the original recipe research, but
  measured directly it underperforms both `ratio_poly` and doing nothing at all (mean rank
  8.1/16, worse mean AUC than baseline). Only add it if your own EDA gives you a specific,
  concrete reason for *this* dataset (e.g. you can see suspiciously round numbers or a
  digit-position pattern in the actual values) — and verify with your own CV that it actually
  helps before trusting it, rather than assuming the original research generalizes here.
- **`multiscale_bin`: not a default, and the one to be most skeptical of.** Consistently the
  worst or near-worst performer alone (mean rank 9.8/16) and drags down every combination it's
  part of, while also *increasing* cross-dataset variance — the opposite of what this project
  actually rewards (generalizing across the family, not maximizing one dataset). Only add it
  with a concrete, EDA-verified reason, and treat a CV improvement from it with real
  suspicion — check it against the public score once submitted (see the selection-rule
  guidance in `system.md`) before trusting it further.

**Baseline (`none`) is a genuinely strong option, not just a fallback** — it had the 2nd-best
mean rank (4.8/16) and won outright on 7 of 16 datasets. Don't assume more feature engineering
is automatically better; the honest empirical answer here is "start minimal, add `ratio_poly`
by default, add anything further only with real EDA justification and your own verification."

All four take `(train_df, test_df, num_cols, ...)` and mutate both dataframes in place, adding
new columns to each. Adapt column names, thresholds, and which numeric columns to target based
on your own EDA — these are starting points, not fixed parameters.

**Detect `num_cols`/categorical columns programmatically at runtime (e.g. `select_dtypes`),
never hardcode a column list from one EDA pass.** Schemas genuinely differ across datasets in
this family — some datasets have zero categorical columns entirely (confirmed directly: one
local practice dataset has 21 numeric feature columns and 0 categorical). Verified directly:
running these exact four functions plus the leak-safe encoding pattern against that
zero-categorical dataset works with no errors and no special-casing, purely because column
lists were built from `select_dtypes` rather than assumed — an empty categorical list just
means the category-handling loops are no-ops, and `add_group_flag_aggregation` already
returns `[]` on its own when no binary-column family exists. Write your own code the same way
and it degrades gracefully for free.

## 1. Ratio & polynomial interaction features

Targets the numeric columns most correlated with the target; adds squares and pairwise
ratios/interactions among them. Capped to a handful of columns/pairs deliberately — this is
about the *top* signal-bearing numerics, not an exhaustive sweep over every pair.

```python
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
```

## 2. Decimal-digit / fractional-residue features

Treated as core, not a thin-evidence bet — this pattern independently recurred across
multiple unrelated solutions on this dataset family, likely a structural artifact of the
synthetic data generator. Apply to every numeric column, not just top-correlated ones.

```python
def add_digit_fraction_features(train_df, test_df, num_cols):
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
```

## 3. Binary-flag group aggregation

**Schema-conditional, not universal** — this is the one technique that should be skipped
when it structurally doesn't apply: if fewer than 3 numeric columns are binary `{0,1}`-valued,
there's no "family of flags" to aggregate. Check first; don't force it.

```python
def add_group_flag_aggregation(train_df, test_df, num_cols):
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
```

## 4. Multi-scale binning

Bins high-cardinality numeric columns four different ways (quantile, fixed-width,
round-then-divide, truncation). The bin labels become new categorical-like columns — feed
them through the leak-safe encoding pattern (`leak_safe_encoding.md`) like any other
categorical column, don't leave them as raw strings. Only apply to columns with real
cardinality (the `high_card_threshold` gate below) — binning a already-low-cardinality column
just duplicates it.

```python
def add_multiscale_bins(train_df, test_df, num_cols, high_card_threshold=20, n_bins=10):
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
```

## Beyond these four

These are a proven starting point, not the ceiling. If EDA reveals something these don't
capture (e.g. a suspicious multimodal distribution suggesting cluster structure, a pair of
columns whose difference looks meaningful, a datetime-like encoding hidden in a numeric
column), add it — the point of this skill is giving you room to do that, not limiting you to
exactly these four.
