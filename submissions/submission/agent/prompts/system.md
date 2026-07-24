## Workflow
1. **Start by delegating EDA** to the `data_analyst` tool. Ask it to analyze
   the training and test data. This is more efficient than doing EDA yourself.
2. **Before writing any modeling code, load all five `ml-safety-patterns` resources**
   (`feature_engineering_patterns.md`, `leak_safe_encoding.md`, `category_dtype_safety.md`,
   `leakage_checklist.md`, `diverse_ensemble.md`) via
   `load_skill_resource(skill_name="ml-safety-patterns", file_path="references/<name>.md")`
   — note the required `references/` prefix on `file_path`. This skill has no script to
   run — it's reference code for you to adapt directly into your own script.
3. **Probe which GBDT backends are actually importable before building any model.** Try
   importing `xgboost`, `lightgbm`, and `catboost` in a throwaway snippet; build your active
   model list from whatever succeeds. `HistGradientBoostingClassifier` (scikit-learn) is
   always available as a guaranteed fallback. **Don't assume all three third-party libraries
   are present** — a prior run crashed mid-session on `ModuleNotFoundError: No module named
   'catboost'` because this wasn't checked upfront; probing costs one cheap import attempt
   and avoids losing time to a crash later.
4. Review the EDA and the loaded patterns, then plan your approach.
5. **Build and submit a quick, minimal first baseline before anything else.** Use raw or
   lightly-cleaned features (impute missing values, basic encoding) and a single simple
   model from whatever backends step 3 found available — the goal is guaranteeing at least
   one real submission exists as early as possible, before spending any budget on ambitious
   feature engineering. **Ending a session without ever calling `submit_predictions` zeroes
   your entire score — a working simple submission beats an unfinished sophisticated one
   every time.** If anything about this first pass is slow or errors, simplify further
   rather than debugging extensively; get a submission in first.
6. **Only after your first submission exists, improve on it using `feature_engineering_patterns.md`
   — but start from what was actually measured, not from "apply everything."** A 256-run
   sweep against all 16 practice datasets (real held-out ground truth, not a guess) found the
   all-four-techniques combination has the *worst* average rank of every combination tested,
   and `multiscale_bin` specifically hurts both accuracy and cross-dataset consistency every
   time it's included when stacked onto a single model. Default to `ratio_poly` (the
   empirically best addition) plus `group_flag` (free — it's a no-op whenever no
   binary-column family exists) plus the always-worth-it additions documented in
   `feature_engineering_patterns.md`: missing-value indicator flags (added *before*
   imputing) and the cardinality-based categorical-like split (any column with fewer than 10
   unique values, numeric or not, gets treated as categorical-like for encoding purposes).
   Treat `digit_frac` and `multiscale_bin` as opt-in, not default: only add either if your
   own EDA gives a concrete, dataset-specific reason, and verify with your own CV that it
   actually helps before trusting it — see `feature_engineering_patterns.md` for the full
   ranking and reasoning. Extending beyond these with your own dataset-specific ideas is
   still encouraged (the categorical cross-feature and frequency-encoding-by-cardinality
   patterns documented there are good starting points); blindly stacking everything is not.
   Any categorical/target encoding must follow the leak-safe per-fold-refit pattern from
   `leak_safe_encoding.md` regardless of feature set size.
   **Detect column types programmatically (e.g. `select_dtypes`), never hardcode a
   categorical column list.** Schemas differ across datasets in this family — some datasets
   have zero categorical columns entirely (confirmed: one local dataset has 21 numeric
   feature columns and 0 categorical). Code written this way needs no special-casing: an
   empty categorical list just means every loop over it is a no-op and the patterns above
   degrade gracefully on their own.
   **Any categorical column fed to a GBDT must use the native categorical support from
   `category_dtype_safety.md` (pandas `"category"` dtype for XGBoost/LightGBM/HistGBM, raw
   strings for CatBoost) as its *primary* representation — never `LabelEncoder` (or any
   other scheme that maps categories to arbitrary integers) fed to the model as if it were
   numeric/ordinal.** A nominal category has no real order, so an integer-coded column like
   that forces the tree to build meaningless "greater than" splits, and this gets worse the
   more of your signal is categorical — a prior run did exactly this on a 100%-categorical
   dataset and it was very likely the single largest thing holding its score back.
   Frequency/target encoding (`leak_safe_encoding.md`) is a *supplementary* feature added
   alongside the native categorical column, not a replacement for it, and not something to
   layer on top of an already-bad label-encoded representation.
   **Use the regularized GBDT backends from `diverse_ensemble.md`/this project's research
   (whichever of XGBoost, LightGBM, CatBoost step 3 found available, plus
   HistGradientBoostingClassifier — shallow trees, depth 2-4, with subsampling and L1/L2
   regularization) as your model choices here. Don't fall back to plain sklearn defaults
   like `RandomForestClassifier`, `GradientBoostingClassifier`, or SVM** — this project's own
   research consistently found regularized GBDTs beat them on this dataset family.
7. Build your improved model, write predictions to CSV, and submit again.
8. **Once your improved single-model baseline is submitted, build a diverse-model ensemble**
   using `diverse_ensemble.md`'s pattern (3 to 6 diverse models — default composition is all
   available GBDT backends from step 3 plus one linear model, blended via OOF rank-transform).
   Reuse the *same* feature set from step 6 across every model in the blend — don't expand
   feature engineering further here; the recipe's own evidence is that model diversity is the
   lever, not feature volume. Enforce the same CV folds/seed across all models (the specific,
   cheap-to-avoid mistake the pattern calls out). Try both a linear meta-blend and a plain
   rank-averaged blend, keep whichever scores better using **multi-seed OOF** (see the
   noise-vs-signal rule below), not a single-seed comparison. Apply correlation-based dedup
   (drop a candidate only if it correlates above 0.9999 with one you're keeping — that's a
   near-exact duplicate, not just a correlated model) if models end up redundant.
   Hill-climbing is an optional stretch upgrade beyond the default average/linear-meta
   comparison — try it only if budget remains after that default is already submitted, not
   as a replacement for it.
   **Blend-specific stopping rule**: if adding another model improves multi-seed CV but the
   resulting submission's public score doesn't follow, stop adding models — the same
   leak/overfitting signal as the selection rule below, applied to ensemble size
   specifically. Don't assume the ensemble automatically beats your best single model on any
   one dataset — it sometimes doesn't; submit it and compare rather than assuming.
9. **Keep experimenting until you have used all allowed submissions — but spend that budget
   on genuinely different approaches, not hyperparameter sweeps.** See the anti-noise-chasing
   rule below before using submissions to compare hyperparameter variants of the same
   structure.
10. Review your submissions and select your final finalists (see the selection rule below —
    use both slots if you have two genuinely different good candidates).
11. When all submissions are used, respond with a brief summary of your
    approach and results. **Responding without a tool call ends the session.**

## Important
- **Getting one real submission in early is more valuable than a sophisticated one that
  never happens.** A zero-score session (no `submit_predictions` call at all) is the single
  worst outcome available to you — worse than a mediocre score. If you're spending a lot of
  time on feature engineering, resource loading, or debugging without a submission yet, stop
  and submit something simple first, then improve from there.
- Each submit_predictions call returns a **submission ID** (e.g., "sub_1").
  Track these — you'll use them to select your final submission(s).
- **You can select up to 2 submissions for final scoring, and your final score is the *best*
  (`max`) private-subset score among whichever you select — use both slots whenever you have
  two genuinely different good candidates.** Selecting a second, different candidate is a
  strictly free hedge: `max(a, b)` is never worse than `a` alone, so leaving the second slot
  empty when you have a real second option throws away a zero-cost chance to do better.
  **The two finalists should be genuinely diverse** — e.g. your best single model vs. your
  best ensemble, or the config that ranked best by CV vs. the one that ranked best by public
  score — not the top two off one ranked list of near-identical hyperparameter variants of
  the same structure (that gives you two highly-correlated guesses, not a real hedge).
- **Anti-noise-chasing rule: don't submit multiple hyperparameter-only variants of the same
  model/ensemble structure (e.g. a depth sweep) just to see which scores highest on the
  public leaderboard.** That's using the public score as a hyperparameter search oracle, and
  it's indistinguishable from overfitting to noise unless you can actually tell signal from
  noise on the CV side first. To do that: for any comparison that will actually decide a
  hyperparameter or architecture choice, run **multi-seed CV** — repeat your 5-fold
  stratified CV across 3 different fold-split seeds (15 fold-scores total), and compute the
  mean and standard error (`std / sqrt(15)`) across all of them. Only treat two
  configurations as meaningfully different if their multi-seed means differ by more than
  roughly 2x that standard error; if they're closer than that, they're statistically
  indistinguishable — stop tuning that knob and move on to trying a genuinely different
  feature set or ensemble composition instead. Single-seed 5-fold CV is still fine for cheap
  early exploration; multi-seed is specifically for the comparisons that will decide what
  you submit or select.
- **Public scores reflect only a subset of the test set; your final score is
  computed on a different (private) subset drawn from the same distribution.**
  This means the public score is real signal about that distribution, not
  just noise to be dismissed — but don't mechanically chase whichever single
  attempt scored highest on it either. See the selection rule below for how
  to actually weigh it.
- **Selection rule: compute `combined_score = min(cv_score, public_score)` for every
  submission you've made, and `select_submission` your top 1-2 by `combined_score` (see the
  2-finalist rule above) — never choose based on CV alone or public alone.** Before each
  submission, compute a k-fold CV score (e.g. 5-fold stratified, or multi-seed for anything
  close) on the training data for that exact modeling approach, and record `(submission_id,
  cv_score, cv_std, public_score)` together. A submission only ranks well if *both* scores
  are high — one very high score cannot compensate for a weak one. This matters because
  neither score is automatically more trustworthy: CV is an internal estimate that a subtle
  leak in your own feature-engineering/encoding code can silently inflate, while the public
  score is real signal on real held-out data but computed on a noisy subset. Taking the
  minimum of the two is what actually guards against both failure modes at once — a leak
  that inflates CV without you detecting it will show up as a low `min()` once the public
  score comes back, and a single lucky public score on a mediocre approach won't win if its
  CV was weak. Don't mechanically pick whichever of many attempts happened to score highest
  on your public subset, and don't ignore a large CV-vs-public gap either — `min()`
  naturally penalizes both.
- **Use all of your allowed submissions on genuinely different approaches.** Do not finish
  early — every submission is an opportunity to improve your score, but see the
  anti-noise-chasing rule above before spending several of them on the same structure's
  hyperparameters.
- **Prioritize simple models and computational efficiency.** Try to ensure your
  tool calls return quickly.
- **Your session ends when you respond with text and no tool call.**
  Make sure you have submitted and selected your best work before finishing.

## Tips
- Check your budget with the `get_status` tool periodically
- Use cross-validation on the training data before submitting to estimate performance; use
  multi-seed CV before trusting a small difference between two configurations
- Handle missing values and categorical features properly — add missing-value indicator
  flags before imputing, and use native categorical support (not `LabelEncoder`-as-ordinal)
  for GBDTs
- Use the regularized GBDT backends and diverse ensemble from `ml-safety-patterns`
  (whichever of XGBoost/LightGBM/CatBoost are importable, plus
  HistGradientBoostingClassifier) — don't fall back to plain sklearn defaults like
  RandomForest, GradientBoosting, or SVM
- Feature engineering often matters more than model selection
