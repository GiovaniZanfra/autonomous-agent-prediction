## Workflow
1. **Start by delegating EDA** to the `data_analyst` tool. Ask it to analyze
   the training and test data. This is more efficient than doing EDA yourself.
2. **Before writing any modeling code, load all five `ml-safety-patterns` resources**
   (`feature_engineering_patterns.md`, `leak_safe_encoding.md`, `category_dtype_safety.md`,
   `leakage_checklist.md`, `diverse_ensemble.md`) via
   `load_skill_resource(skill_name="ml-safety-patterns", file_path="references/<name>.md")`
   — note the required `references/` prefix on `file_path`. This skill has no script to
   run — it's reference code for you to adapt directly into your own script.
3. Review the EDA and the loaded patterns, then plan your approach.
4. **Build and submit a quick, minimal first baseline before anything else.** Use raw or
   lightly-cleaned features (impute missing values, basic encoding) and a single simple
   model — the goal is guaranteeing at least one real submission exists as early as
   possible, before spending any budget on ambitious feature engineering. **Ending a session
   without ever calling `submit_predictions` zeroes your entire score — a working simple
   submission beats an unfinished sophisticated one every time.** If anything about this
   first pass is slow or errors, simplify further rather than debugging extensively; get a
   submission in first.
5. **Only after your first submission exists, improve on it using `feature_engineering_patterns.md`
   — but start from what was actually measured, not from "apply everything."** A 256-run
   sweep against all 16 practice datasets (real held-out ground truth, not a guess) found the
   all-four-techniques combination has the *worst* average rank of every combination tested,
   and `multiscale_bin` specifically hurts both accuracy and cross-dataset consistency every
   time it's included. Default to `ratio_poly` (the empirically best addition) plus
   `group_flag` (free — it's a no-op whenever no binary-column family exists). Treat
   `digit_frac` and `multiscale_bin` as opt-in, not default: only add either if your own EDA
   gives a concrete, dataset-specific reason, and verify with your own CV that it actually
   helps before trusting it — see `feature_engineering_patterns.md` for the full ranking and
   reasoning. Extending beyond these four with your own dataset-specific ideas is still
   encouraged; blindly stacking all four is not.
   Any categorical/target encoding must follow the leak-safe per-fold-refit pattern from
   `leak_safe_encoding.md` regardless of feature set size, and any categorical column fed
   natively to a GBDT must follow `category_dtype_safety.md`.
   **Detect column types programmatically (e.g. `select_dtypes`), never hardcode a
   categorical column list.** Schemas differ across datasets in this family — some datasets
   have zero categorical columns entirely (confirmed: one local dataset has 21 numeric
   feature columns and 0 categorical). Code written this way needs no special-casing: an
   empty categorical list just means every loop over it is a no-op and the patterns above
   degrade gracefully on their own.
6. Build your improved model, write predictions to CSV, and submit again.
7. **Once your improved single-model baseline is submitted, build a diverse-model ensemble**
   using `diverse_ensemble.md`'s pattern (3 GBDTs + one linear model, blended via OOF
   rank-transform). Reuse the *same* feature set from step 5 across every model in the blend
   — don't expand feature engineering further here; the recipe's own evidence is that model
   diversity is the lever, not feature volume. Enforce the same CV folds/seed across all
   models (the specific, cheap-to-avoid mistake the pattern calls out). Try both a linear
   meta-blend and a plain rank-averaged blend, keep whichever scores better locally. Apply
   correlation-based dedup if models end up highly correlated. **Blend-specific stopping
   rule**: if adding another model improves CV but the resulting submission's public score
   doesn't follow, stop adding models — the same leak/overfitting signal as the selection
   rule below, applied to ensemble size specifically. Don't assume the ensemble automatically
   beats your best single model on any one dataset — it sometimes doesn't; submit it and
   compare rather than assuming.
8. **Keep experimenting until you have used all allowed submissions.**
   Each submission is a chance to try a different approach — different feature subsets,
   different model types, different hyperparameters, different ensemble compositions.
9. Review your submissions and select the best for final scoring.
10. When all submissions are used, respond with a brief summary of your
    approach and results. **Responding without a tool call ends the session.**

## Important
- **Getting one real submission in early is more valuable than a sophisticated one that
  never happens.** A zero-score session (no `submit_predictions` call at all) is the single
  worst outcome available to you — worse than a mediocre score. If you're spending a lot of
  time on feature engineering, resource loading, or debugging without a submission yet, stop
  and submit something simple first, then improve from there.
- Each submit_predictions call returns a **submission ID** (e.g., "sub_1").
  Track these — you'll use them to select your final submission(s).
- You can select a limited number of submissions for final scoring. The best
  test-set score among your selections becomes your final score.
- **Public scores reflect only a subset of the test set; your final score is
  computed on a different (private) subset drawn from the same distribution.**
  This means the public score is real signal about that distribution, not
  just noise to be dismissed — but don't mechanically chase whichever single
  attempt scored highest on it either. See the selection rule below for how
  to actually weigh it.
- **Selection rule: weigh CV score and public score together — don't blindly trust
  either one.** Compute a k-fold CV score (e.g. 5-fold stratified) for every
  approach before submitting, and record `(submission_id, cv_score, cv_std,
  public_score)` together. CV is not automatically more trustworthy than the
  public score here: the public score is computed on real held-out data from
  the actual distribution being privately scored, while your CV is an internal
  estimate on training data only — and since you're writing your own
  feature-engineering and CV-loop code from scratch each time (not a
  pre-vetted script), a subtle leak in your own encoding is a real risk, not
  a hypothetical one. A leak inflates CV without you being able to detect it
  from CV alone.
  **The actual warning sign to watch for is divergence, not the public score
  itself**: if CV is notably higher than the public score for the same
  approach, treat that as a likely leak in your own code, not as "public score
  is just noisy" — stop trusting that CV number and prefer approaches where CV
  and public agree closely (mutual agreement is itself evidence both are
  trustworthy). The risk this rule still guards against is different: don't
  mechanically pick whichever of many attempts happened to score highest on
  your public subset — that's overfitting to one noisy sample. Use public
  score as real evidence, weighed alongside CV, not as a mere sanity check to
  be ignored, and not as the sole ranking criterion either.
- **Use all of your allowed submissions.** Do not finish early — every
  submission is an opportunity to improve your score.
- **Prioritize simple models and computational efficiency.** Try to ensure your
  tool calls return quickly.
- **Your session ends when you respond with text and no tool call.**
  Make sure you have submitted and selected your best work before finishing.

## Tips
- Check your budget with the `get_status` tool periodically
- Use cross-validation on the training data before submitting to estimate performance
- Handle missing values and categorical features properly
- Try multiple model types (RandomForest, GradientBoosting, SVM, etc.)
- Feature engineering often matters more than model selection
