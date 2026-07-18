## Workflow
1. **Start by delegating EDA** to the `data_analyst` tool. Ask it to analyze
   the training and test data. This is more efficient than doing EDA yourself.
2. Review the analysis and plan your modeling approach.
3. Build several baseline models, write predictions to CSV, and submit for
   scoring.
4. Run the `lean-gbdt-baseline` skill's `train_baseline.py` via `run_skill_script` (see its
   `SKILL.md` — always pass absolute `/work/...` paths for every file argument, since
   `run_skill_script` runs the script in an isolated temporary directory that does not
   contain your problem's data files, and a relative path will cause "not found" errors on
   input or a vanished output CSV). It implements a stronger, leak-free recipe than
   hand-rolled code (per-fold-refit frequency/target encoding, capped feature engineering,
   GBDT backend auto-select) — use it to improve on your first hand-rolled baseline.
5. **You must try at least 3 distinctly different configurations before selecting your
   final submission(s) — this is required, not optional:**
   a. Default arguments.
   b. `--features none` vs `--features all` (or a specific feature subset) — compare
      `VALIDATION_SCORE`; more engineered features is not guaranteed to score higher.
   c. An alternate `--model` backend, or `--second-model <backend>` to blend two backends
      within one run.
   Record `(config, VALIDATION_SCORE, VALIDATION_STD, public_score)` for every run you try.
6. **Ensemble your two best distinct configurations.** If two tried configs have
   `VALIDATION_SCORE`s within roughly one `VALIDATION_STD` of each other, blend their
   submission CSVs: read both, merge on the id column, average the target column, write a
   new CSV, and submit it as an additional candidate. Averaging comparable-quality,
   differently-configured models reduces variance — this is directly aligned with the
   actual goal (generalizing across the dataset family, not maximizing one score).
7. **Keep experimenting until you have used all allowed submissions.**
   Each submission is a chance to try a different approach.
8. Review your submissions and select the best for final scoring.
9. When all submissions are used, respond with a brief summary of your
   approach and results. **Responding without a tool call ends the session.**

## Important
- Each submit_predictions call returns a **submission ID** (e.g., "sub_1").
  Track these — you'll use them to select your final submission(s).
- You can select a limited number of submissions for final scoring. The best
  test-set score among your selections becomes your final score.
- **Public scores reflect only a subset of the test set.** Your final score
  is computed on a different (private) subset. Prefer models that generalize
  well — avoid overfitting to public leaderboard scores.
- **Selection rule: choose your final submission(s) by internal cross-validation
  score, not by the public score shown after `submit_predictions`.** Before each
  submission, compute a k-fold CV score (e.g. 5-fold stratified) on the training
  data for that exact modeling approach, and record `(submission_id, cv_score,
  cv_std, public_score)` together. When it's time to call `select_submission`,
  rank by `cv_score` (and prefer lower `cv_std` to break ties), and only use the
  public score as a sanity check that nothing is badly broken. The public score
  is a noisy subset and chasing it directly is how you overfit to it.
  **If a submission came from the `lean-gbdt-baseline` skill, use its printed
  `VALIDATION_SCORE`/`VALIDATION_STD` directly instead of hand-computing your own CV for
  that attempt** — it's already leak-free and fold-averaged. When selecting, weigh the
  ensemble candidate (step 6) favorably if the `VALIDATION_SCORE`s of the configs it blends
  were close — blending comparable models is itself a variance-reduction bet.
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
