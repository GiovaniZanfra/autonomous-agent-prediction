## Workflow (Recipe 1: Lean GBDT Baseline)
1. **Delegate EDA** to the `data_analyst` tool first — it's more efficient than doing
   it yourself, and the `lean-gbdt-baseline` skill's script also prints its own compact
   EDA summary (target balance, categorical cardinality, missing pattern, train/test
   duplicate-row check) as a fast cross-check.
2. **Run the `lean-gbdt-baseline` skill's `train_baseline.py`** via `run_skill_script`
   with default arguments first — this alone runs the full recipe (feature engineering
   + CV-safe encoding + one GBDT) and prints `VALIDATION_SCORE` / `VALIDATION_STD`. Submit
   its output CSV. This is your fast first real public-score baseline — always do this
   before anything fancier.
   **Always pass absolute `/work/...` paths for `--train`/`--test`/`--sample-submission`/
   `--output`** — `run_skill_script` runs the script in an isolated temporary directory
   that does not contain your problem's data files, and any relative `--output` path
   will vanish before you can submit it. See the skill's own `SKILL.md` for why.
3. **Time-permitting, iterate cheaply using the script's own flags rather than writing
   new code**:
   - `--features none|all|ratio_poly,digit_frac,...`: try `none` vs `all` and compare
     `VALIDATION_SCORE` — on one local dataset the plain-columns version scored *higher*
     than the fully-engineered version, so do not assume more features helps. Add feature
     groups incrementally and check each one's own effect, per the recipe's own advice,
     rather than trusting `all` blindly.
   - `--second-model lightgbm` (or another available library): a 50/50 blend with the
     primary GBDT is the cheapest next step once the single-model baseline is submitted.
   - `--model xgboost|lightgbm|catboost|hist_gbm`: try an alternative single backend if
     time remains.
4. **Do not hand-roll feature engineering or target encoding from scratch** unless the
   skill script's approach is clearly insufficient for what EDA revealed — the script
   already implements the recipe's leakage-safe pattern (frequency/target encoding refit
   per CV fold, never on the full train set). Reimplementing it ad hoc risks
   reintroducing the leakage the recipe explicitly warns about.
5. **Keep experimenting until you have used all allowed submissions.** Each submission
   is a chance to try a different flag combination or backend.
6. Review your submissions and select the best for final scoring.
7. When all submissions are used, respond with a brief summary of your approach and
   results. **Responding without a tool call ends the session.**

## Important
- Each submit_predictions call returns a **submission ID** (e.g., "sub_1"). Track these
  — you'll use them to select your final submission(s).
- You can select a limited number of submissions for final scoring. The best test-set
  score among your selections becomes your final score.
- **Public scores reflect only a subset of the test set.** Your final score is computed
  on a different (private) subset. Prefer models that generalize well — avoid
  overfitting to public leaderboard scores.
- **Selection rule: choose your final submission(s) by internal cross-validation score,
  not by the public score shown after `submit_predictions`.** The skill script already
  prints `VALIDATION_SCORE` (OOF ROC-AUC) and `VALIDATION_STD` (fold-to-fold variance)
  for exactly this purpose — record `(submission_id, VALIDATION_SCORE, VALIDATION_STD,
  public_score)` for every attempt. When it's time to call `select_submission`, rank by
  `VALIDATION_SCORE` (prefer lower `VALIDATION_STD` to break ties — it's a direct signal
  of how consistent the approach is across folds, which is what generalizing across the
  dataset family actually requires), and only use the public score as a sanity check
  that nothing is badly broken.
- **Use all of your allowed submissions.** Do not finish early — every submission is an
  opportunity to improve your score.
- **Prioritize simple models and computational efficiency.** The skill script typically
  runs in single-digit seconds on the local datasets; if a run takes far longer, that's
  a sign something (dataset size, `--n-folds`) needs adjusting, not a reason to wait
  indefinitely.
- **Your session ends when you respond with text and no tool call.** Make sure you have
  submitted and selected your best work before finishing.

## Tips
- Check your budget with the `get_status` tool periodically.
- `load_skill_resource(skill_name="lean-gbdt-baseline", resource_name="recipe_notes.md")`
  has the full recipe writeup, including techniques the script does *not* implement
  (Benford's-law digit features, TF-IDF over character n-grams, pseudo-target encoding
  against an auxiliary column) — worth reading if the baseline underperforms and you
  have budget left to hand-roll something extra.
