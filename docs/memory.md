# Project memory / work log

Running log of setup and decisions made while working on the Autonomous Agent
Prediction submission. Newest entries at the top.

---

## 2026-07-16 — Second submission: `submissions/02_lean_gbdt_baseline` (Recipe 1)

**What was done:** turned a pasted "Recipe 1: Lean GBDT Baseline" writeup (a
trimmed-down common core distilled from several heavier community solutions to similar
tabular competitions) into a real skill + agent, built from scratch rather than forked
from a demo. `notebooks/02-lean-gbdt-baseline-agent.ipynb` builds it.

- **`skills/lean-gbdt-baseline/scripts/train_baseline.py`** runs the whole recipe in one
  CV-safe script: EDA summary → capped feature engineering (ratio/interaction + square
  terms on the columns most correlated with target; decimal-digit/fractional-residue
  features on every numeric column, treated as core per the recipe's cross-competition
  evidence; multi-scale binning on high-cardinality numerics; binary-flag-family
  aggregation when detected) → frequency + smoothed target encoding **refit inside every
  CV fold** (never on the full train set — the recipe's own leakage warning, implemented
  literally rather than just noted) → one GBDT, auto-selecting xgboost → lightgbm →
  catboost → sklearn `HistGradientBoostingClassifier`, whichever import succeeds (offline
  sandbox, "whichever is pre-installed"). Shallow depth (default 4) with heavy
  regularization, per the recipe's own evidence that shallow trees generalize better on
  this data family than deeper ones.
- **Tested directly, not just eyeballed**, against `train_01`, `train_05`, `train_09`,
  and `train_16` (deliberately chosen: `train_16` has **zero** categorical columns —
  the harshest edge case for the schema-agnostic encoding path) — all ran cleanly in
  single-digit seconds. Also verified the `--second-model` 50/50 blend path and the
  sklearn `hist_gbm` fallback path explicitly (only xgboost/lightgbm are installed
  locally for dev-testing; catboost was left untested locally but follows the same
  pattern).
- **A real finding, not just a passing test**: on `train_05`, `--features none` scored
  *higher* (0.6749 OOF AUC) than `--features all` (0.6494) — concrete evidence for the
  recipe's own "add features incrementally, check each one's contribution" advice. This
  is why `--features` is a runtime flag the agent can toggle across submissions rather
  than a fixed always-on pipeline — confirmed to matter, not decorative.
- Added fold-wise `VALIDATION_STD` reporting (not just the pooled OOF
  `VALIDATION_SCORE`) so `system.md`'s CV-based selection rule (carried over from
  experiment 1) has an actual variance signal to break ties with, instead of asserting a
  metric the script didn't produce.
- Suppressed pandas' `PerformanceWarning` in the script — the column-by-column feature
  engineering fragments the dataframe and was printing enough warning spam to risk
  burying the `VALIDATION_SCORE` line past the agent's default 5000-char
  `max_stdout_chars` truncation. Found by actually reading the script's stdout during
  testing, not by inspection.
- Deliberately did **not** implement three of the recipe's own "time-permitting extras"
  (Benford's-law digit deviation, TF-IDF over character n-grams of stringified floats,
  offline pseudo-target encoding against an auxiliary column) — documented instead in
  `resources/recipe_notes.md` as follow-ups, since the last one specifically needs a
  per-dataset auxiliary-column candidate identified via EDA first, not something a
  generic script can pick blindly.
- Used the canonical `submissions/<name>/agent/` + per-experiment `submission.zip`
  layout from the start (see the layout-change entry below) — this one didn't need
  migrating.

**Still open:** same as always — no `.env` API key, no container runtime, so this is
validated structurally (`validate_submission.py`) and at the standalone-script level
(direct runs against 4+ local datasets), not yet via a full `run_local_eval.py` agent
loop.

---

## 2026-07-16 — Repo layout changed again: starter kit moved to repo root

**What changed:** between sessions, `kaggle-kaggle-skill/`, `wheels/`, `models.yaml`,
`sample_submission/`, `run_local_eval.py`, `validate_submission.py`, `requirements.txt`,
`.env.example` moved from `data/raw/` to the **repo root**. The 16 datasets stayed under
`data/raw/train_{01-16}/`, but lost the extra nested `data/raw/data/` level (one level
shallower than before). `CLAUDE.md` had to be re-fixed accordingly (see its "Local
development data" section for the current authoritative paths — don't trust older
descriptions in this log, including the 2026-07-15 entries below, for exact paths).

**Consequence discovered while fixing it:** `run_local_eval.py` hardcodes
`base_dir / "data" / <dataset>` (`base_dir` = the script's own directory, now the repo
root) — it has no flag to point elsewhere. Since the real data lives at
`data/raw/train_XX`, not `data/train_XX`, this would fail outright. Fixed by symlinking
`data/train_01` … `data/train_16` → `raw/train_01` … `raw/train_16` (cheap, no data
duplication; `/data/` is already fully gitignored so this is purely local convenience,
not something that needs to survive a fresh clone — redo it if `data/train_XX` is ever
missing).

**Also discovered**: `kaggle-kaggle-skill/SKILL.md`'s canonical layout is
`submissions/<name>/agent/{agent.yaml,prompts,tools,skills}` +
`submissions/<name>/output/` + `submissions/<name>/submission.zip` — **not** the flatter
`submissions/<name>/{agent.yaml,...}` + a single shared root `submission.zip` used for
the first experiment (`01_fe_categorical_encoding`). `run_local_eval.py` specifically
auto-detects the `.../submissions/<name>/agent` suffix to route trace output to
`submissions/<name>/output/`. Migrating experiment 1 to match this layout is tracked as
follow-up work; new experiments should use the canonical layout from the start.

**Lesson for next time:** the starter-kit's own file locations are not stable across
sessions on this machine (this is the second reorg) — re-verify actual paths with `ls`
before trusting any previously-recorded path, including in this file.

---

## 2026-07-15 — First real submission built: `submissions/01_fe_categorical_encoding/`

**What was done:**

- Forked `notebooks/demo_notebooks/autonomous-agent-prediction-beta-demo-agent.ipynb`
  into `notebooks/01-fe-categorical-encoding-agent.ipynb` (first cell notes the fork
  and the rationale) and used it as the baseline agent config.
- Found and fixed a real bug in the demo's `feature-engineer` skill: it imputes
  missing values in categorical/ordinal columns but never encodes them to numeric.
  Confirmed by inspecting `data/raw/data/train_01/train.csv` — columns like
  `feature_0` (`cat_1`, ...) and `feature_8` (`ord_3`, ...) are `object` dtype, so any
  sklearn estimator fit directly on `generate_features.py`'s output would crash.
- **Key discovery driving the fix's design**: the dataset family does **not** keep a
  fixed feature-to-type mapping. Diffing `DATA.md` across `train_01`/`train_05`/`train_09`
  shows `feature_1` is `ordinal` in one dataset and `count` in another, and the number
  of categorical columns varies (`train_16` has none at all). So the updated
  `generate_features.py` types each non-numeric column **by its values**
  (`ord_<int>` pattern → ordinal, encoded to the trailing integer to preserve order;
  anything else → nominal, label-encoded via a train-fitted mapping, unseen test
  categories → `-1`), never by column name or position. This is the only approach that
  generalizes to unseen datasets from the same family — the actual thing being scored.
  Also excluded `row_id` from imputation/encoding/`row_mean` (previously it would have
  been swept into the categorical columns as a useless high-cardinality feature).
- Strengthened `prompts/system.md`: added an explicit rule that `select_submission`
  choices must be ranked by internal CV score (record `(submission_id, cv_score,
  cv_std, public_score)` per attempt), using the public score only as a sanity check —
  directly implementing the competition's winner's-curse warning rather than leaving it
  implicit.
- Verified the fix for real, not just by inspection: the notebook runs
  `generate_features.py` directly against `data/raw/data/train_01`, asserts the output
  has zero non-numeric columns, and fits a `RandomForestClassifier` with 5-fold CV
  (`roc_auc = 0.6978 ± 0.0059`) — this would have raised on the un-forked version.
- Packaged `submissions/01_fe_categorical_encoding/` into `submission.zip` (repo root)
  using Python's stdlib `zipfile` — the `zip` CLI isn't installed on this machine, so
  shelling out to it fails silently different ways; stdlib avoids the dependency
  entirely and is what the notebook now uses.
- Ran `data/raw/validate_submission.py` against it: **passed** — YAML/include syntax
  valid, all requested models in `models.yaml`, dry-run ADK compilation succeeded with
  7 tools.
- Installed `nbconvert`, `nbclient`, `ipykernel` into `.venv` (dev-only tooling, not
  part of the submission) so notebooks can be executed headlessly via
  `jupyter nbconvert --to notebook --execute --inplace`.
- Added `/submission.zip`, `train_engineered.csv`, `test_engineered.csv` to
  `.gitignore` — build/scratch artifacts regenerated by the notebook, not source.

**Still open / not done yet:** same gaps as below — no `data/raw/.env` (LLM API key),
no container runtime. So this validates the submission **structurally** and validates
the **modeling logic** in isolation (outside the agent loop), but a full
`run_local_eval.py` pass (real agent, real LLM, inside the sandbox container) has
still never been run. That's the next real test once a key + runtime are in place.

---

## 2026-07-15 — Local dev environment set up (venv via `uv`)

**What was done:**

- Discovered that the repo layout doesn't match what `CLAUDE.md` originally described:
  this is a cookiecutter-data-science project, so the Kaggle-in-Kaggle starter kit
  (datasets, eval scripts, skill docs) landed one level down under `data/raw/` instead
  of at the repo root. Fixed `CLAUDE.md` to reflect the real paths
  (`data/raw/data/train_XX`, `data/raw/run_local_eval.py`,
  `data/raw/validate_submission.py`, `data/raw/kaggle-kaggle-skill/`,
  `data/raw/.env.example`, `data/raw/sample_submission/`).
- Installed `uv` (0.11.29) to `~/.local/bin` via the official install script.
- Created a **virtual environment at the repo root**: `.venv/` (already covered by
  `.gitignore`, not committed).
  - Built with **Python 3.12**, not 3.10. The root `pyproject.toml` pins
    `requires-python ~=3.10.0`, but that constraint belongs to the unrelated
    `autonomous_agent_prediction` cookiecutter package skeleton, not to the
    Kaggle-in-Kaggle tooling. The `adk-submission` wheel requires Python **>=3.11**,
    so a 3.10 venv fails to resolve. Used the system's available Python 3.12
    interpreter instead (no conflict with the cookiecutter package since it isn't
    installed into this venv).
  - Command used: `uv venv .venv --python 3.12`
- Installed dependencies from `data/raw/requirements.txt` into that venv:
  - `uv pip install --python .venv -r data/raw/requirements.txt` **fails** if run from
    the repo root — the requirements file references the local wheels with relative
    paths (`./wheels/adk_submission-...whl`, `./wheels/kaggle_kaggle-...whl`), and those
    resolve relative to the **current working directory**, not the requirements file's
    location.
  - Fix: `cd data/raw && uv pip install --python ../../.venv -r requirements.txt`.
  - Installs the two local wheels (`adk_submission`, `kaggle_kaggle`) plus
    `litellm`, `python-dotenv`, `pyyaml`, `pydantic`, `pandas`, `numpy`,
    `scikit-learn`, and their transitive deps.
- Verified the environment works:
  - `.venv/bin/python -c "import adk_submission; import kaggle_kaggle"` succeeds.
  - `.venv/bin/python data/raw/validate_submission.py --agent-dir sample_submission`
    succeeds end-to-end against the baseline template (compiles the ADK agent, all
    checks pass). Note: `--agent-dir` is resolved **relative to the script's own
    directory** (`data/raw/`), not the caller's cwd — pass `sample_submission`, not
    `data/raw/sample_submission`.

**Still open / not done yet:**

- No `data/raw/.env` created yet — no LLM API key configured, so
  `run_local_eval.py` (which actually drives the agent with a live LLM) can't run
  yet. `validate_submission.py` (static structure/spec checks) does **not** need a
  key and already works.
- No container runtime installed (`podman`/`docker` both absent). `run_local_eval.py`
  runs the agent inside the Kaggle sandbox container image, so this is required before
  a full local evaluation can execute, independent of the API key question.

**Key insight worth remembering:** `models.yaml` model paths are all
`openai/anthropic/...`, `openai/google/...`, `openai/xai/...`, etc. — meaning the
**real Kaggle submission runs through Kaggle's own model proxy** (`.env.example`
Option 2), not personal API keys. Personal keys (`GEMINI_API_KEY`, etc., Option 1 in
`.env.example`) are only needed for **local testing** on this machine, not for the
actual competition run. Recommended path for local testing without paying for
API access: a free Gemini API key from Google AI Studio (no credit card, generous
free tier), since it's in the same model family (`gemini-3.1-flash-lite`,
`gemini-3.5-flash`, etc.) listed in `models.yaml`.
