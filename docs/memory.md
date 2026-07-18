# Project memory / work log

Running log of setup and decisions made while working on the Autonomous Agent
Prediction submission. Newest entries at the top.

---

## 2026-07-17 — Third submission: `submissions/03_lean_gbdt_hybrid`

**What it is:** a fork of `01_fe_categorical_encoding`'s workflow (hand-rolled baseline
submitted first — a safety net that doesn't depend on any skill call succeeding) with its
weak `feature-engineer` skill swapped for the fixed, far more capable `lean-gbdt-baseline`
skill from `02` (CV-safe target encoding, engineered features, multi-backend GBDT). Built
by copying files directly rather than via a notebook, given today's time constraint.

**Two prompt improvements made here that should have been in `02` from the start** (user
caught this — see [[feedback_agent_prompt_strictness]] in cross-session memory): the old
"time-permitting, iterate cheaply" language is now a **required** minimum of 3 distinct
configurations (default, `--features none/all`, alternate `--model`/`--second-model`), each
logged as `(config, VALIDATION_SCORE, VALIDATION_STD, public_score)`. Also added a required
ensembling step: if two tried configs have close `VALIDATION_SCORE`s, blend their submission
CSVs (simple average of the target column) and submit that as an extra candidate.

**Honest framing:** the underlying model/recipe is identical to `02` — same skill, same
script. The real differences are (a) hand-rolled-baseline-first ordering as a hedge against
today's path-handling bugfix not being re-verified through a live LLM run (Gemini free-tier
quota was exhausted), and (b) the two required-not-soft prompt changes above.

**Verified before packaging:** `validate_submission.py` passed structurally; the copied
`train_baseline.py` was re-run directly (no LLM) against `train_01` from a fake unrelated
cwd with absolute paths, confirming the copy works and output lands correctly. Not yet
verified through an actual agent loop (blocked by the same quota issue as the rest of
today's work) — this is a real gap, not a formality skipped for no reason.

**Submission mechanics:** no Kaggle API credentials are configured in this dev environment
(no `~/.kaggle/kaggle.json`, no env vars) — the first two submissions were uploaded outside
this environment too. `submission.zip` was packaged here; actual `kaggle competitions submit`
is done by the user directly.

---

## 2026-07-17 — Container runtime set up; first real local agent run finds a genuine bug

**Environment finished:** installed podman, enabled `podman.socket`, pulled the official
sandbox image (`gcr.io/kaggle-images/python`, 25.6GB), added `GEMINI_API_KEY` and
`DOCKER_HOST=unix:///run/user/1000/podman/podman.sock` to `.env`. This unblocks
`run_local_eval.py` for the first time — previously only static validation and directly
invoking modeling scripts (bypassing the LLM) were possible.

**Two infra snags on the way, both fixed:**
- A bare `uv run python run_local_eval.py` auto-syncs the venv to the root
  `pyproject.toml`'s `requires-python = "~=3.10.0"` pin (meant for the unrelated
  cookiecutter package) and silently **deleted** the working 3.12 venv, wiping
  `adk-submission`/`kaggle-kaggle`/pandas etc. Fix: rebuild `.venv --python 3.12` via
  `uv venv`/`uv pip install` directly, then always invoke `.venv/bin/python` rather than
  `uv run` for these two scripts.
- `litellm`'s MCP codepath imports `orjson`, which isn't in `requirements.txt`. Fixed by
  installing it directly into `.venv`.

**First real agent-loop run (train_01, `submissions/02_lean_gbdt_baseline`) surfaced a
genuine, previously-unknown bug — not a fluke, a structural one:** `run_skill_script`
(from `google.adk.tools.skill_toolset`) materializes a skill's own `scripts/`/`references/`
/`assets/` files into a **fresh `tempfile.TemporaryDirectory()`**, `chdir`s into it, runs
the script, then deletes that directory the moment the tool call returns. That directory
never contains the problem's `/work/train.csv` etc. — only the skill's bundled files.
Confirmed directly by reading the installed package source, not inferred.

The agent hit exactly this: called `run_skill_script` with the `SKILL.md`-documented
relative path (`--train train.csv`), got back `Error: 'train.csv' not found.`, then ran
`pwd && ls -la` via `run_command` (different tool, different cwd — `/work`, where the file
plainly exists) to debug — and was mid-investigation when Gemini's free-tier quota
(20 requests/day for `gemini-3.5-flash`, a hard daily cap, not a transient 429) cut the
run off before it could retry with a fix.

**Both submissions had this bug baked into their own `SKILL.md` examples** (inherited
from the shared template pattern, not something introduced by hand in just one): relative
`--train`/`--test`/`--output` paths that were always going to fail under
`run_skill_script`, plus a `skill_name` typo in both (`feature_engineer` /
`lean_gbdt_baseline` with underscores, vs. the real hyphenated names in each `SKILL.md`
frontmatter).

**Fixed properly, not just documented:**
- `submissions/01_fe_categorical_encoding/agent/skills/feature-engineer/scripts/generate_features.py`
  had **hardcoded** output filenames (`train_engineered.csv`/`test_engineered.csv` in
  cwd) with no CLI flag to redirect them — a real code bug, since no prompt wording can
  fix a hardcoded path. Added a proper `--output-dir` argument.
- `submissions/02_lean_gbdt_baseline`'s `train_baseline.py` was already parameterized
  (`args.output`) — that one only needed the `SKILL.md`/`system.md` docs fixed to *use*
  absolute paths.
- Both `SKILL.md` files now show absolute `/work/...` paths in their usage examples and
  explicitly explain *why* (the temp-dir isolation), so the agent doesn't have to
  rediscover this by trial and error every session. Both `system.md` files got a
  one-line reminder in the relevant workflow step. Both `skill_name` typos fixed.
- Verified manually (not via the LLM, since the Gemini free-tier quota was already
  exhausted for the day): ran each script from a fake unrelated cwd with absolute I/O
  paths, confirmed outputs land exactly where directed regardless of the script's actual
  working directory.

**Retroactive implication for the 2026-07-16 entry below:** this is a plausible (not
confirmed — Kaggle still exposes no trace for graded runs) explanation for why the real
0.818/0.808 scores looked reasonable despite this bug likely being present in the graded
runs too — the agent may have spent real effort recovering from/working around exactly
this path failure (e.g. falling back to hand-rolled code using `run_command`'s working
`/work` cwd) rather than actually exploring the `--features`/`--model` toggles as
intended. Still can't know for certain without a trace, but it's a more concrete
candidate explanation than "noise" alone.

**Not yet done:** none of today's fixes are committed yet. Also still need to re-run the
local eval end-to-end (blocked until the Gemini free-tier daily quota resets) to confirm
the fix actually lets the agent complete a full submit → select cycle.

---

## 2026-07-17 — First real Kaggle scores, and why they're inconclusive

**What happened:** submitted both `submission.zip`s to the actual competition (not local
eval). Real public leaderboard scores came back: **submission 01 (`feature-engineer`,
freeform LLM-authored modeling) scored 0.818**; **submission 02
(`lean-gbdt-baseline`, packaged recipe script) scored 0.808**. This is the first time
the *full* agentic loop — real LLM, real Kaggle model proxy, real sandbox — has actually
run for either submission; everything before this was either static validation or
directly invoking the modeling scripts myself, bypassing the LLM entirely.

**Kaggle exposes no execution trace/log for a successful graded submission** — confirmed
directly, not assumed. There is no way to inspect after the fact what the agent actually
did (which tools it called, whether it explored flags, how many submissions it used)
for a real graded run. `parse_eval_trace.py` / `trace_*.json` only exist for **local**
`run_local_eval.py` runs — they say nothing about what happened on Kaggle's
infrastructure.

**Why the 0.818 vs 0.808 gap is likely noise, not evidence "01 beat 02":** a ~0.01 AUC
gap is well inside the fold-to-fold variance we measured locally (`VALIDATION_STD`
ranged ~0.003–0.05 across the 16 practice datasets in `submissions/02_lean_gbdt_baseline`'s
own benchmark). A single public-subset draw producing this gap doesn't establish which
recipe generalizes better — this is exactly the "don't overfit to the public score" trap
CLAUDE.md warns about, and it cuts both ways: it's equally weak evidence that 02
under-explored *and* that 01 is the better design.

**Real, unresolved concern surfaced by this**: `submissions/02_lean_gbdt_baseline/agent/prompts/system.md`
only asks the agent to explore `--features`/`--model` toggles under soft
"time-permitting, iterate cheaply" language, with **no enforcement** that the tried
configurations actually differ (resubmitting near-identical configs, or only tweaking
`--n-folds`/`--depth`, would technically satisfy "keep experimenting until submissions
run out"). Best guess, not confirmed: more likely than not the agent ran the default
once or twice and moved on, since nothing in the prompt penalizes that. Two ways to
actually resolve this rather than keep guessing:
1. Tighten `system.md` to *require* a minimum number of distinctly different
   configurations tried before `select_submission` is allowed (a concrete follow-up, not
   yet done).
2. Once `data/.env`/API key + container runtime exist (still the blocker — see the
   2026-07-15 entry), run `run_local_eval.py` against this same agent and inspect the
   real tool-call trace via `parse_eval_trace.py` to see empirically whether the current
   prompt induces exploration at all.

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
