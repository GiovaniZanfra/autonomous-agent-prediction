# CLAUDE.md
 
Project context for Claude (Claude Code or similar) working in this repository.
 
## About the project
 
This repo is for the Kaggle competition **Autonomous Agent Prediction (Beta)** — a
meta-competition. We're not training a model and submitting predictions directly:
we're building an **Agent Config** (prompts, tools, skills, subagents) that, inside a
sandboxed Docker container, autonomously explores data, trains models, and submits
predictions on binary-classification datasets it has never seen before.
 
- Overview: https://www.kaggle.com/competitions/autonomous-agent-prediction-beta/overview
- Data: https://www.kaggle.com/competitions/autonomous-agent-prediction-beta/data
- Metric: AUC-ROC
- Start: July 6, 2026 · Final deadline: August 6, 2026 (11:59 PM UTC)
## How evaluation works
 
Each submission runs **two sessions** (independent mini-competitions, both on synthetic
datasets from the same family of generating processes):
 
1. One session populates the **Public Leaderboard**.
2. One session populates the **Private Leaderboard**.
In each session, the agent gets `train.csv`, `test.csv`, and `sample_submission.csv` in a
clean sandbox, explores the data, trains models, and calls `submit_predictions` as many
times as it wants (up to the limit). Each submission is scored on a public subset of the
test set — visible to the agent, which can use that to pick its best work. At the end, the
agent calls `select_submission` to choose a small number of final submissions; those are
scored on a **private** subset, which produces that session's final score.
 
The session ends when: the budget runs out, OR the agent replies with plain text and no
tool call. **Ending without having submitted/selected anything is costly** — be careful
about this when writing prompts.
 
## Budget per session (there are 2 sessions per submission)
 
| Resource | Limit |
|---|---|
| `max_time_minutes` | 60 min |
| `max_submissions` | 30 |
| `max_budget_usd` | $2.00 (LLM token spend) |
 
The agent's sandbox is **fully offline** — no internet access, no `pip install`. Only
packages already installed in the standard Kaggle environment are available.
 
## Required submission.zip structure
 
```
submission.zip
├── agent.yaml              # required, at the root
├── prompts/
│   ├── system.md
│   └── subagent_1.md       # optional
├── tools/
│   └── subagent_1.yaml     # optional
└── skills/
    └── my-skill/
        ├── SKILL.md        # YAML frontmatter with `name: <skill-name>`
        ├── scripts/        # run via run_skill_script, inside the sandbox
        └── resources/       # domain-knowledge markdown, via load_skill_resource
```
 
Config rules:
- Config language follows the **Google ADK Agent Config spec**, with extra restrictions to
  prevent code execution outside the sandbox.
- `!include some/path.md` resolves the path **relative to the directory of the file that
  contains the tag** (not relative to the repo root).
- **Sandboxing**: path traversal outside the submission root (`../`, symlinks) is not
  allowed — neither in `!include` nor inside skills.
### Tools allowed by the harness
 
The agent may only request these native tools, or custom subagents via `agent_tool`:
 
```
run_command, write_file, edit_file, submit_predictions, select_submission, get_status
```
 
Any other tool in `agent.yaml` will likely be rejected during validation.
 
## Models (models.yaml)
 
Example models available (price per 1M tokens: input / cached input / output):
 
| Model ID | Input | Cached Input | Output |
|---|---|---|---|
| `gemini-3.1-flash-lite` | $0.25 | $0.025 | $1.50 |
| `gemini-3.1-pro-preview` | $2.00 | $0.20 | $12.00 |
| `gemini-3.5-flash` | $1.50 | $0.15 | $9.00 |
 
The full, current list is in `models.yaml` (competition dataset). Given the $2.00/session
budget, it's worth using "lite" models for routine subagents (e.g. policy/tree manager)
and reserving a stronger model for where it actually matters (e.g. the coding operator).
 
## Local development data
 
The starter-kit files live at the **repo root** (as of 2026-07-16 — this was previously
nested under `data/raw/`, moved back out since):
 
- `wheels/` — `adk_submission` and `kaggle_kaggle` wheels needed by the two scripts below
  (install via `requirements.txt` at repo root, not on PyPI; requires Python >=3.11).
- `run_local_eval.py` — evaluates an agent locally against a single dataset. Resolves
  `--submission-dir` and `--dataset` **relative to its own directory** (the repo root),
  and always looks for datasets at `<repo_root>/data/<dataset>`.
- `validate_submission.py` — validates a submission directory/zip (structure, spec)
  without running a full evaluation. `--agent-dir` is also resolved relative to the
  script's own directory.
- `kaggle-kaggle-skill/` — skill for creating/submitting agents; `resources/` has full
  documentation on the evaluation system, agent config spec, and environment setup.
- `sample_submission/` — immutable reference template (trivial baseline agent that just
  submits `sample_submission.csv`). Never edit in place — copy it into
  `submissions/<experiment_name>/agent/` for real work.
- `.env.example` — copy to `.env` and add LLM API keys for local evaluation (local eval
  also needs a container runtime — podman preferred — with the Kaggle sandbox image
  pulled).
- `models.yaml` — the competition's model registry; `validate_submission.py` checks every
  model referenced in an agent config against this file.
 
The actual 16 synthetic datasets (`train.csv`, `test.csv`, `sample_submission.csv`,
`solution.csv`, `DATA.md` each) live at **`data/raw/train_{01-16}/`** (cookiecutter
`data/raw` convention — this part did not move). Since `run_local_eval.py` hardcodes
`<repo_root>/data/<dataset>`, `data/train_01` … `data/train_16` are **symlinks** to
`raw/train_01` … `raw/train_16` so the harness's expected flat path resolves without
duplicating ~40MB of data. **Schemas differ across datasets** — don't assume a fixed
feature-to-type mapping (e.g. `feature_1` is `ordinal` in one dataset, `count` in
another; some datasets have zero categorical columns). Type columns by inspecting their
values, not by name or position.
 
### Canonical submission layout (from `kaggle-kaggle-skill/SKILL.md`)
 
```
submissions/
└── 01_baseline/
    ├── agent/                   # agent.yaml MANDATORY at this level
    │   ├── agent.yaml
    │   ├── prompts/
    │   ├── tools/
    │   └── skills/
    ├── output/                  # eval traces for this experiment (run_local_eval.py auto-routes here)
    └── submission.zip           # packaged archive for Kaggle — one per experiment
```
 
`run_local_eval.py` auto-detects this layout (when `--submission-dir` ends in
`.../submissions/<name>/agent`) and routes trace output to `submissions/<name>/output/`
automatically.
 
## Recommended development workflow
 
1. Copy `sample_submission/` to `submissions/<experiment_name>/agent/` and edit
   `agent.yaml`, `prompts/`, `tools/`, `skills/` there.
2. `uv run python validate_submission.py --agent-dir submissions/<experiment_name>/agent`
   to check structure/spec before burning budget.
3. `uv run python run_local_eval.py --submission-dir submissions/<experiment_name>/agent
   --dataset train_XX --metric roc_auc` against several of the 16 `train_XX` sets to
   estimate performance and catch sandbox execution bugs — check several, not just one,
   since low cross-dataset variance is the actual goal.
4. Package it: `(cd submissions/<experiment_name>/agent && zip -r ../submission.zip .)`
5. `kaggle competitions submit <slug> -f submissions/<experiment_name>/submission.zip -m "<message>"`.
## Architecture patterns from the demo notebooks
 
Two reference examples (not mandatory, but they show the expected format):
 
**Simple agent**: a central `system.md` with a linear workflow (delegate EDA → plan →
train baselines → iterate with a feature-engineering skill → submit and select), plus a
`data_analyst` subagent (EDA only, no modeling) and a `feature-engineer` skill with a CLI
script for imputation + a row-mean feature.
 
**AIDE-style agent**: a root `LoopAgent` driving a `SequentialAgent` cycle
(`tree_manager` → `coding_operator` → `evaluator` → `reviewer`) that implements
Draft → Debug → Improve search over a solution tree (`solution_tree.json`), maintained by
an `aide-manager` skill (search-policy, data-preview, evaluation, and feedback-persistence
scripts). More complex, but gives fine-grained control over budget and iteration.
 
### Useful conventions to reuse (from the AIDE agent)
 
- Model scripts saved as `model_{NODE-ID}.py`, always with a **module-level docstring**
  (2-4 sentences summarizing the approach) — this becomes the "memory" injected into
  later prompts.
- The script must print `VALIDATION_SCORE: <score>` at the end of execution.
- `submission_{NODE-ID}.csv` must match `sample_submission.csv` in columns and row count.
- **Always seed 0** and **`n_jobs=1` / `thread_count=1`** on every estimator — the
  container has threading limits.
- Watch out for `early_stopping_rounds` in XGBoost (use in the estimator's init or via a
  callback), and don't assign a float directly into a `CategoricalDtype` column (cast the
  dtype first).
- Always verify feature column names match between `train.csv` and `test.csv`.
## Things to avoid
 
- Requesting tools outside the harness's allowed list.
- Path traversal in `!include` or inside skills.
- Assuming internet access or `pip install` inside the agent's sandbox — it's offline.
- Prompts that make the agent end the session too early without submitting/selecting
  anything (that zeroes out the session), or that blow through the time/token/submission
  budget.
- Overfitting to each session's **public** score (visible to the agent) — what actually
  counts is the **private** score, computed only after final selection.

## notebooks
- on notebooks/demo_notebooks there are the demo notebooks provided by competition host, you can follow those to understand how submission works and use as baseline
- on notebooks/community_notebooks i cherry-picked a few community notebooks to be used as guide and baseline. if you feel like working on one of them, copy to the notebooks/ folder and on the first cell add a markdown saying that it is "forked" from the other notebook

## remarks
- this is not a modeling competition, it's a generalization competition. You never touch the two scored datasets. Your agent does. And it runs twice (public session, private session) on data drawn from the same synthetic family as your 16 training sets. So you're not optimizing "best model on a dataset" — you're optimizing "lowest-variance, most robust recipe across a family of datasets." That reframe changes almost every design decision below.
- Deterministic robust recipe → low variance across the family + selection by internal CV (not public LB) → avoids winner's-curse overfit + tight token budget → never times out → reliably high private AUC