# Setup — one-time, per person

Four people, one repository, **no pull requests**. Each person works on their own
branch, merges into `main` when their tests pass, and commits under their own git
identity.

---

## 0. One person only: create the GitHub repository

This machine has git but not the GitHub CLI, so use the web UI (or install `gh`
if you prefer the terminal).

**Web UI route**

1. <https://github.com/new> → Repository name `pathsnap` → **Public** (required:
   Deliverables says "GitHub Repository (made public)") → do **not** initialise
   with a README, `.gitignore` or licence.
2. Add the other three as collaborators: *Settings → Collaborators → Add people*.
3. Then, from this folder, push the scaffold:

```powershell
cd pathsnap
git init -b main
git add -A
git commit -m "chore: project scaffold, shared contracts and team plan"
git remote add origin https://github.com/<YOUR-USERNAME>/pathsnap.git
git push -u origin main
```

**Terminal route (if you install `gh`)**

```powershell
winget install --id GitHub.cli
gh auth login
cd pathsnap
git init -b main
git add -A
git commit -m "chore: project scaffold, shared contracts and team plan"
gh repo create pathsnap --public --source . --remote origin --push
gh repo edit --add-collaborator <teammate1>,<teammate2>,<teammate3>
```

Verify: the repo is public, all four accounts can open it, and `data/raw/` is
**not** in it. If you see data files on GitHub, stop and fix `.gitignore` before
anyone clones.

---

## 1. Everyone: set your git identity

This is how your commits get attributed to you. Do it once per machine.
**Use the email address attached to your GitHub account** — a wrong email means
the commit shows up as an unlinked grey avatar and nobody, human or otherwise,
can tell it was yours.

```powershell
git config --global user.name  "Your Real Name"
git config --global user.email "you@example.com"

# check what GitHub will see
git config --global --list | Select-String "user\."
```

> If you use a GitHub `noreply` address, use the full
> `12345678+username@users.noreply.github.com` form, not just the username.

---

## 2. Everyone: clone and get running

```powershell
git clone https://github.com/<owner>/pathsnap.git
cd pathsnap

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy the dataset in (see `data/README.md` for the expected layout):

```powershell
Copy-Item -Recurse "<path to>\Student folder\train"        data\raw\train
Copy-Item -Recurse "<path to>\Student folder\student_test" data\raw\student_test
```

Verify everything before writing code:

```powershell
python scripts/check_data.py
pytest -q
```

Expected: `80` train trips, `10` student-test trips, no validation issues, and all
tests passing. If `pytest` fails on the scaffold, tell the team before doing
anything else — it means the shared contract is broken.

---

## 3. Branch workflow (no PRs)

Create **one long-lived branch** named after your workstream and work there.
Do not work on `main`; it should always be in a working state.

```powershell
# once
git checkout -b feat/p2-models

# every session
git pull origin main          # keep up to date
git status                    # see what you touched
git add -A
git commit -m "feat(models): logistic regression baseline with group-aware CV"
git push -u origin feat/p2-models
```

When your workstream is green (tests pass, results committed), merge it yourself:

```powershell
git checkout main
git pull origin main
git merge --no-ff feat/p2-models -m "merge feat/p2-models: models 1-2 + eval harness"
pytest -q
git push origin main
```

Then keep working on your branch and repeat. `--no-ff` keeps it obvious in the
history which workstream produced which change.

Branch names to use:

| Person | Branch |
|---|---|
| P1 | `feat/p1-data-features` |
| P2 | `feat/p2-models-eval` |
| P3 | `feat/p3-models-tuning` |
| P4 | `feat/p4-decoder-api-ui` |

### Merge conflicts

They will happen in exactly two places: `requirements.txt` and
`experiments/results/`. Both are append-only in practice, so resolve by keeping
**both** sides:

```powershell
git checkout main
git pull origin main
git merge feat/p2-models        # conflict
# edit the file, keep both blocks, remove the <<<<<<< markers
git add requirements.txt
git commit
```

Do **not** resolve a conflict by reverting someone else's work. If you are unsure,
call them.

---

## 4. What gets committed and what does not

| Committed | Not committed |
|---|---|
| All source in `src/`, `scripts/`, `tests/`, `ui/` | `data/raw/` (85 MB, instructor-owned) |
| `experiments/results/*.json`, `*.csv` | `experiments/logs/*.log` |
| `experiments/configs/*.json` | `artifacts/*.joblib` (force-add at submission) |
| `reports/maps/*.html` (small, required) | `.env`, any key or token |
| `docs/`, `README.md`, `requirements.txt` | `__pycache__/`, `.venv/` |

Results are committed on purpose: the report must be verifiable from the
repository, and the brief asks you to retain experiment logs, seeds,
configurations and results.

---

## 5. Secrets

Never commit credentials, API keys or tokens. If you need a local setting, put it
in `.env` (already gitignored) and load it with `python-dotenv`.

If you accidentally commit a secret: rotate/revoke it immediately. Deleting the
commit is not enough — it stays in the history and the repo is public.

---

## 6. Submission checklist

Run through this as a team before you submit. Every line maps to an explicit
requirement in the brief.

- [ ] Repository is **public** and all four members can open it
- [ ] `python scripts/check_data.py` and `pytest -q` pass on a fresh clone
- [ ] README has installation, execution and deployment instructions that actually work
- [ ] Folders are clearly named and match the required-components table
- [ ] All five models present; `assert_five_models` passes
- [ ] `experiments/results/comparison.csv` has five real rows
- [ ] `experiments/results/trials.csv` (Bayesian search record) committed
- [ ] Model weights + `feature_schema.json` + `decoder_config.json` force-added
- [ ] Folium map HTML committed **and** at least one screenshot in the PDF
- [ ] No `.route` file anywhere under `data/processed/` or in any feature cache
- [ ] No secrets, no `.env`, no API keys in the history
- [ ] API screenshot from the receiving laptop's terminal captured for the report
- [ ] UI screenshot captured for the report
- [ ] Every member can explain their own section — rehearse the viva table in
      `docs/TEAM_PLAN.md` §5 out loud, once, with the group listening
