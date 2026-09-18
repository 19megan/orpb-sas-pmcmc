# Running this repo on Rockfish

How the ORPB pMCMC code is set up on Rockfish (JHU ARCH cluster), and the day-to-day
workflow for running jobs there. Written 2026-09-09, after migrating off the old
scp'd `mesas` monorepo.

**The model in one sentence:** the repo is the single source of truth, GitHub is the
transport, and Rockfish holds a clone you never hand-edit.

---

## 1. Why it works this way

Previously, `~/ORPB_19megan/mesas` on Rockfish was an `scp` copy of the old monorepo, and
every change had to be made twice — once locally, once there. Two things went wrong with
that:

- The copies drifted. The Rockfish `ORPB_pmcmc_script_parallel.py` ended up *ahead* of the
  local one (correct `_cp` case name, `update_theta_dist=False`, bfill data file) while the
  local copy kept older values.
- An `scp` copy has no `.git`, so `_get_git_commit` in `functions/run_config.py` returned
  `"unknown"`. **Every `run_config.json` written on Rockfish before this migration has no
  usable commit hash**, which makes those results hard to tie back to the code that produced
  them.

A clone fixes both: one edit, one commit, and the SHA lands in every snapshot automatically.

---

## 2. Layout on Rockfish

```
~/ORPB_19megan/              <- submit jobs from HERE
├── run_ORPB_pmcmc.sh        <- YOUR editable slurm file (untracked, outside the repo)
├── logs/                    <- SLURM .out/.err, named by job id
├── orpb-sas-pmcmc/          <- git clone of github.com/19megan/orpb-sas-pmcmc
│   └── orpb_stochastic/
│       └── tests_mesas/     <- slurm_ORPB_pmcmc.sh lives here as a clean template
├── resolution_datasets/     <- input data                     ($MESAS_DATA_ROOT)
├── ORPB_results/<job_id>/   <- one directory per SLURM job     ($MESAS_RESULT_ROOT)
└── mesas_OLD/               <- retired scp'd monorepo; delete when confident
```

Data, results, logs, and your editable slurm file all live **outside** the clone. Nothing
the model reads or writes is tracked by git, so `git pull` can never touch your data,
clobber a result, or collide with a run script.

Keeping `run_ORPB_pmcmc.sh` at `~/ORPB_19megan/` rather than inside the repo works because
the script `cd`s to `$MESAS_REPO_ROOT/orpb_stochastic/tests_mesas` itself — nothing in it
depends on where the script file sits. The payoff is that `git status` inside the clone
stays completely clean, with no `.gitignore` entries needed.

---

## 3. One-time setup (already done; recorded for reference)

```bash
module load anaconda
conda activate /home/19megan/.conda/envs/mesas_v2_env

# confirm mesas v2, not the old v1
python -c "import mesas; print(mesas.__file__)"

cd ~/ORPB_19megan
git clone https://github.com/19megan/orpb-sas-pmcmc.git
cd orpb-sas-pmcmc
pip install -e .            # makes `orpb_stochastic` importable from anywhere

python -c "import orpb_stochastic, mesas; print('ok')"
```

`pip install -e .` points the install at the clone, so a `git pull` updates the installed
package too — no reinstall needed after pulling.

### Portability changes that made one file serve both machines

- **`ORPB_pmcmc_script_parallel.py`** reads its data path from `MESAS_DATA_ROOT` (via the
  `--data-root` default), falling back to the local macOS path so nothing changes on the
  laptop. The spinup is not a separate file: `prepend_spinup` in `ORPB_cases.py` prepends
  `SPINUP_YEAR` (2014) from the same dataset, so there is no spinup CSV or tag to keep in sync.
- **`slurm_ORPB_pmcmc.sh`** loads the mesas v2 env and `cd`s to
  `$MESAS_REPO_ROOT/orpb_stochastic/tests_mesas` (was `mesas.stochastic/tests_mesas`).

The `cd` is load-bearing: `tests_mesas/` has no `__init__.py`, so `ORPB_cases` and
`ORPB_mesas_interface` are only importable as top-level modules from inside that directory.

---

## 4. The everyday loop

```
LAPTOP                                    ROCKFISH
──────                                    ────────
edit tracked files                        cd ~/ORPB_19megan/orpb-sas-pmcmc
git commit                                git pull
git push            ──────────────────>   cd ~/ORPB_19megan
                                          nano run_ORPB_pmcmc.sh    (untracked)
                                          sbatch run_ORPB_pmcmc.sh
```

**You never edit a tracked file on Rockfish.** All code changes originate on the laptop.

The one exception is deliberate: `~/ORPB_19megan/run_ORPB_pmcmc.sh` (Section 6), which
exists precisely so you have somewhere on Rockfish to edit freely.

### Commit is not enough — you must push

`git commit` only writes to your local history. Rockfish pulls from GitHub, so a change is
invisible there until `git push`. If a pull brings nothing and you expected a change, check
`git status` on the laptop for unpushed commits or uncommitted edits.

---

## 5. When do I need to `git pull`?

**Only after you have pushed something.** A pull with nothing upstream is a no-op that
takes about a second.

In practice: **just run it every time before submitting.** It is cheap, and it removes the
"did I actually push that?" question entirely. The useful habit is:

```bash
cd ~/ORPB_19megan/orpb-sas-pmcmc
git pull
git log -1 --oneline        # what am I about to run?
```

You do **not** need to pull between two runs of the same code — resubmitting `sbatch` uses
whatever is on disk right now.

### Whichever you do, provenance is automatic

`save_run_config` records `git rev-parse --short HEAD` into every `run_config.json`
(`functions/run_config.py`). So even if you forget to pull, you can always reconstruct
exactly which commit produced a given result:

```bash
grep git_commit ~/ORPB_19megan/ORPB_results/<job_id>/run_config_*.json
git show <that_sha>
```

---

## 6. Per-run edits: `run_ORPB_pmcmc.sh`

Editing the tracked `slurm_ORPB_pmcmc.sh` on Rockfish would make `git pull` refuse. It stays
a clean template; the working copy lives outside the repo at
`~/ORPB_19megan/run_ORPB_pmcmc.sh`.

Created once with:

```bash
mkdir -p ~/ORPB_19megan/logs
cd ~/ORPB_19megan
cp orpb-sas-pmcmc/orpb_stochastic/tests_mesas/slurm_ORPB_pmcmc.sh run_ORPB_pmcmc.sh
```

Thereafter, every run is: edit it, submit it, from `~/ORPB_19megan/`.

```bash
cd ~/ORPB_19megan
nano run_ORPB_pmcmc.sh     # dates, N/D/L, --run-tag, walltime, --notes
sbatch run_ORPB_pmcmc.sh
```

### Which changes go where

| Change | Where |
|---|---|
| Dates, N/D/L, case, seed, run tag, notes, walltime, memory | `run_ORPB_pmcmc.sh` on Rockfish |
| New CLI flags, env vars, path logic, conda env, anything in Python | Laptop → commit → push → `git pull` |

Because there is one working copy rather than one per run, the file always reflects your
*most recent* run, not past ones. That is fine — provenance lives in
`ORPB_results/<job_id>/run_config_*.json`, which records N/D/L, dates, seed, priors, notes,
and the git SHA for every job.

**After a `git pull` that changed the template**, re-check your working copy: `git pull`
updates `slurm_ORPB_pmcmc.sh` but never touches `run_ORPB_pmcmc.sh`. Diff them and hand-copy
anything structural:

```bash
diff ~/ORPB_19megan/run_ORPB_pmcmc.sh \
     ~/ORPB_19megan/orpb-sas-pmcmc/orpb_stochastic/tests_mesas/slurm_ORPB_pmcmc.sh
```

Expect the per-run values to differ — you are looking for new exports, changed paths, or new
flags, not for your dates to match.

### Smoke test before a long run

Worth doing after any change to paths, priors, the conda env, or the mesas version — the
failure modes there surface at import or in the first few seconds, and it is much cheaper to
find them than to wait out a queue slot and a 2-hour job.

**1. Imports, straight on the login node** (no SLURM, ~10 seconds):

```bash
cd ~/ORPB_19megan/orpb-sas-pmcmc/orpb_stochastic/tests_mesas
python -c "import ORPB_cases; print('cases ok'); import ORPB_mesas_interface; print('interface ok')"
```

This catches a wrong conda env or a broken import. A bad `MESAS_DATA_ROOT` surfaces in the
tiny job below, when the data CSV is read.

**2. A tiny job** — copy to `run_smoke.sh` and set:

```
#SBATCH --time=00:20:00
--num-particles 3 --num-samples 3 --num-mcmc 2
--start-date 2015-01-01 --end-date 2015-01-31
--run-tag "_smoke"
```

```bash
sbatch run_smoke.sh
```

This is the first thing that exercises mesas end-to-end. In particular it confirms that
stripping the `prior` key from `solute_parameters` satisfies mesas v2's strict key
validation at `model.py:734` — the C_old uniform-prior path (§12.2 of
`papers/codebase_evaluation_ORPB.md`) was verified in isolation but not against a live
mesas model.

Success looks like: the job exits 0, `ORPB_results/<job_id>/` contains `theta_*.csv` and
`ess_record_*.csv`, and `run_config_*.json` shows a real `git_commit` rather than
`"unknown"`.

---

## 7. Environment variables

Set in the slurm script, read by the Python code. **All of them must be exported before the
`python` line** — an export after it has no effect on the run.

| Variable | Read by | Purpose |
|---|---|---|
| `MESAS_REPO_ROOT` | slurm script | Repo location; used for the `cd` |
| `MESAS_DATA_ROOT` | `--data-root` default | Input CSVs |
| `MESAS_RESULT_ROOT` | `--result-root` default | Per-job output directory |
| `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS` | BLAS | Set to 1 so the D worker processes don't oversubscribe cores |

Per-run settings that are **CLI flags**, not env vars: `--num-particles` (N),
`--num-samples` (D), `--num-mcmc` (L), `--case-name`, `--start-date`, `--end-date`,
`--resolution`, `--run-tag`, `--seed`, `--notes`.

---

## 8. Gotchas

**Every env var must be exported before the `python` line.** An export placed after it has
no effect on the run. This once bit the now-removed `ORPB_SPINUP_TAG` (it sat on the last
line of the script, after the run), and the failure was silent: the code fell back to a
default. Worth re-checking whenever you add a variable.

**The calibration window must start after the spinup year.** `SPINUP_YEAR = '2014'` in
`ORPB_cases.py` is prepended to every run and excluded from the likelihood. A
`--start-date` inside 2014 raises a `ValueError` at data load rather than duplicating data.

**`logs/` must exist in the submit directory before the first `sbatch`.** SLURM opens
`--output=logs/...` before your script runs, so the `mkdir -p logs` inside the script is too
late on a first run. Already created at `~/ORPB_19megan/logs`; only matters if you submit
from somewhere else.

**Always `sbatch` from `~/ORPB_19megan/`.** Both `--output=logs/...` and `--error=logs/...`
are relative to the submit directory, not to where the script lives. Submitting from
elsewhere scatters logs or fails outright.

**`git pull` refuses: "local changes would be overwritten".** You edited a tracked file
inside the clone. Either discard it (`git checkout -- <file>`) or, to keep it, move it out of
the repo (`mv <file> ~/ORPB_19megan/`) and then pull. This should not happen with
`run_ORPB_pmcmc.sh` living outside the clone.

**Set `D <= --cpus-per-task`.** The parallel runner caps `num_cores` at
`$SLURM_CPUS_PER_TASK` automatically, so a larger D just serializes and inflates walltime.

**Keep `update_theta_dist=False`.** The parallel script sets it in `build_model_interface`.
It must match the serial script — see §12.4 of `papers/codebase_evaluation_ORPB.md` for why,
and note that `ssm_model.py:95` means `True` would only half-work anyway.

**`mesas_OLD/` is still on disk.** Delete once you have a successful run from the clone and
have salvaged anything you still need.

---

## 9. Quick reference

```bash
# 1. pull whatever you pushed from the laptop, and see what you're about to run
cd ~/ORPB_19megan/orpb-sas-pmcmc && git pull && git log -1 --oneline

# 2. edit + submit (always from ~/ORPB_19megan)
cd ~/ORPB_19megan
nano run_ORPB_pmcmc.sh
sbatch run_ORPB_pmcmc.sh

# 3. monitor
squeue -u $USER
tail -f ~/ORPB_19megan/logs/ORPB_pmcmc_<jobid>.out
scancel <jobid>

# 4. results
ls ~/ORPB_19megan/ORPB_results/<jobid>/
grep git_commit ~/ORPB_19megan/ORPB_results/<jobid>/run_config_*.json

# 5. bring results back (run from the laptop)
scp -r <rockfish>:~/ORPB_19megan/ORPB_results/<jobid> /Users/simon/Desktop/ORPB_results/
```
