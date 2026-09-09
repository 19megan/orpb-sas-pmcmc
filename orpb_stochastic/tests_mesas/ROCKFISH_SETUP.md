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
~/ORPB_19megan/
├── orpb-sas-pmcmc/          <- git clone of github.com/19megan/orpb-sas-pmcmc
│   └── orpb_stochastic/
│       └── tests_mesas/     <- run jobs from HERE
├── resolution_datasets/     <- input data + sT_mT_init_*.csv  ($MESAS_DATA_ROOT)
├── ORPB_results/<job_id>/   <- one directory per SLURM job     ($MESAS_RESULT_ROOT)
└── mesas_OLD/               <- retired scp'd monorepo; delete when confident
```

Data and results live **outside** the repo. Nothing the model reads or writes is tracked by
git, so `git pull` can never touch your data or clobber a result.

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

- **`ORPB_cases.py`** reads its spinup path and tag from the environment, falling back to
  the local macOS paths so nothing changes on the laptop:
  ```python
  _data_root = os.environ.get("MESAS_DATA_ROOT", "/Users/simon/Desktop/ORPB_resolution_datasets")
  tag        = os.environ.get("ORPB_SPINUP_TAG", "D_std_3M")
  sTmT       = pd.read_csv(f'{_data_root}/sT_mT_init_{tag}.csv')
  ```
  This runs at **import** time, so a bad path here fails before anything else executes.
- **`slurm_ORPB_pmcmc.sh`** loads the mesas v2 env and `cd`s to
  `$MESAS_REPO_ROOT/orpb_stochastic/tests_mesas` (was `mesas.stochastic/tests_mesas`).

The `cd` is load-bearing: `tests_mesas/` has no `__init__.py`, so `ORPB_cases` and
`ORPB_mesas_interface` are only importable as top-level modules from inside that directory.

---

## 4. The everyday loop

```
LAPTOP                                    ROCKFISH
──────                                    ────────
edit tracked files
git commit
git push            ──────────────────>   git pull
                                          edit run_<name>.sh   (untracked)
                                          sbatch run_<name>.sh
```

**You never edit a tracked file on Rockfish.** All code changes originate on the laptop.

The one exception is deliberate: the untracked per-run slurm copy (Section 6), which exists
precisely so you have somewhere on Rockfish to edit freely.

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

## 6. Per-run edits: the untracked copy

Editing the tracked `slurm_ORPB_pmcmc.sh` on Rockfish will make `git pull` refuse. Instead,
keep it as a clean template and work from a copy:

```bash
cd ~/ORPB_19megan/orpb-sas-pmcmc/orpb_stochastic/tests_mesas
cp slurm_ORPB_pmcmc.sh run_D6M.sh     # untracked
# edit run_D6M.sh: dates, N/D/L, run tag, walltime, notes
sbatch run_D6M.sh
```

Each run keeps its own script on disk, so you can see exactly how a past job was invoked.
Structural changes (new CLI flags, env vars, path changes) still belong in the tracked
template on the laptop, pushed and pulled like everything else.

> Consider adding `run_*.sh` and `logs/` to `.gitignore` so `git status` on Rockfish stays
> clean. Not currently ignored.

---

## 7. Environment variables

Set in the slurm script, read by the Python code. **All of them must be exported before the
`python` line** — an export after it has no effect on the run.

| Variable | Read by | Purpose |
|---|---|---|
| `MESAS_REPO_ROOT` | slurm script | Repo location; used for the `cd` |
| `MESAS_DATA_ROOT` | `ORPB_cases.py`, `--data-root` default | Input CSVs and `sT_mT_init_*.csv` |
| `MESAS_RESULT_ROOT` | `--result-root` default | Per-job output directory |
| `ORPB_SPINUP_TAG` | `ORPB_cases.py` | Picks `sT_mT_init_<TAG>.csv`; defaults to `D_std_3M` |
| `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS` | BLAS | Set to 1 so the D worker processes don't oversubscribe cores |

Per-run settings that are **CLI flags**, not env vars: `--num-particles` (N),
`--num-samples` (D), `--num-mcmc` (L), `--case-name`, `--start-date`, `--end-date`,
`--resolution`, `--run-tag`, `--seed`, `--notes`.

---

## 8. Gotchas

**`ORPB_SPINUP_TAG` must be exported before the python call.** As of 2026-09-09 the tracked
slurm script exports it on the last line, *after* the run, where it does nothing. The import
then silently falls back to the `D_std_3M` default — a wrong-spinup run with no error, if
that file happens to exist. Move it up with the other exports.

**`logs/` must exist in the submit directory before the first `sbatch`.** SLURM opens
`--output=logs/...` before your script runs, so the `mkdir -p logs` inside the script is too
late on a first run. `mkdir -p logs` by hand once.

**`git pull` refuses: "local changes would be overwritten".** You edited a tracked file on
Rockfish. Either discard it (`git checkout -- <file>`) or, if you want to keep it, rename it
to an untracked name (`mv slurm_ORPB_pmcmc.sh run_whatever.sh`) and then pull.

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
# submit
cd ~/ORPB_19megan/orpb-sas-pmcmc && git pull
cd orpb_stochastic/tests_mesas
sbatch run_D6M.sh

# monitor
squeue -u $USER
tail -f logs/ORPB_pmcmc_<jobid>.out
scancel <jobid>

# results
ls ~/ORPB_19megan/ORPB_results/<jobid>/

# bring results back (run from the laptop)
scp -r <rockfish>:~/ORPB_19megan/ORPB_results/<jobid> /Users/simon/Desktop/ORPB_results/
```
