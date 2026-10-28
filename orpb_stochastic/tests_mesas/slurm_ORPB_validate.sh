#!/bin/bash
# SLURM submission script for validate.py on Rockfish: scores finished PMMH runs on the common
# held-out observations (VALIDATION_PERIODS in ORPB_cases.py). Submit after the calibrations
# finish; rerun whenever more runs are done (it rescores everything it matches).
#
# Submit:    sbatch slurm_ORPB_validate.sh
# Monitor:   squeue -u $USER

#SBATCH --job-name=ORPB_validate
#SBATCH --partition=shared

#SBATCH --time=4:00:00                   # ~10 min per run per core at --num-draws 200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16               # runs are validated in parallel, one per core
#SBATCH --mem=16G                        # ~0.5G per process
#SBATCH --output=logs/ORPB_validate_%j.out
#SBATCH --error=logs/ORPB_validate_%j.err


set -euo pipefail
mkdir -p logs

# ---------------------------------------------------------------------------
# Environment (same as slurm_ORPB_pmcmc.sh)
# ---------------------------------------------------------------------------
module purge
module load anaconda
conda activate /home/19megan/.conda/envs/mesas_v2_env

export MESAS_REPO_ROOT="$HOME/ORPB_19megan/orpb-sas-pmcmc"
export MESAS_DATA_ROOT="$HOME/ORPB_19megan/resolution_datasets"
RESULTS="$HOME/ORPB_19megan/ORPB_results"   # parent of the per-job result folders
OUT_DIR="$RESULTS/validation"
mkdir -p "$OUT_DIR"

# one thread per process; see slurm_ORPB_pmcmc.sh for why numba needs its own settings
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MESAS_PARALLEL=0
export NUMBA_NUM_THREADS=1

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cd "$MESAS_REPO_ROOT/orpb_stochastic/tests_mesas"

# --runs is quoted so python expands the glob: every pmmh_*.npz in every job folder. Runs
# calibrated before the held-out mask (no holdout_months in run_config) are skipped.
# --num-cores defaults to SLURM_CPUS_PER_TASK.
python -u validate.py \
    --runs "$RESULTS/*/pmmh_*.npz" \
    --out "$OUT_DIR/validation_scores_job${SLURM_JOB_ID}.csv" \
    --num-draws 200 \
    --seed 0

echo "Job ${SLURM_JOB_ID} finished. Scores in $OUT_DIR"
