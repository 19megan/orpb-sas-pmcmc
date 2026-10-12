#!/bin/bash
# SLURM script for tune_particles.py: choose N (--num-particles) for a PMMH window on Rockfish.
#
# Workflow (papers/codebase_evaluation_PMMH.md Section 7.2):
#   1. pilot:      sbatch slurm_ORPB_pmcmc.sh with e.g. --num-particles 50 --num-mcmc 600 --burnin 300
#   2. tune:       set PILOT_JOB / PILOT_TAG below to that pilot, then sbatch slurm_tune_particles.sh
#   3. production: sbatch slurm_ORPB_pmcmc.sh with the recommended --num-particles
#
# The result (sd(log Z) per N and the recommended N) is printed to logs/tune_particles_<jobid>.out.
#
# Serial: the filter runs in one process, so 1 CPU is enough and asking for more only lengthens
# the queue wait.

#SBATCH --job-name=tune_particles
#SBATCH --partition=shared
#SBATCH --time=1:00:00                    # 20 filters per N; raise if a 4-year window at large N runs out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G                          # particle arrays grow as N x T; raise for long windows at large N
#SBATCH --output=logs/tune_particles_%j.out
#SBATCH --error=logs/tune_particles_%j.err

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

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MESAS_PARALLEL=0
export NUMBA_NUM_THREADS=1

# ---------------------------------------------------------------------------
# Pilot run to tune from. EDIT THESE for each window.
# ---------------------------------------------------------------------------
# PILOT_JOB is the PILOT run's SLURM job ID (not this job's: SLURM_JOB_ID here is the tuning
# job itself). PILOT_TAG is the --run-tag the pilot was run with.
# slurm_ORPB_pmcmc.sh writes results to ORPB_results/<job ID>/pmmh_<case><run-tag>.npz.
CASE_NAME="storage_q_ug_et_u_cp"
PILOT_JOB="00000000"
PILOT_TAG="_D6M_job${PILOT_JOB}"
PILOT="$HOME/ORPB_19megan/ORPB_results/${PILOT_JOB}/pmmh_${CASE_NAME}${PILOT_TAG}.npz"

if [ ! -f "$PILOT" ]; then
    echo "pilot file not found: $PILOT" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cd "$MESAS_REPO_ROOT/orpb_stochastic/tests_mesas"

# --case-name, --start-date, --end-date and --resolution MUST match the pilot run.
# --particles: real 3-month daily data needed 800; longer windows need more, so extend the
# list upward for them (sd(log Z) falls roughly as 1/sqrt(N)).
python -u tune_particles.py \
    --pilot "$PILOT" \
    --case-name "$CASE_NAME" \
    --start-date 2015-01-01 \
    --end-date 2015-06-30 \
    --resolution daily \
    --particles 200 400 800 1600 3200 \
    --reps 20 \
    --seed 5

echo "Job ${SLURM_JOB_ID} finished. Recommended N is printed above."
