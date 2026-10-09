#!/bin/bash
# SLURM submission script for the ORPB pMCMC runner on Rockfish.
#
# Submit:    sbatch slurm_ORPB_pmcmc.sh
# Monitor:   squeue -u $USER
# Cancel:    scancel <jobid>
#
# Rockfish partitions (as of 2025): "shared" for <=1 node multi-core jobs,
# "parallel" for multi-node MPI. We're using "shared" because the parallelism
# here is intra-node (multiprocessing.Pool over D chains).

#SBATCH --job-name=ORPB_pmcmc
#SBATCH --partition=shared

#SBATCH --time=4:00:00                   # walltime; pmmh ~ num-mcmc x one mesas solve; bump for longer timeseries
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24                # match or exceed --num-samples (D) if you want it to run fast otherwise this will increase wall time
#SBATCH --mem=12G                         # raise for multi-year hourly runs
#SBATCH --output=logs/ORPB_pmcmc_%j.out
#SBATCH --error=logs/ORPB_pmcmc_%j.err


set -euo pipefail
mkdir -p logs

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# Rockfish uses Lmod. Load anaconda then activate your env.
module purge
module load anaconda

#source activate mesas v2
conda activate /home/19megan/.conda/envs/mesas_v2_env

# Repo and data locations. EDIT THESE for your Rockfish layout.
export MESAS_REPO_ROOT="$HOME/ORPB_19megan/orpb-sas-pmcmc"
export MESAS_DATA_ROOT="$HOME/ORPB_19megan/resolution_datasets"
export MESAS_RESULT_ROOT="$HOME/ORPB_19megan/ORPB_results/${SLURM_JOB_ID}"
mkdir -p "$MESAS_RESULT_ROOT"

# Keep BLAS/OpenMP from oversubscribing inside each worker process. Each of the
# D chains will run sequential numpy ops; one thread per worker is the safe
# default. Tune up only if you have spare cores beyond D.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Same reasoning for numba, which mesas v2 uses for the solver and which has its OWN thread
# pool -- it is not covered by OMP_NUM_THREADS. Left at its default, every one of the D
# worker processes starts a pool of (cores) threads, so D x cores threads fight over the
# allocation and the idle ones busy-wait (CPU looks 95% used while nothing progresses).
export MESAS_PARALLEL=0      # compile the solver's hot loop serially (mesas reads this at import)
export NUMBA_NUM_THREADS=1

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cd "$MESAS_REPO_ROOT/orpb_stochastic/tests_mesas"

# Choose D <= $SLURM_CPUS_PER_TASK for full utilization. The script auto-caps
# num_cores to SLURM_CPUS_PER_TASK so you don't need to pass --num-cores.
#
# --sampler pmmh (default): --num-samples = number of independent chains,
# --num-mcmc = steps per chain, --burnin = adaptive steps discarded (default half).
# Each step costs about one mesas solve per chain, so wall time ~ num-mcmc x (one solve).
# --num-particles: set per window with tune_particles.py from a short pilot run (see
# papers/codebase_evaluation_PMMH.md 23.5). Real 3-month daily data needed N=800; 50 left
# the chains stuck (acceptance 1.5%). Longer windows need their own tuning.
# --update-theta-dist is a switch: pass it to turn it on, omit it for off (it takes no value;
# "--update-theta-dist False" is an argparse error). pmmh ignores it.
python -u ORPB_pmcmc_script_parallel.py \
    --parallel \
    --sampler pmmh \
    --num-particles 50 \
    --num-samples 24 \
    --num-mcmc 3000 \
    --burnin 1000 \
    --case-name storage_q_ug_et_u_cp \
    --start-date 2015-01-01 \
    --end-date 2015-06-30 \
    --resolution daily \
    --run-tag "_D6M_job${SLURM_JOB_ID}" \
    --no-plot \
    --seed 5 \
    --notes "PMMH, 2014 spinup prepended (max_age = 1 year)." \

echo "Job ${SLURM_JOB_ID} finished. Results in $MESAS_RESULT_ROOT"

#NOTE: --start-date/--end-date are the calibration window. SPINUP_YEAR in ORPB_cases.py is
#prepended automatically and excluded from the likelihood; the window must start after it.