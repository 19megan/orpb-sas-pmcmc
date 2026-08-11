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

#SBATCH --time=2:00:00                   # walltime; bump up for longer timeseries
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
# If you built a CUDA-free env named 'mesas', activate it. Otherwise replace.
#source activate mesas
conda activate /home/19megan/.conda/envs/mesas_env

# Repo and data locations. EDIT THESE for your Rockfish layout.
export MESAS_REPO_ROOT="$HOME/ORPB_19megan/mesas"
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

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cd "$MESAS_REPO_ROOT/mesas.stochastic/tests_mesas"

# Choose D <= $SLURM_CPUS_PER_TASK for full utilization. The script auto-caps
# num_cores to SLURM_CPUS_PER_TASK so you don't need to pass --num-cores.
python -u ORPB_pmcmc_script_parallel.py \
    --parallel \
    --num-particles 25 \
    --num-samples 24 \
    --num-mcmc 100 \
    --case-name storage_q_ug_et_u_cp \
    --start-date 2015-01-01 \
    --end-date 2015-06-30 \
    --resolution daily \
    --run-tag "_D6M_job${SLURM_JOB_ID}" \
    --no-plot \
    --seed 5 \
    --notes "Testing convergence for case: W6M. Made correct prior sigma filled C in equal to 0.01 for weekly resolution. Running pMCMC with sT and mT init set from 5yr looped 2014 spinup but capped at 150 days max age using D_std version." \

echo "Job ${SLURM_JOB_ID} finished. Results in $MESAS_RESULT_ROOT"

#NOTE: Need to manually change run tag in ORPB_cases.py since it can't be passed
