# Choose the particle count N for a PMMH run on a given data window.
#
# PMMH mixes well when the noise of the particle-filter log-likelihood, sd(log Z_hat), is about
# 1-1.5 at a theta the chain actually visits. Below that, extra particles buy little; above
# about 2, the chain sticks after accepting a lucky overestimate. The noise grows with the
# number of observations, so every window length needs its own N.
#
# The theta must come from the posterior: at a poor-fitting prior draw the noise is far
# larger than anywhere the chain will spend time. So this reads a pilot PMMH run's
# pmmh_*.npz and uses its most typical post-burn-in state (the one with the median log Z_hat).
#
# Workflow for a new window:
#   1. pilot run on that window, e.g. --num-particles 50 --num-mcmc 600 --burnin 300
#      (convergence is not needed, only a post-burn-in state near the posterior)
#   2. python tune_particles.py --pilot <result_root>/pmmh_<tag>.npz \
#          --start-date ... --end-date ... --resolution ... --particles 50 100 200 400
#   3. use the recommended N for the production run
#
# The window, resolution and case must match the pilot run.
#
# Created on: 2026-09-29
import argparse
import os
import sys
import time
import types

script_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_folder)
import numpy as np   # noqa: E402

import ORPB_pmcmc_script_parallel as runner   # noqa: E402  (path setup, data loading, cases)
from orpb_stochastic.model.ssm_model import SSModel   # noqa: E402


def pilot_theta(pilot_path, theta_names):
    """The post-burn-in state with the median log Z_hat, pooled over chains."""
    pm = np.load(pilot_path)
    names = [str(n) for n in pm["theta_names"]]
    if names != list(theta_names):
        raise ValueError("pilot run estimated different parameters than this case:\n"
                         f"  pilot: {names}\n  case:  {list(theta_names)}")
    b = int(pm["num_burnin"])
    theta = pm["theta_chains"][:, b + 1:, :].reshape(-1, len(names))
    logZ = pm["logZ_chains"][:, b + 1:].ravel()
    i = np.argsort(logZ)[len(logZ) // 2]
    return theta[i], logZ[i]


def main():
    p = argparse.ArgumentParser(description="Choose N (particles) for a PMMH run")
    p.add_argument("--pilot", required=True, help="pmmh_*.npz from a pilot run on this window")
    p.add_argument("--particles", type=int, nargs="+", default=[50, 100, 200, 400])
    p.add_argument("--reps", type=int, default=20,
                   help="filter runs per N; the sd estimate is good to about +-15%% at 20")
    p.add_argument("--target", type=float, default=1.5, help="largest acceptable sd(log Z)")
    p.add_argument("--case-name", default="storage_q_ug_et_u_cp")
    p.add_argument("--start-date", default="2015-01-01")
    p.add_argument("--end-date", default="2015-12-31")
    p.add_argument("--resolution", default="daily",
                   choices=["hourly", "daily", "weekly", "biweekly", "monthly"])
    p.add_argument("--data-root", default=os.environ.get(
        "MESAS_DATA_ROOT", "/Users/simon/Desktop/ORPB_resolution_datasets"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    run_args = types.SimpleNamespace(**vars(args), sampler="pmmh", update_theta_dist=False)
    df, n_spinup = runner.load_data(run_args)
    n_obs = int(df["ORPB 18O"].notna().sum())
    print(f"window {args.start_date}..{args.end_date} ({args.resolution}), "
          f"{n_obs} stream observations")

    rows = []
    for N in args.particles:
        run_args.num_particles = N
        mi, _, _ = runner.build_model_interface(run_args, df, n_spinup)
        if not rows:
            theta, pilot_logZ = pilot_theta(args.pilot, mi._theta_to_estimate)
            print(f"theta from pilot (log Z_hat {pilot_logZ:.1f}):")
            for name, v in zip(mi._theta_to_estimate, theta):
                print(f"    {name.replace('#@#', ' / '):55s} {v:.4g}")
            print(f"\n{'N':>6} {'sd(log Z)':>10} {'s per filter':>13}")
        model = SSModel(mi, num_parameter_samples=1, len_parameter_MCMC=1)
        t0 = time.time()
        sd = model.estimate_logZ_sd(theta, reps=args.reps)
        per = (time.time() - t0) / args.reps
        rows.append((N, sd, per))
        print(f"{N:6d} {sd:10.2f} {per:13.2f}", flush=True)

    ok = [N for N, sd, _ in rows if sd <= args.target]
    if ok:
        print(f"\nrecommended: --num-particles {min(ok)}  "
              f"(smallest N with sd(log Z) <= {args.target})")
    else:
        print(f"\nno N tried reaches sd(log Z) <= {args.target}. sd falls roughly as "
              f"1/sqrt(N), so try about {int(rows[-1][0] * (rows[-1][1] / args.target) ** 2)}.")
    print("s per filter is approximate and mostly excludes the mesas solve for a new theta, "
          "which each PMMH step also pays.")


if __name__ == "__main__":
    main()
