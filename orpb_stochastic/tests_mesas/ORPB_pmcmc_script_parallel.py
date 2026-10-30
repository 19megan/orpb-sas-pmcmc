# Parallel/HPC version of ORPB_pmcmc_script.py.
# Mirrors the serial script's computation but adds:
#   - CLI args for N, D, L, case, date range, paths, run tag
#   - --parallel flag (calls model.run_particle_Gibbs_parallel)
#   - __main__ guard (required for multiprocessing on Windows / spawn)
#   - configurable paths via env vars (MESAS_REPO_ROOT, MESAS_DATA_ROOT,
#     MESAS_RESULT_ROOT) so the same file runs on your laptop and on Rockfish
#   - headless plotting (writes PDFs; no plt.show)
#
# The original ORPB_pmcmc_script.py is unchanged.
#
# Examples
#   Local serial (matches your old defaults):
#       python ORPB_pmcmc_script_parallel.py
#   Local parallel:
#       python ORPB_pmcmc_script_parallel.py --parallel
#   On Rockfish: see slurm_ORPB_pmcmc.sh
#
# Created on: 2026-05-19
# %%
import argparse
import os
import sys


# ----------------------------------------------------------------------------
# PATH SETUP
# ----------------------------------------------------------------------------
script_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_folder)

# Repo root: env var wins so the same script works on Windows and Rockfish.
DEFAULT_REPO_ROOT = (
    r"C:\Users\simon\Desktop\mesas"
    if sys.platform.startswith("win")
    else os.path.abspath(os.path.join(script_folder, "..", ".."))
)
REPO_ROOT = os.environ.get("MESAS_REPO_ROOT", DEFAULT_REPO_ROOT)
SAS_FOLDER = os.path.join(REPO_ROOT, "mesas", "sas")
STOCHASTIC_FOLDER = os.path.join(REPO_ROOT, "mesas.stochastic")

# Windows needs the sas dll directory on PATH for compiled .pyd extensions.
if sys.platform.startswith("win"):
    os.environ["PATH"] = SAS_FOLDER + os.pathsep + os.environ.get("PATH", "")

for p in (REPO_ROOT, STOCHASTIC_FOLDER):
    if p not in sys.path:
        sys.path.insert(0, p)

# Clear cached imports so edits to these modules take effect on re-run.
for mod in (
    "ORPB_mesas_interface",
    "mesas.sas.model",
    "mesas.sas.utils_chain",
    "mesas.sas.ssm_model",
    "mesas.sas.specs",
    "utils",
):
    sys.modules.pop(mod, None)


# ----------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from ORPB_mesas_interface import ModelInterfaceMesas
from orpb_stochastic.model.ssm_model import SSModel
from mesas.sas.model import Model as SAS_Model

from ORPB_cases import *  # noqa: F401,F403  (theta_init constants)
from orpb_stochastic.functions.run_config import save_run_config, set_run_seed

#ignore warnings on division by zero in normalize_over_interval
np.seterr(divide='ignore', invalid='ignore')


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ORPB pMCMC runner (parallel/HPC variant)")
    p.add_argument("--parallel", action="store_true",
                   help="Run the D chains across multiple processes")
    p.add_argument("--num-cores", type=int, default=None,
                   help="Worker count for --parallel. Defaults to min(D, cpu_count); "
                        "on SLURM also caps at $SLURM_CPUS_PER_TASK.")
    p.add_argument("--num-particles", type=int, default=3,
                   help="N: number of input scenarios (particles)")
    p.add_argument("--num-samples", type=int, default=3,
                   help="D: number of parameter samples per MCMC step")
    p.add_argument("--num-mcmc", type=int, default=2,
                   help="L: length of parameter MCMC chain")
    p.add_argument("--case-name", default="storage_q_ug_et_u",
                   choices=["storage_q_ug_et_u_cp","storage_q_gg_et_u", "storage_q_ug_et_u",
                            "storage_q_u_et_u", "storage_q_g_et_u"])
    p.add_argument("--start-date", default="2015-01-01",
                   help="First date of the calibration window. The year of fluxes before it is "
                        "prepended as spinup (no sampled isotopes; precip 18O from the IsoMAP "
                        "monthly climatology, <data-root>/isomap.csv; see prepend_spinup in "
                        "ORPB_cases.py), so the earliest valid start is one year into the data.")
    p.add_argument("--end-date", default="2015-12-31")
    p.add_argument("--resolution", default="daily", choices=["hourly", "daily", "weekly", "biweekly", "monthly"])
    p.add_argument("--coarsen-output", action="store_true",
                   help="Scenario 2: also use the stream isotopes at --resolution (means of the "
                        "grab samples). Default (scenario 1): stream isotopes at native daily.")
    p.add_argument("--data-root", default=os.environ.get(
        "MESAS_DATA_ROOT", "/Users/simon/Desktop/ORPB_resolution_datasets"))
    p.add_argument("--result-root", default=os.environ.get(
        "MESAS_RESULT_ROOT", "/Users/simon/Desktop/ORPB_results"))
    p.add_argument("--run-tag", default="",
                   help="Suffix added to output filenames (e.g. '_1Y', '_3Y_run01')")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip plotting (use on HPC / headless nodes)")
    p.add_argument("--seed", type=int, default=42,
                   help="numpy random seed; recorded in run_config snapshot")
    p.add_argument("--notes", default="",
                   help="Additional notes for the run")
    p.add_argument("--update-theta-dist", action="store_true",
                   help="gibbs sampler only: adapt the proposal to the running estimate "
                        "each iteration. Ignored by pmmh, which adapts its own proposal.")
    p.add_argument("--sampler", default="pmmh", choices=["pmmh", "gibbs"],
                   help="pmmh: particle marginal Metropolis-Hastings (D chains of L steps; "
                        "posterior = post-burn-in draws). gibbs: Xu's particle Gibbs "
                        "(Algorithm 3), kept for comparison. See "
                        "papers/codebase_evaluation_ORPB.md and codebase_evaluation_PMMH.md.")
    p.add_argument("--burnin", type=int, default=None,
                   help="pmmh only: adaptive burn-in length, discarded from the posterior. "
                        "Default L/2.")
    p.add_argument("--traj-thin", type=int, default=10,
                   help="pmmh only: keep the sampled output trajectory every this many steps")
    p.add_argument("--init-from", default=None,
                   help="pmmh only: warm start. Path to a pmmh_*.npz from a cold-started pilot "
                        "of the SAME window, case and resolution; each chain starts from one of "
                        "its post-burn-in states. Omit for a cold start from the priors.")
    return p.parse_args()


# ----------------------------------------------------------------------------
# DATA LOADING (mirrors the original script)
# ----------------------------------------------------------------------------
def load_init_theta(path, theta_names, D, max_logZ_gap=10.0):
    """D warm-start states from a pilot's pmmh_*.npz, one post-burn-in draw per pilot chain in
    turn, so every pilot chain (including any sitting in a second mode) is represented.

    Pilot chains that never reached the posterior are skipped: those whose median log Z_hat is
    more than max_logZ_gap below the median over chains. A genuine second mode fits about as
    well as the main one and is kept; a stuck chain fits far worse."""
    pm = np.load(path)
    names = [str(n) for n in pm["theta_names"]]
    if names != list(theta_names):
        raise ValueError("--init-from run estimated different parameters than this case:\n"
                         f"  file: {names}\n  case: {list(theta_names)}")
    b = int(pm["num_burnin"])
    draws = pm["theta_chains"][:, b + 1:, :]          # [pilot chains, steps, p]
    chain_logZ = np.median(pm["logZ_chains"][:, b + 1:], axis=1)
    keep = np.flatnonzero(chain_logZ >= np.median(chain_logZ) - max_logZ_gap)
    skipped = np.setdiff1d(np.arange(len(chain_logZ)), keep)
    if skipped.size:
        print(f"[info] warm start skips pilot chains {skipped.tolist()} (median log Z "
              f"{np.round(chain_logZ[skipped], 1).tolist()} vs median "
              f"{np.median(chain_logZ):.1f})", flush=True)
    return np.array([draws[keep[d % len(keep)], np.random.randint(draws.shape[1])]
                     for d in range(D)])


def data_filename(args):
    if args.coarsen_output:
        return f"ORPB_isotope_data_precip 18O_{args.resolution}_ORPB 18O_{args.resolution}.csv"
    return f"ORPB_isotope_data_precip 18O_{args.resolution}.csv"


def load_data(args):
    res_map = {"hourly": "h", "daily": "D", "weekly": "W", "biweekly": "2W", "monthly": "ME"}
    if args.resolution not in res_map:
        raise ValueError(f"Unknown resolution: {args.resolution!r}. Expected one of {list(res_map)}.")

    data_df = pd.read_csv(os.path.join(args.data_root, data_filename(args)),
                          index_col=0, parse_dates=[0])

    data_df["precip 18O"] = data_df["mean_c"]

    # calibration window with the preceding year of fluxes prepended: no stream isotopes, and
    # the IsoMAP monthly climatology as precip 18O (excluded from the input fit)
    data_df, n_spinup = prepend_spinup(data_df, args.start_date, args.end_date,  # noqa: F405
                                       load_isoscape(args.data_root))  # noqa: F405
    # held-out validation months never enter calibration (VALIDATION_PERIODS, ORPB_cases.py)
    mask_holdout(data_df, args.resolution if args.coarsen_output else "daily")  # noqa: F405
    return prepare_frame(data_df), n_spinup


def prepare_frame(data_df):
    """Model columns shared by the calibration frame and validate.py's validation frame."""
    data_df["influx (mm/hr)"] = data_df[["rainfall (mm/hr)", "snowmelt (mm/hr)"]].sum(axis=1)
    data_df["quickflow (mm/hr)"] = data_df["discharge (mm/hr)"] - data_df["baseflow 1 (mm/hr)"]
    data_df["bf1_weight"] = data_df["baseflow 1 (mm/hr)"] / data_df["discharge (mm/hr)"]
    data_df["qf_weight"] = data_df["quickflow (mm/hr)"] / data_df["discharge (mm/hr)"]

    df = data_df.copy()
    df["precip 18O"] = df["precip 18O"].bfill().ffill()
    df["is_obs_input"] = df["precip 18O"].notna()
    df["is_obs_output"] = df["ORPB 18O"].notna()
    return df


# ----------------------------------------------------------------------------
# MODEL INITIALIZATION
# ----------------------------------------------------------------------------
def build_model_interface(args, df, n_spinup):
    config = {
        "dt": 1,
        "observed_made_each_step": df["ORPB 18O"].notna().to_list(),
        "influx": ["influx (mm/hr)"],
        "outflux": ["discharge (mm/hr)", "ET (mm/hr)"],
        "use_MAP_AS_weight": True,
        "use_MAP_ref_traj": True,
        "use_MAP_MCMC": True,
        "update_theta_dist": args.update_theta_dist,
        # PMMH needs C_old to be a parameter shared by all particles, or it has no posterior
        "C_old_as_parameter": args.sampler == "pmmh",
    }

    theta_lookup = {
        "storage_q_ug_et_u_cp": theta_storage_q_ug_et_u_cp,  # noqa: F405
        "storage_q_gg_et_u": theta_storage_q_gg_et_u,  # noqa: F405
        "storage_q_ug_et_u": theta_storage_q_ug_et_u,  # noqa: F405
        "storage_q_u_et_u":  theta_storage_q_u_et_u,   # noqa: F405
        "storage_q_g_et_u":  theta_storage_q_g_et_u,   # noqa: F405
    }
    if args.case_name not in theta_lookup:
        raise ValueError(f"Case name not found: {args.case_name}")

    theta_init = theta_lookup[args.case_name]
    # resolve one year of ages at every observation, identically across run lengths
    theta_init["options"]["max_age"] = n_spinup

    model_interface = ModelInterfaceMesas(
        df=df,
        customized_model=SAS_Model,
        num_input_scenarios=args.num_particles,
        config=config,
        theta_init=theta_init,
    )
    return model_interface, config, theta_init


# ----------------------------------------------------------------------------
# OUTPUT
# ----------------------------------------------------------------------------
def save_results(args, model, model_interface):
    os.makedirs(args.result_root, exist_ok=True)
    tag = f"{args.case_name}{args.run_tag}"

    theta_name = model_interface._theta_to_estimate
    pd.DataFrame(model.theta_record, columns=theta_name).to_csv(
        os.path.join(args.result_root, f"theta_{tag}.csv"))
    if args.sampler != "pmmh":
        # Weighted-candidate diagnostics of the particle-Gibbs paths. PMMH has no candidate
        # weights, so these would be all zeros; its spread and convergence are in
        # posterior_summary_*.csv (sd, rhat, accept) and the pmmh_*.npz chains.
        pd.DataFrame(model.theta_std, columns=theta_name).to_csv(
            os.path.join(args.result_root, f"theta_std_{tag}.csv"))
        pd.DataFrame(model.ess_record, columns=theta_name).to_csv(
            os.path.join(args.result_root, f"ess_record_{tag}.csv"))
        pd.DataFrame(model.varll_record, columns=theta_name).to_csv(
            os.path.join(args.result_root, f"varll_record_{tag}.csv"))
    np.savetxt(os.path.join(args.result_root, f"input_scenarios_{tag}.csv"),
               model.input_record, delimiter=",")
    np.savetxt(os.path.join(args.result_root, f"output_scenarios_{tag}.csv"),
               model.output_record, delimiter=",")
    np.savetxt(os.path.join(args.result_root, f"state_record_{tag}.csv"),
               model.state_record, delimiter=",")
    np.savetxt(os.path.join(args.result_root, f"pQ_mle_{tag}.csv"),
               model.pQ_mle['discharge (mm/hr)'], delimiter=",")
    np.savetxt(os.path.join(args.result_root, f"sT_mle_{tag}.csv"),
               model.sT_mle, delimiter=",")
    np.savetxt(os.path.join(args.result_root, f"ST_mle_{tag}.csv"),
               model.ST_mle, delimiter=",")

    if args.sampler == "pmmh":
        # Everything needed for the posterior: all chains, all steps (burn-in included, its
        # length stored alongside), and the output trajectories sampled with each state.
        # For pmmh, pQ/sT/ST_mle above are at the pooled posterior MEDIAN, and theta_*.csv /
        # output_scenarios_*.csv hold chain 0 only (kept for the old plotting code).
        np.savez_compressed(
            os.path.join(args.result_root, f"pmmh_{tag}.npz"),
            theta_chains=model.theta_chains, logZ_chains=model.logZ_chains,
            accept_chains=model.accept_chains, output_chains=model.output_chains,
            state_chains=model.state_chains, input_chains=model.input_chains,
            traj_iters=model.traj_iters, num_burnin=model.num_burnin,
            prop_scale=model.prop_scale_record,
            theta_names=np.array(model_interface._theta_to_estimate),
            observed_ind=model_interface.observed_ind,
            dates=model_interface.df.index.astype(str).to_numpy())
        summary = model.pmmh_summary()
        summary.to_csv(os.path.join(args.result_root, f"posterior_summary_{tag}.csv"))
        print(summary.to_string(), flush=True)

    try:
        import dill
        session_path = os.path.join(args.result_root, f"session_{tag}.pkl")
        with open(session_path, "wb") as f:
            dill.dump({"model": model, "model_interface": model_interface,
                       "args": vars(args)}, f)
    except Exception as e:
        print(f"[warn] could not pickle session: {e}")


def make_plots(args, model, model_interface):
    if args.no_plot:
        return
    os.makedirs(args.result_root, exist_ok=True)
    tag = f"{args.case_name}{args.run_tag}"
    df = model_interface.df

    st = model_interface.observed_ind[0]
    et = model_interface.observed_ind[-1]
    time = df.index
    if args.sampler == "pmmh":
        # one figure: the trajectories sampled with every post-burn-in state, all chains
        keep = model.traj_iters > model.num_burnin
        trajs = model.output_chains[:, keep, st:et + 1].reshape(-1, et + 1 - st)
        lo, mid, hi = np.percentile(trajs, [5, 50, 95], axis=0)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.fill_between(time[st:et + 1], lo, hi, color="orange", alpha=0.3,
                        label="posterior 5-95%")
        ax.plot(time[st:et + 1], mid, color="orange", label="posterior median")
        obs = df["ORPB 18O"].iloc[st:et + 1]
        ax.plot(time[st:et + 1][obs.notna()], obs.dropna(), "k.", label="observed")
        ax.set_xlabel("Date")
        ax.set_ylabel("delta 18O in stream (per mil)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(args.result_root, f"output_{tag}_posterior.pdf"))
        plt.close(fig)
        return
    for i in range(args.num_mcmc + 1):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.step(time[st:et], df["ORPB 18O"].backfill().iloc[st:et], label="observed")
        ax.plot(time[st:et], model.output_record[i, st:et].T,
                color="orange", alpha=0.4, label="predicted")
        ax.set_title(f"pMCMC iteration {i}")
        ax.set_xlabel("Date")
        ax.set_ylabel("delta 18O in stream (per mil)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(args.result_root, f"output_{tag}_iter{i}.pdf"))
        plt.close(fig)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    args = parse_args()

    # Reproducibility: set seed BEFORE any sampling (model_interface init draws thetas).
    set_run_seed(args.seed)

    # On HPC nodes there is no display; force a non-interactive backend.
    if args.no_plot or not sys.stdout.isatty():
        matplotlib.use("Agg", force=True)

    # On SLURM, honour the CPU allocation rather than the node's full cpu_count.
    if args.num_cores is None and "SLURM_CPUS_PER_TASK" in os.environ:
        try:
            args.num_cores = int(os.environ["SLURM_CPUS_PER_TASK"])
        except ValueError:
            pass

    print(f"[info] parallel={args.parallel}  num_cores={args.num_cores}  "
          f"N={args.num_particles} D={args.num_samples} L={args.num_mcmc}  "
          f"case={args.case_name}  range={args.start_date}..{args.end_date}  "
          f"seed={args.seed}",
          flush=True)

    df, n_spinup = load_data(args)
    model_interface, config, theta_init = build_model_interface(args, df, n_spinup)
    print(f"[info] isoscape spinup={df.index[0].date()}..{df.index[n_spinup - 1].date()} "
          f"({n_spinup} steps, = max_age) calibration window={df.index[n_spinup].date()}..{df.index[-1].date()}", flush=True)

    model = SSModel(
        model_interface=model_interface,
        num_parameter_samples=args.num_samples,
        len_parameter_MCMC=args.num_mcmc,
    )

    if args.sampler == "pmmh":
        init_theta = None
        if args.init_from:
            init_theta = load_init_theta(args.init_from, model_interface._theta_to_estimate,
                                         args.num_samples)
            print(f"[info] warm start from {args.init_from}", flush=True)
        model.run_pmmh(num_burnin=args.burnin, num_cores=args.num_cores,
                       parallel=args.parallel, traj_thin=args.traj_thin,
                       init_theta=init_theta)
    elif args.parallel:
        model.run_particle_Gibbs_parallel(num_cores=args.num_cores)
    else:
        model.run_particle_Gibbs()

    save_results(args, model, model_interface)
    make_plots(args, model, model_interface)

    # Snapshot the run config alongside results (for reproducibility / auditing).
    tag = f"{args.case_name}{args.run_tag}"
    res_map = {"hourly": "h", "daily": "D", "weekly": "W", "biweekly": "2W", "monthly": "ME"}
    if args.resolution not in res_map:
        raise ValueError(f"Unknown resolution: {args.resolution!r}. Expected one of {list(res_map)}.")
    res_code = res_map[args.resolution]
    extra = {
        "sampler": args.sampler,
        "burnin": getattr(model, "num_burnin", None),
        "init_from": args.init_from,
        "parallel": args.parallel,
        "num_cores": args.num_cores,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "hostname": os.environ.get("HOSTNAME") or os.environ.get("COMPUTERNAME"),
        "spinup": "isoscape",  # preceding year of fluxes, precip 18O = IsoMAP monthly climatology
        "isoscape_col": ISOSCAPE_COL,  # noqa: F405
        "coarsen_output": args.coarsen_output,  # scenario 2 if True
        "holdout_months": HOLDOUT_MONTHS,  # noqa: F405  stream isotopes masked for validation
        "spinup_steps": n_spinup,
        "spinup_start": str(df.index[0]),
        "max_age": theta_init["options"]["max_age"],  # mesas options are not otherwise in the snapshot
    }
    save_run_config(
        out_path=os.path.join(args.result_root, f"run_config_{tag}.json"),
        case_name=args.case_name,
        config=config,
        N=args.num_particles,
        D=args.num_samples,
        L=args.num_mcmc,
        date_start=df.index[n_spinup],  # calibration window; spinup recorded in extra
        date_end=df.index[-1],
        resolution=res_code,
        data_source=data_filename(args),
        theta_to_estimate=model_interface._theta_to_estimate,
        theta_init=theta_init,
        seed=args.seed,
        extra=extra,
        notes=args.notes
    )

    print("[info] done", flush=True)


if __name__ == "__main__":
    # The __main__ guard is REQUIRED for multiprocessing.Pool on Windows and
    # under the 'spawn' start method; do not remove.
    main()
