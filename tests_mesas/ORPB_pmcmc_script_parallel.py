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
from model.ssm_model import SSModel
from mesas.sas.model import Model as SAS_Model

from ORPB_cases import *  # noqa: F401,F403  (theta_init constants)
from functions.run_config import save_run_config, set_run_seed


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
                   choices=["storage_q_gg_et_u", "storage_q_ug_et_u",
                            "storage_q_u_et_u", "storage_q_g_et_u"])
    p.add_argument("--start-date", default="2014-01-01")
    p.add_argument("--end-date", default="2014-12-31")
    p.add_argument("--resolution", default="hourly", choices=["hourly", "daily"])
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
    return p.parse_args()


# ----------------------------------------------------------------------------
# DATA LOADING (mirrors the original script)
# ----------------------------------------------------------------------------
def load_data(args):
    res_map = {"hourly": "h", "daily": "D", "weekly": "W", "biweekly": "2W", "monthly": "M"}
    if args.resolution not in res_map:
        raise ValueError(f"Unknown resolution: {args.resolution!r}. Expected one of {list(res_map)}.")
    res = res_map[args.resolution]

    fname = f"ORPB_isotope_data_isoMAP_precip 18O_{args.resolution}.csv"
    data_df = pd.read_csv(os.path.join(args.data_root, fname),
                          index_col=0, parse_dates=[0])

    data_df["precip 18O"] = data_df["mean_c"]
    for col in ("discharge", "baseflow 1", "snowmelt", "rainfall", "ET"):
        data_df[f"{col} (mm/hr)"] = data_df[f"{col} (mm/{res})"]

    data_df = data_df.loc[pd.Timestamp(args.start_date): pd.Timestamp(args.end_date)]

    data_df["influx (mm/hr)"] = data_df[["rainfall (mm/hr)", "snowmelt (mm/hr)"]].sum(axis=1)
    data_df["quickflow (mm/hr)"] = data_df["discharge (mm/hr)"] - data_df["baseflow 1 (mm/hr)"]
    data_df["bf1_weight"] = data_df["baseflow 1 (mm/hr)"] / data_df["discharge (mm/hr)"]
    data_df["qf_weight"] = data_df["quickflow (mm/hr)"] / data_df["discharge (mm/hr)"]

    data_df["is_obs_input"] = data_df["precip 18O"].notna()
    data_df["is_obs_input_filled"] = data_df["precip 18O"].isna()

    df = data_df.copy()
    df["precip 18O"] = df["precip 18O"].ffill().bfill()
    df["is_obs_output"] = df["ORPB 18O"].notna()
    return df


# ----------------------------------------------------------------------------
# MODEL INITIALIZATION
# ----------------------------------------------------------------------------
def build_model_interface(args, df):
    config = {
        "dt": 1,
        "observed_made_each_step": df["ORPB 18O"].notna().to_list(),
        "influx": ["influx (mm/hr)"],
        "outflux": ["discharge (mm/hr)", "ET (mm/hr)"],
        "use_MAP_AS_weight": True,
        "use_MAP_ref_traj": True,
        "use_MAP_MCMC": True,
        "update_theta_dist": True,
    }

    theta_lookup = {
        "storage_q_gg_et_u": theta_storage_q_gg_et_u,  # noqa: F405
        "storage_q_ug_et_u": theta_storage_q_ug_et_u,  # noqa: F405
        "storage_q_u_et_u":  theta_storage_q_u_et_u,   # noqa: F405
        "storage_q_g_et_u":  theta_storage_q_g_et_u,   # noqa: F405
    }
    if args.case_name not in theta_lookup:
        raise ValueError(f"Case name not found: {args.case_name}")

    theta_init = theta_lookup[args.case_name]

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
    pd.DataFrame(model.theta_std, columns=theta_name).to_csv(
        os.path.join(args.result_root, f"theta_std_{tag}.csv"))
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

    df = load_data(args)
    model_interface, config, theta_init = build_model_interface(args, df)

    model = SSModel(
        model_interface=model_interface,
        num_parameter_samples=args.num_samples,
        len_parameter_MCMC=args.num_mcmc,
    )

    if args.parallel:
        model.run_particle_Gibbs_parallel(num_cores=args.num_cores)
    else:
        model.run_particle_Gibbs()

    save_results(args, model, model_interface)
    make_plots(args, model, model_interface)

    # Snapshot the run config alongside results (for reproducibility / auditing).
    tag = f"{args.case_name}{args.run_tag}"
    res_code = "h" if args.resolution == "hourly" else "D"
    extra = {
        "parallel": args.parallel,
        "num_cores": args.num_cores,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "hostname": os.environ.get("HOSTNAME") or os.environ.get("COMPUTERNAME"),
    }
    save_run_config(
        out_path=os.path.join(args.result_root, f"run_config_{tag}.json"),
        case_name=args.case_name,
        config=config,
        N=args.num_particles,
        D=args.num_samples,
        L=args.num_mcmc,
        date_start=df.index[0],
        date_end=df.index[-1],
        resolution=res_code,
        data_source=f"ORPB_isotope_data_isoMAP_precip 18O_{args.resolution}.csv",
        theta_to_estimate=model_interface._theta_to_estimate,
        theta_init=theta_init,
        seed=args.seed,
        extra=extra,
    )

    print("[info] done", flush=True)


if __name__ == "__main__":
    # The __main__ guard is REQUIRED for multiprocessing.Pool on Windows and
    # under the 'spawn' start method; do not remove.
    main()
