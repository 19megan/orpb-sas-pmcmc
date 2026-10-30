# Out-of-sample validation of finished PMMH runs on one common set of held-out observations.
#
# Every run, whatever its window, resolution or scenario, is scored on the same stream
# observations (VALIDATION_PERIODS in ORPB_cases.py): 2014 Jul-Dec, which lies in no
# calibration window, and the HOLDOUT_MONTHS that load_data masks out of every calibration
# dataset. So differences in score come only from the parameters each sampling design learned.
#
# What it does, per run:
#   - takes post-burn-in posterior draws pooled over chains (as pmmh_summary does)
#   - runs each draw forward over 2014-2018 on ONE common validation frame: the daily
#     precipitation input and native daily stream isotopes, with 2014 looped once before it as
#     spinup (fluxes AND isotopes -- this is validation forcing, identical for every run) so
#     every scored observation sees a full year of resolved input, like calibration
#   - uses the calibrated model exactly: the same convolution as transition_model, the same
#     max_age (one year), each draw's own C_old for older water. The observed input is used
#     as is (no sigma C in noise): input uncertainty is resolution-specific, so leaving it out
#     keeps every run on the same footing. Predictive draws add N(0, sigma C out).
#
# Scores per validation period, and their mean ("mean of periods", each period weighted equally):
#   rmse, bias  -- of the pointwise posterior-median prediction; comparable across all runs
#   crps, cover90 -- of the predictive distribution. These include sigma C out, which in
#                 scenario 2 is the error of a coarse MEAN and so too small for daily data:
#                 compare them only between runs with the same output resolution.
#
# Usage:
#   python validate.py --runs <result_root>/pmmh_*.npz --out <result_root>/validation_scores.csv
#   (each pmmh_<tag>.npz needs its run_config_<tag>.json alongside)
#    
#   Also can use slurm_ORPB_validate.sh to run on a cluster, with --runs and --out passed through.
#
# Created on: 2026-10-28
import argparse
import glob
import json
import os
import sys
from types import SimpleNamespace

script_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_folder)
import numpy as np   # noqa: E402
import pandas as pd   # noqa: E402

import ORPB_pmcmc_script_parallel as runner   # noqa: E402
from ORPB_cases import VALIDATION_PERIODS, HOLDOUT_MONTHS   # noqa: E402

FLUX, SOL, OBS = "discharge (mm/hr)", "precip 18O", "ORPB 18O"
LOOP_YEAR, FRAME_END = "2014", "2018-12-31"   # 2019 has almost no stream isotopes
MAX_AGE = 365   # one year of daily steps, the calibration max_age


def build_validation_interface(case_name, data_root):
    """The common validation frame and a model interface on it for one SAS case."""
    raw = pd.read_csv(os.path.join(data_root, "ORPB_isotope_data_precip 18O_daily.csv"),
                      index_col=0, parse_dates=[0])
    raw["precip 18O"] = raw["mean_c"]
    loop = raw.loc[LOOP_YEAR].copy()
    loop.index = loop.index - pd.DateOffset(years=1)
    loop[OBS] = np.nan   # spinup is never scored
    df = runner.prepare_frame(pd.concat([loop, raw.loc[:FRAME_END]]))
    args = SimpleNamespace(case_name=case_name, num_particles=1, update_theta_dist=False,
                           sampler="pmmh")
    mi, _, _ = runner.build_model_interface(args, df, MAX_AGE)
    return mi, df


def predict(mi, cin):
    """C_Q for the interface's current theta and a deterministic input, as transition_model."""
    W = mi._conv_weights[FLUX][SOL]   # [max_age, T]
    max_age = W.shape[0]
    # row t holds cin[t], cin[t-1], ..., cin[t-max_age+1]; the zero padding has zero weight
    C_J = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([np.zeros(max_age - 1), cin]), max_age)[:, ::-1]
    return np.einsum("ta,at->t", C_J, W) + mi.theta.initial_state[0] * (1.0 - mi._P_resolved[FLUX])


def scores(ens, point, y):
    """ens [S, n] predictive draws, point [n] median prediction, y [n] observations."""
    S = ens.shape[0]
    srt = np.sort(ens, axis=0)
    # ensemble CRPS: E|X - y| - E|X - X'| / 2, the second term from the sorted draws
    crps = (np.abs(ens - y).mean(axis=0)
            - ((2 * np.arange(1, S + 1) - S - 1)[:, None] * srt).sum(axis=0) / S ** 2)
    lo, hi = np.percentile(ens, [5, 95], axis=0)
    return {"n_obs": y.size,
            "rmse": float(np.sqrt(np.mean((point - y) ** 2))),
            "bias": float(np.mean(point - y)),
            "crps": float(crps.mean()),
            "cover90": float(np.mean((y >= lo) & (y <= hi)))}


_INTERFACES = {}   # one validation interface per SAS case, per process


def validate_run(npz_path, data_root, num_draws, seed):
    # same seed for every run: common draws of the predictive noise, so score differences
    # between runs are not partly sampling noise
    rng = np.random.default_rng(seed)
    interfaces = _INTERFACES
    tag = os.path.basename(npz_path)[len("pmmh_"):-len(".npz")]
    with open(os.path.join(os.path.dirname(npz_path), f"run_config_{tag}.json")) as f:
        cfg = json.load(f)
    extra = cfg["metadata"]["extra"]
    if extra.get("holdout_months") != HOLDOUT_MONTHS or extra.get("spinup") != "isoscape":
        print(f"[skip] {tag}: calibrated before the held-out months were masked (or with "
              f"another spinup), so its validation scores would be partly in-sample", flush=True)
        return []

    case = cfg["case"]["name"]
    if case not in interfaces:
        interfaces[case] = build_validation_interface(case, data_root)
    mi, df = interfaces[case]

    pm = np.load(npz_path)
    names = [str(n) for n in pm["theta_names"]]
    if names != list(mi._theta_to_estimate):
        raise ValueError(f"{tag}: parameters {names} do not match case {case}")
    flat = pm["theta_chains"][:, int(pm["num_burnin"]) + 1:, :].reshape(-1, len(names))
    draws = flat[rng.choice(len(flat), size=min(num_draws, len(flat)), replace=False)]

    cin = df[SOL].to_numpy()
    preds, sigmas = [], []
    for theta in draws:
        mi.update_theta(theta)
        preds.append(predict(mi, cin))
        sigmas.append(mi.theta.observation_model[0])
    preds = np.array(preds)                                    # [S, T]
    ens = preds + rng.normal(size=preds.shape) * np.array(sigmas)[:, None]
    point = np.median(preds, axis=0)

    y_all = df[OBS].to_numpy()
    rows = []
    for period, (start, end) in VALIDATION_PERIODS.items():
        # label slicing is inclusive and reads '2015-04' as the whole month
        sel = df.index.isin(df.loc[start:end].index) & ~np.isnan(y_all)
        rows.append({"period": period, **scores(ens[:, sel], point[sel], y_all[sel])})
    # each period (season / flow condition) counts equally; pooling the observations would
    # let 2014 Jul-Dec, about two thirds of them, dominate
    summary = pd.DataFrame(rows).drop(columns="period").mean()
    summary["n_obs"] = sum(r["n_obs"] for r in rows)
    rows.append({"period": "mean of periods", **summary.to_dict()})

    meta = {"tag": tag, "case": case, "source": cfg["data"]["source"],
            "coarsen_output": extra.get("coarsen_output", False),
            "date_start": cfg["data"]["date_start"], "date_end": cfg["data"]["date_end"],
            "num_draws": len(draws)}
    print(f"[done] {tag}: mean-of-periods rmse {rows[-1]['rmse']:.3f}, crps {rows[-1]['crps']:.3f}",
          flush=True)
    return [{**meta, **r} for r in rows]


def main():
    p = argparse.ArgumentParser(description="Score finished PMMH runs on the held-out data")
    p.add_argument("--runs", nargs="+", required=True,
                   help="pmmh_<tag>.npz files (globs allowed), each with run_config_<tag>.json")
    p.add_argument("--data-root", default=os.environ.get(
        "MESAS_DATA_ROOT", "/Users/simon/Desktop/ORPB_resolution_datasets"))
    p.add_argument("--out", required=True, help="CSV of scores, one row per run and period")
    p.add_argument("--num-draws", type=int, default=200,
                   help="posterior draws per run (each is one mesas solve)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-cores", type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
                   help="runs validated in parallel, one per process (default: SLURM allocation)")
    args = p.parse_args()

    paths = sorted({f for pattern in args.runs for f in glob.glob(pattern)})
    if not paths:
        raise SystemExit(f"No files match {args.runs}")
    jobs = [(path, args.data_root, args.num_draws, args.seed) for path in paths]
    num_cores = max(1, min(args.num_cores, len(paths)))
    print(f"[info] {len(paths)} runs on {num_cores} cores", flush=True)
    if num_cores == 1:
        results = [validate_run(*job) for job in jobs]
    else:
        import multiprocessing as mp
        with mp.Pool(num_cores) as pool:
            results = pool.starmap(validate_run, jobs)
    rows = [row for result in results for row in result]
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[info] wrote {args.out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
