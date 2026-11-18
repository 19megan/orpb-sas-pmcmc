# Per-run table for the Ch1 evaluation: parameter identifiability (KLD) and convergence, one row
# per finished PMMH run, to be joined to validate.py's validation_scores.csv by tag.
#
# What it does, per run (pmmh_<tag>.npz with run_config_<tag>.json alongside):
#   - post-burn-in draws pooled over chains, leaving out the run_config's "excluded_chains"
#     (stuck_check.py), the same draws validate.py scores
#   - KLD(posterior || prior) in nats for each physical parameter, against the prior saved in the
#     run_config, with the same bin-free estimator as ORPB_pmcmc_script.py (kld_reflect)
#   - "<param> KLD split": largest difference in KLD between two random halves of the chains over
#     SPLITS splits. It is the Monte Carlo uncertainty of the KLD, large where chains disagree
#     (e.g. a parameter with a second mode visited by few chains); don't read KLD differences
#     between runs smaller than it
#   - rank-normalized split R-hat and bulk ESS (Vehtari et al. 2021) for every parameter, plus
#     N, D, L, burn-in and acceptance, for the convergence table
#
# Usage:
#   python compile_runs.py --runs <folder>/pmmh_*.npz --out <folder>/run_summary.csv
#   Pass only the final converged run of each window, not the warm starts before it.
#
# Created on: 2026-11-16
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm, rankdata

RES = {"h": "hourly", "D": "daily", "W": "weekly", "2W": "biweekly", "ME": "monthly"}
LABELS = {"qf_weight#@#scale": "qf scale", "bf1_weight#@#a": "bf1 a", "bf1_weight#@#M": "bf1 M",
          "ET (mm/hr) SAS function#@#scale": "ET scale", "C_old": "C_old",
          "sigma C in": "sigma C in", "sigma C out": "sigma C out"}
PHYSICAL = ["qf scale", "bf1 a", "bf1 M", "ET scale", "C_old"]   # KLD for these; sigmas are noise
SPLITS = 10


def label(name):
    for key, lab in LABELS.items():
        if name.endswith(key):
            return lab
    raise ValueError(f"no label for parameter {name}")


def prior_of(name, cfg):
    """(lo, hi, 'log'|'lin') of a parameter, from the priors saved in the run_config."""
    parts = name.split("#@#")
    if name in cfg["priors"]:                                     # sigma C in / out
        p = {"prior_dis": cfg["priors"][name]["distribution"], "prior_params": cfg["priors"][name]["params"]}
    elif parts[-1] == "C_old":
        p = cfg["case"]["solute_parameters"][parts[0]]["prior"]
    else:
        flux, sas, key = parts
        spec = cfg["case"]["sas_specs"][flux][sas]
        p = spec["priors"][key] if "priors" in spec else spec["prior"]
    if p["prior_dis"] not in ("uniform", "loguniform"):
        raise ValueError(f"{name}: KLD here assumes a (log-)uniform prior, got {p['prior_dis']}")
    lo, hi = p["prior_params"]
    return lo, hi, "log" if p["prior_dis"] == "loguniform" else "lin"


# Copied from ORPB_pmcmc_script.py (an interactive script, so not importable); keep them the same.
# In u = position within the prior range in the prior's flat coordinate, the prior is Uniform(0, 1),
# so KLD = -H(posterior of u), estimated with a Gaussian KDE mirrored across 0 and 1.
def kld_reflect(x, lo, hi, shape, n_fit=20000, n_eval=5000, seed=0):
    f = np.log if shape == 'log' else (lambda v: v)
    u = (f(np.asarray(x, float)) - f(lo)) / (f(hi) - f(lo))
    rng = np.random.default_rng(seed)
    kde = gaussian_kde(rng.choice(u, size=min(n_fit, u.size), replace=False))
    ev = rng.choice(u, size=min(n_eval, u.size), replace=False)
    return float(np.mean(np.log(kde(ev) + kde(-ev) + kde(2.0 - ev))))


# Same as rhat_rank.py
def _split(x):
    half = x.shape[1] // 2
    return np.concatenate([x[:, :half], x[:, half:2 * half]], axis=0)


def _rhat(x):
    s = _split(x)
    h = s.shape[1]
    W = s.var(axis=1, ddof=1).mean()
    B = h * s.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((h - 1) / h * W + B / h) / W))


def _z_rank(x):
    r = rankdata(x.ravel(), method="average").reshape(x.shape)
    return norm.ppf((r - 0.375) / (x.size + 0.25))


def _bulk_ess(x):
    s = _split(_z_rank(x))
    m, h = s.shape
    f = np.fft.rfft(s - s.mean(axis=1, keepdims=True), n=2 * h, axis=1)
    ac = np.fft.irfft(f * np.conj(f), axis=1)[:, :h] / (h - np.arange(h))
    W = (ac[:, 0] * h / (h - 1)).mean()
    var_plus = (h - 1) / h * W + s.mean(axis=1).var(ddof=1)
    rho = 1 - (W - (ac * (h / (h - 1))).mean(axis=0)) / var_plus
    tau, t = 1.0, 1
    while t + 1 < h and rho[t] + rho[t + 1] > 0:
        tau += 2 * (rho[t] + rho[t + 1])
        t += 2
    return float(m * h / tau)


def summarize(npz_path):
    tag = os.path.basename(npz_path)[len("pmmh_"):-len(".npz")]
    with open(os.path.join(os.path.dirname(os.path.abspath(npz_path)), f"run_config_{tag}.json")) as f:
        cfg = json.load(f)
    extra = cfg["metadata"]["extra"]
    # m = re.search(r"_(h|D|W|2W|ME)(\d+[MY])(?:_(cold|warm))?_job", tag)
    # window labels as in sample_counts.py and the cost workbook's Evaluation Table (6M_cold, 3M_winter, 2Y)
    m = re.search(r"_(h|D|W|2W|ME)(\d+[MY](?:_[a-z]+)?)_job", tag)
    if m is None:
        raise ValueError(f"{tag}: can't read resolution and length from the tag")
    # window = m.group(2) + (f" {m.group(3)}" if m.group(3) else "")
    window = m.group(2) # Daily

    pm = np.load(npz_path)
    names = [str(n) for n in pm["theta_names"]]
    b = int(pm["num_burnin"])
    excluded = extra.get("excluded_chains", [])
    post = np.delete(pm["theta_chains"], excluded, axis=0)[:, b + 1:]       # [D, n, p]
    acc = np.delete(pm["accept_chains"], excluded, axis=0)
    D = post.shape[0]

    row = {"tag": tag,
           # "Scenario": "Scenario 2: stream coarsened" if extra.get("coarsen_output", False)
           "Scenario": "Scenario 2: stream = precip resolution" if extra.get("coarsen_output", False)
           else "Scenario 1: stream daily, precip varies",
           "Resolution": RES[m.group(1)], "Window": window,
           "date_start": cfg["data"]["date_start"][:10], "date_end": cfg["data"]["date_end"][:10],
           "N": int(cfg["mcmc"]["N_particles"]), "D": D, "L": int(cfg["mcmc"]["L_iterations"]),
           "burnin": b, "init_from": os.path.basename(str(extra.get("init_from") or "")),
           "excluded_chains": " ".join(map(str, excluded)),
           "accept": float(np.asarray(acc, float)[:, b + 1:].mean()) if np.ndim(acc) == 2
           else float(np.mean(acc))}
    rng = np.random.default_rng(0)
    splits = [rng.permutation(D) for _ in range(SPLITS)]
    for j, name in enumerate(names):
        lab, x = label(name), post[:, :, j]
        row[f"{lab} rhat"] = _rhat(_z_rank(x))
        row[f"{lab} ess"] = _bulk_ess(x)
        if lab in PHYSICAL:
            lo, hi, shape = prior_of(name, cfg)
            row[f"{lab} KLD"] = kld_reflect(x.ravel(), lo, hi, shape)
            # smaller KDE for the halves (0.7 s vs 8 s per call; its own noise is about 0.01 nats)
            row[f"{lab} KLD split"] = max(
                abs(kld_reflect(x[o[:D // 2]].ravel(), lo, hi, shape, n_fit=5000, n_eval=2000)
                    - kld_reflect(x[o[D // 2:]].ravel(), lo, hi, shape, n_fit=5000, n_eval=2000))
                for o in splits)
    print(f"[done] {tag}" + (f" (left out chains {excluded})" if excluded else ""), flush=True)
    return row


def main():
    p = argparse.ArgumentParser(description="KLD and convergence per finished PMMH run")
    p.add_argument("--runs", nargs="+", required=True,
                   help="final pmmh_<tag>.npz files (globs allowed), each with run_config_<tag>.json")
    p.add_argument("--out", required=True, help="CSV, one row per run")
    p.add_argument("--num-cores", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
                   help="runs summarized in parallel, one per process")
    args = p.parse_args()
    paths = sorted({f for pattern in args.runs for f in glob.glob(pattern)})
    if not paths:
        raise SystemExit(f"No files match {args.runs}")
    num_cores = max(1, min(args.num_cores, len(paths)))
    if num_cores == 1:
        rows = [summarize(path) for path in paths]
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(num_cores) as pool:
            rows = pool.map(summarize, paths)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"[info] wrote {args.out} ({len(df)} runs)", flush=True)


if __name__ == "__main__":
    main()
