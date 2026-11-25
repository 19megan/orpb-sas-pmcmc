# Usage: python predictive_by_chain.py <pmmh npz> [--data-root DIR] [--per-chain 20] [--num-cores K]
#        [--drop C ...] [--out predictive_rhat.csv]
# Do the chains agree on what validate.py scores, even where they disagree on parameters?
# Same frame, prediction and scores as validate.py (imported, not copied). No run_config needed.
# Chains are solved in parallel, one per process (default: the Slurm allocation), with a line per
# chain. --out adds this run's predictive R-hats as one row (keyed by tag; a rerun replaces the
# row) to a CSV that evaluate_ch1.py reads with --predictive-rhat.
# --tests-dir is only needed if this script is not next to validate.py (validate.py itself finds
# tests_mesas).
import argparse, os, sys, time
import numpy as np

_mi = {}


def setup(tests_dir):
    if tests_dir:
        sys.path.insert(0, tests_dir)


def solve_chain(job):
    tests_dir, case, data_root, thetas = job
    setup(tests_dir)
    import validate as V
    if "mi" not in _mi:
        _mi["mi"], _mi["df"] = V.build_validation_interface(case, data_root)
    mi, df = _mi["mi"], _mi["df"]
    cin = df[V.SOL].to_numpy()
    preds, sig = [], []
    for th in thetas:
        mi.update_theta(th)
        preds.append(V.predict(mi, cin))
        sig.append(mi.theta.observation_model[0])
    return np.array(preds), np.array(sig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("--data-root", default=os.environ.get("MESAS_DATA_ROOT"))
    p.add_argument("--tests-dir", default=None)
    p.add_argument("--case", default="storage_q_ug_et_u_cp")
    p.add_argument("--per-chain", type=int, default=20)
    p.add_argument("--num-cores", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    p.add_argument("--drop", type=int, nargs="*", default=[],
                   help="chains to leave out, e.g. one stuck on a filter overestimate")
    p.add_argument("--out", default=None,
                   help="CSV to add this run's predictive R-hats to (one row per run tag)")
    args = p.parse_args()
    setup(args.tests_dir)
    import pandas as pd
    import validate as V
    from ORPB_cases import VALIDATION_PERIODS

    tag = os.path.basename(args.npz)[len("pmmh_"):-len(".npz")]
    print(f"[info] run {tag}", flush=True)
    pm = np.load(args.npz)
    b = int(pm["num_burnin"])
    post = pm["theta_chains"][:, b + 1:]
    if args.drop:
        print(f"[info] leaving out chains {args.drop}; chain numbers below are renumbered 0..", flush=True)
        post = np.delete(post, args.drop, axis=0)
    D, n, _ = post.shape
    steps = np.linspace(0, n - 1, args.per_chain).astype(int)

    mi, df = V.build_validation_interface(args.case, args.data_root)
    if [str(x) for x in pm["theta_names"]] != list(mi._theta_to_estimate):
        raise ValueError("parameters do not match the case")
    y_all = df[V.OBS].to_numpy()
    jobs = [(args.tests_dir, args.case, args.data_root, post[d, steps]) for d in range(D)]
    print(f"[info] {D} chains x {len(steps)} draws on {args.num_cores} cores", flush=True)
    t0 = time.time()
    out = [None] * D
    if args.num_cores <= 1:
        for d, job in enumerate(jobs):
            out[d] = solve_chain(job)
            print(f"[chain {d}] done, {time.time() - t0:.0f} s", flush=True)
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.num_cores) as pool:
            for d, res in enumerate(pool.imap(solve_chain, jobs)):
                out[d] = res
                print(f"[chain {d}] done, {time.time() - t0:.0f} s", flush=True)
    preds = np.array([o[0] for o in out])
    sig = np.array([o[1] for o in out])

    sels = {k: df.index.isin(df.loc[a:c].index) & ~np.isnan(y_all) for k, (a, c) in VALIDATION_PERIODS.items()}
    ens_all = preds + np.random.default_rng(0).normal(size=preds.shape) * sig[:, :, None]

    def score(chains):
        # validate.py's mean of periods, for draws pooled over these chains
        P = preds[chains].reshape(-1, preds.shape[-1])
        E = ens_all[chains].reshape(-1, preds.shape[-1])
        point = np.median(P, axis=0)
        rows = pd.DataFrame([V.scores(E[:, s], point[s], y_all[s]) for s in sels.values()])
        return rows[["rmse", "bias", "crps", "cover90"]].mean()

    def rhat(x):
        h = x.shape[1] // 2
        s = np.concatenate([x[:, :h], x[:, h:2 * h]], axis=0)
        W = s.var(axis=1, ddof=1).mean(axis=0)
        B = h * s.mean(axis=1).var(axis=0, ddof=1)
        return np.sqrt(((h - 1) / h * W + B / h) / W)

    from scipy.stats import norm, rankdata

    def rank_rhat(x):
        # rank-normalized split R-hat (Vehtari et al. 2021), the larger of bulk and folded, as in
        # SSModel.pmmh_summary and compile_runs.py; x is [chains, draws] or [chains, draws, dates],
        # ranks pooled over chains and draws separately for each date
        flat = x.reshape(x.shape[0] * x.shape[1], -1)

        def z(v):
            return norm.ppf((rankdata(v, axis=0) - 0.375) / (v.shape[0] + 0.25)).reshape(x.shape)

        return np.maximum(rhat(z(flat)), rhat(z(np.abs(flat - np.median(flat, axis=0)))))

    # one RMSE per draw, each period counted equally
    draw_rmse = np.mean([np.sqrt(((preds[:, :, s] - y_all[s]) ** 2).mean(axis=2)) for s in sels.values()], axis=0)
    scored = np.any(list(sels.values()), axis=0)
    # r_pt = rhat(preds[:, :, scored])
    # r_rmse = float(rhat(draw_rmse))
    r_pt = rank_rhat(preds[:, :, scored])
    r_rmse = float(rank_rhat(draw_rmse))
    r_pt_classic = rhat(preds[:, :, scored])
    r_rmse_classic = float(rhat(draw_rmse))
    print(f"\n{D} chains, {len(steps)} draws per chain, {scored.sum()} scored observations")
    print(f"Rank R-hat of the prediction at each scored date: median {np.median(r_pt):.2f}, 95th pct {np.percentile(r_pt, 95):.2f}, max {r_pt.max():.2f}")
    print(f"Rank R-hat of per-draw RMSE: {r_rmse:.2f}")
    print(f"(classic: median {np.median(r_pt_classic):.2f}, 95th pct {np.percentile(r_pt_classic, 95):.2f}, max {r_pt_classic.max():.2f}, RMSE {r_rmse_classic:.2f})")

    if args.out:
        row = pd.DataFrame([{"tag": tag, "pred chains": D, "pred draws per chain": len(steps),
                             "pred dropped chains": " ".join(map(str, args.drop)),
                             "pred rhat median": float(np.median(r_pt)),
                             "pred rhat p95": float(np.percentile(r_pt, 95)),
                             "pred rhat max": float(r_pt.max()), "rmse rhat": r_rmse,
                             "pred rhat classic median": float(np.median(r_pt_classic)),
                             "rmse rhat classic": r_rmse_classic}])
        if os.path.exists(args.out):
            old = pd.read_csv(args.out)
            row = pd.concat([old[old["tag"] != tag], row], ignore_index=True)
        row.to_csv(args.out, index=False)
        print(f"[info] wrote {tag} to {args.out}", flush=True)

    print("\nall chains:", score(np.arange(D)).round(3).to_dict())
    rng = np.random.default_rng(1)
    diffs = []
    for _ in range(20):
        o = rng.permutation(D)
        diffs.append((score(o[:D // 2]) - score(o[D // 2:])).abs())
    diffs = pd.DataFrame(diffs)
    print("difference between two random halves of the chains (median, max of 20 splits):")
    for c in diffs:
        print(f"  {c:8s} {diffs[c].median():.4f}  {diffs[c].max():.4f}")

    rows = [{"chain": d, **score(np.array([d])).to_dict()} for d in range(D)]
    print("\nper chain:")
    print(pd.DataFrame(rows).round(3).sort_values("rmse").to_string(index=False))


if __name__ == "__main__":
    main()
