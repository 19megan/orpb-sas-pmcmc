# Ch1 evaluation: validation accuracy, parameter identifiability and cost of each sampling design.
#
# Joins three tables, one row per run and validation period:
#   validation_scores.csv   validate.py: rmse, bias, crps, cover90 per held-out period and their mean
#   run_summary.csv         compile_runs.py: KLD (+ split uncertainty), R-hat, ESS, N, D, L, accept
#   'Evaluation Table'      sheet of the cost workbook: samples and Average / Min / Max cost per
#                           (Scenario, Resolution, Window), from sample_counts.py
# A daily run is the same design in both scenarios (stream and precip both daily), so when there
# are Scenario 2 runs the Scenario 1 daily runs are also shown as Scenario 2.
#
# Outputs in --out-dir:
#   evaluation_table.csv                every run x period with scores, KLD, convergence, cost
#   <scenario>_validation.png           scores vs resolution, one column per validation period
#   <scenario>_kld.png                  KLD vs resolution per physical parameter; bars span KLD split
#   <scenario>_cost_tradeoff.png        rmse and crps vs cost (bars: min-max sample price), per period
#   <scenario>_rhat.png                 parameter rank R-hat per run
#   <scenario>_pred_rhat.png            predictive R-hat (--predictive-rhat, from predictive_by_chain.py
#                                       --out) of runs that fail on parameters; only if there are any
#   <scenario>_cost.png, _cost_by_window.png   cost of every design in the cost sheet, run or not
# crps and cover90 include sigma C out, which in Scenario 2 is the error of a coarse mean: compare
# them only within one scenario (and output resolution); rmse and bias compare across everything.
#
# Usage:
#   python evaluate_ch1.py --validation <folder>/validation_scores.csv --runs-summary <folder>/run_summary.csv
#       --workbook "<Ch1 Results>/Ch1 Isotope Sampling Cost Analysis.xlsx" --out-dir <folder>
#
# Created on: 2026-11-16
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np   # noqa: E402
import pandas as pd   # noqa: E402

RES_ORDER = ["daily", "weekly", "biweekly", "monthly"]
RES_TICKS = ["D", "W", "2W", "ME"]
WINDOW_ORDER = ["3M_winter", "3M_spring", "3M_summer", "3M_fall", "6M_cold", "6M_warm",
                "1Y_avg", "1Y_dry", "2Y", "4Y"]
# colour = record length (fixed categorical order), marker / fill = replicate window of that length
LENGTH_COLORS = {"3M": "#2a78d6", "6M": "#eb6834", "1Y": "#1baf7a", "2Y": "#eda100", "4Y": "#e87ba4"}
# REP_MARKERS = ["o", "s", "^", "D"]
# marker per window: 6M cold square, 6M warm circle
WINDOW_MARKERS = {"3M_winter": "s", "3M_spring": "^", "3M_summer": "o", "3M_fall": "D",
                  "6M_cold": "s", "6M_warm": "o", "1Y_avg": "o", "1Y_dry": "^", "2Y": "o", "4Y": "o"}
REP_FILLS = ["full", "none", "left", "bottom"]
# marker per resolution in the cost figures
RES_MARKERS = {"daily": "o", "weekly": "^", "biweekly": "D", "monthly": "s"}
PERIOD_ORDER = ["2014 Jul-Dec", "2015-04", "2016-10", "2017-01", "2018-07", "mean of periods"]
METRICS = {"rmse": "RMSE (‰)", "bias": "Bias (‰)", "crps": "CRPS (‰)", "cover90": "90% coverage"}
PHYSICAL = ["qf scale", "bf1 a", "bf1 M", "ET scale", "C_old"]
S1, S2 = "Scenario 1", "Scenario 2"


def scenario_key(label):
    return str(label).split(":")[0].strip()        # "Scenario 1: ..." -> "Scenario 1"


def style(window):
    length = window.split("_")[0]
    reps = [w for w in WINDOW_ORDER if w.split("_")[0] == length]
    i = reps.index(window) if window in reps else 0
    # return LENGTH_COLORS.get(length, "#52514e"), REP_MARKERS[i], REP_FILLS[i]
    return LENGTH_COLORS.get(length, "#52514e"), WINDOW_MARKERS.get(window, "o"), REP_FILLS[i]


def windows_in(df):
    return [w for w in WINDOW_ORDER if w in set(df["Window"])] + \
           sorted(set(df["Window"]) - set(WINDOW_ORDER))


def periods_in(df):
    return [p for p in PERIOD_ORDER if p in set(df["period"])] + \
           sorted(set(df["period"]) - set(PERIOD_ORDER))


def load_cost(args):
    """The cost sheet, every design (scenario x resolution x window), whether or not it has a run."""
    cost = pd.read_excel(args.workbook, sheet_name=args.sheet)
    cost["scenario"] = cost["Scenario"].map(scenario_key)
    cost["Resolution"] = cost["Resolution"].str.strip().str.lower()
    cost["Window"] = cost["Window"].str.strip()
    return cost.drop(columns="Scenario")


def load(args):
    val = pd.read_csv(args.validation)
    runs = pd.read_csv(args.runs_summary)
    # cost = pd.read_excel(args.workbook, sheet_name=args.sheet)
    cost = load_cost(args)
    # predictive R-hats from predictive_by_chain.py --out, only for the runs it was run on
    if args.predictive_rhat:
        pred = pd.read_csv(args.predictive_rhat)
        runs = runs.merge(pred, on="tag", how="left", validate="one_to_one")

    runs["scenario"] = runs["Scenario"].map(scenario_key)
    # window from the tag, in case run_summary.csv was written by an older compile_runs.py
    runs["Window"] = runs["tag"].map(
        lambda t: re.search(r"_(?:h|D|W|2W|ME)(\d+[MY](?:_[a-z]+)?)_job", t).group(1))
    df = val.merge(runs.drop(columns=["date_start", "date_end", "excluded_chains"]), on="tag",
                   how="left", validate="many_to_one")
    missing = df.loc[df["scenario"].isna(), "tag"].unique()
    if len(missing):
        print(f"[warn] not in {args.runs_summary}, left out: {list(missing)}")
        df = df.dropna(subset=["scenario"])

    # the daily design is shared by both scenarios
    if (df["scenario"] == S2).any():
        have = set(df.loc[(df["scenario"] == S2) & (df["Resolution"] == "daily"), "Window"])
        shared = df[(df["scenario"] == S1) & (df["Resolution"] == "daily") & ~df["Window"].isin(have)]
        df = pd.concat([df, shared.assign(scenario=S2)], ignore_index=True)

    # cost["scenario"] = cost["Scenario"].map(scenario_key)
    # cost["Resolution"] = cost["Resolution"].str.strip().str.lower()
    # cost["Window"] = cost["Window"].str.strip()
    # cost = cost.drop(columns="Scenario")
    df = df.merge(cost, on=["scenario", "Resolution", "Window"], how="left", validate="many_to_one")
    nocost = df.loc[df["Average cost"].isna(), "tag"].unique()
    if len(nocost):
        print(f"[warn] no cost row in '{args.sheet}' for: {list(nocost)}")
    df["Scenario"] = df["scenario"]
    return df.drop(columns="scenario")


def fig_validation(df, scen, out):
    periods, windows = periods_in(df), windows_in(df)
    fig, axes = plt.subplots(len(METRICS), len(periods), figsize=(2.6 * len(periods), 2.3 * len(METRICS)),
                             sharex=True, sharey="row", squeeze=False)
    x = np.arange(len(RES_ORDER))
    for j, period in enumerate(periods):
        d = df[df["period"] == period]
        for i, (metric, ylab) in enumerate(METRICS.items()):
            ax = axes[i, j]
            ref = {"bias": 0.0, "cover90": 0.9}.get(metric)
            if ref is not None:
                ax.axhline(ref, color="#52514e", lw=1, ls="--", zorder=1)
            for w in windows:
                dw = d[d["Window"] == w].set_index("Resolution").reindex(RES_ORDER)
                c, mk, fill = style(w)
                ax.plot(x, dw[metric], color=c, lw=2, marker=mk, ms=8, fillstyle=fill,
                        markeredgewidth=1.5, label=w.replace("_", " "), zorder=3)
            if i == 0:
                n = d["n_obs"].iloc[0] if len(d) else 0
                ax.set_title(f"{period}\n(n = {int(n)})", fontsize=10)
            if j == 0:
                ax.set_ylabel(ylab)
            ax.set_xticks(x, RES_TICKS)
            ax.grid(axis="y", color="#e5e4e0", lw=0.8)
            ax.spines[["top", "right"]].set_visible(False)
    axes[-1, 0].set_xlabel("Precip resolution" if scen == S1 else "Precip and stream resolution")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, title="Window", loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle(f"{scen}: validation on the held-out stream 18O", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out, f"{scen.replace(' ', '_')}_validation.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_kld(runs, scen, out):
    windows = windows_in(runs)
    fig, axes = plt.subplots(1, len(PHYSICAL), figsize=(2.8 * len(PHYSICAL), 3.0), sharey=True)
    x = np.arange(len(RES_ORDER))
    off = np.linspace(-0.12, 0.12, len(windows)) if len(windows) > 1 else [0.0]
    for ax, par in zip(axes, PHYSICAL):
        for k, w in enumerate(windows):
            dw = runs[runs["Window"] == w].set_index("Resolution").reindex(RES_ORDER)
            c, mk, fill = style(w)
            # bar spans the largest difference between two random halves of the chains
            ax.errorbar(x + off[k], dw[f"{par} KLD"], yerr=dw[f"{par} KLD split"] / 2, color=c, lw=2,
                        marker=mk, ms=8, fillstyle=fill, markeredgewidth=1.5, capsize=0, elinewidth=1.5,
                        label=w.replace("_", " "), zorder=3)
        ax.set_title(par, fontsize=10)
        ax.set_xticks(x, RES_TICKS)
        ax.grid(axis="y", color="#e5e4e0", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("KLD posterior || prior (nats)")
    axes[0].set_ylim(bottom=0)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, title="Window", loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle(f"{scen}: parameter identifiability (bars: KLD difference between chain halves)",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(out, f"{scen.replace(' ', '_')}_kld.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _by_resolution_panel(ax, d, col, windows, err_lo=None, err_hi=None):
    """As fig_kld: x = resolution, one line per window, optional asymmetric bars."""
    x = np.arange(len(RES_ORDER))
    off = np.linspace(-0.12, 0.12, len(windows)) if len(windows) > 1 else [0.0]
    for k, w in enumerate(windows):
        dw = d[d["Window"] == w].set_index("Resolution").reindex(RES_ORDER)
        c, mk, fill = style(w)
        # yerr = None if err_lo is None else [dw[col] - dw[err_lo], dw[err_hi] - dw[col]]
        # ax.errorbar(x + off[k], dw[col], yerr=yerr, color=c, lw=2, marker=mk, ms=8, fillstyle=fill,
        #             markeredgewidth=1.5, capsize=0, elinewidth=1.5, label=w.replace("_", " "), zorder=3)
        if err_lo is not None:   # light bars so they don't swamp the points
            ax.errorbar(x + off[k], dw[col], yerr=[dw[col] - dw[err_lo], dw[err_hi] - dw[col]], color=c,
                        alpha=0.35, elinewidth=1, capsize=0, ls="none", zorder=2)
        ax.plot(x + off[k], dw[col], color=c, lw=2, marker=mk, ms=8, fillstyle=fill,
                markeredgewidth=1.5, label=w.replace("_", " "), zorder=3)
    ax.set_xticks(x, RES_TICKS)
    ax.grid(axis="y", color="#e5e4e0", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)


# RHAT_PASS = 1.05   # a run passes when every parameter's rank R-hat is below this
# a run passes when its worst parameter rank R-hat (max of bulk and folded), to 2 decimals, is at
# most this
# RHAT_PASS = 1.06
RHAT_PASS = 1.05
RHAT_PARAMS = PHYSICAL + ["sigma C in", "sigma C out"]


def failed_rhat(runs):
    return runs[[f"{p} rhat" for p in RHAT_PARAMS]].max(axis=1).round(2) > RHAT_PASS


def _rhat_figure(panels, windows, title, ylabel, path):
    """One row of by-resolution panels (title, column, rows, bar low, bar high), shared y."""
    fig, axes = plt.subplots(1, len(panels), figsize=(2.4 * len(panels), 3.2), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (t, col, d, lo, hi) in zip(axes, panels):
        ax.axhline(RHAT_PASS, color="#52514e", lw=1, ls="--", zorder=1)
        _by_resolution_panel(ax, d, col, windows, err_lo=lo, err_hi=hi)
        ax.set_title(t, fontsize=9)
    axes[0].set_ylabel(ylabel)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, title="Window", loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_rhat(runs, scen, out):
    """Rank R-hat per parameter, every run."""
    panels = [(p, f"{p} rhat", runs, None, None) for p in RHAT_PARAMS]
    return _rhat_figure(panels, windows_in(runs),
                        f"{scen}: parameter convergence (dashed: R-hat {RHAT_PASS})", "Rank R-hat",
                        os.path.join(out, f"{scen.replace(' ', '_')}_rhat.png"))


def fig_pred_rhat(runs, scen, out):
    """Predictive R-hats (predictive_by_chain.py --out) of the runs that did not pass on parameters:
    do the chains agree on the prediction anyway? None if there are no such values."""
    if "pred rhat median" not in runs:
        return None
    pred = runs[failed_rhat(runs) & runs["pred rhat median"].notna()]
    if pred.empty:
        return None
    panels = [("prediction at each scored date\n(point: median; bar: to 95th pct)", "pred rhat median",
               pred, "pred rhat median", "pred rhat p95"),
              ("per-draw RMSE", "rmse rhat", pred, None, None)]
    return _rhat_figure(panels, windows_in(runs),
                        f"{scen}: predictive convergence of runs with a parameter R-hat above "
                        f"{RHAT_PASS} (dashed: {RHAT_PASS})", "R-hat",
                        os.path.join(out, f"{scen.replace(' ', '_')}_pred_rhat.png"))


def fig_rmse(df, scen, out):
    """RMSE vs resolution, one panel per validation period and the mean of periods."""
    periods, windows = periods_in(df), windows_in(df)
    fig, axes = plt.subplots(1, len(periods), figsize=(2.8 * len(periods), 3.0), sharey=True)
    for ax, period in zip(axes, periods):
        _by_resolution_panel(ax, df[df["period"] == period], "rmse", windows)
        ax.set_title(period, fontsize=10)
    axes[0].set_ylabel(METRICS["rmse"])
    axes[0].set_ylim(bottom=0)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, title="Window", loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle(f"{scen}: RMSE of the held-out stream 18O", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out, f"{scen.replace(' ', '_')}_rmse.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_cost_by_res(cost, scen, out):
    """Cost vs resolution; one panel, since cost does not depend on the validation period.
    Bars span the min-max sample price. Drawn from the cost sheet: every design, run or not."""
    # runs, windows = df.drop_duplicates("tag"), windows_in(df)
    runs, windows = cost, windows_in(cost)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    _by_resolution_panel(ax, runs, "Average cost", windows, err_lo="Min cost", err_hi="Max cost")
    ax.set_ylabel("Cost (USD)")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, title="Window", loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle(f"{scen}: sampling cost\n(point: average sample price; bars: min-max price)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out, f"{scen.replace(' ', '_')}_cost.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# resolution is ordered (daily -> monthly), so one hue stepped light -> dark, with the markers
RES_COLORS = {"daily": "#86b6ef", "weekly": "#3987e5", "biweekly": "#1c5cab", "monthly": "#0d366b"}


def fig_cost_by_window(cost, scen, out):
    """Cost vs window, one line per resolution; bars span the min-max sample price.
    Drawn from the cost sheet: every design, run or not."""
    windows = windows_in(cost)
    x = np.arange(len(windows))
    off = np.linspace(-0.15, 0.15, len(RES_ORDER))
    fig, ax = plt.subplots(figsize=(0.75 * len(windows) + 2.0, 3.6))
    for k, (res, tick) in enumerate(zip(RES_ORDER, RES_TICKS)):
        d = cost[cost["Resolution"] == res].set_index("Window").reindex(windows)
        c = RES_COLORS[res]
        ax.errorbar(x + off[k], d["Average cost"],
                    yerr=[d["Average cost"] - d["Min cost"], d["Max cost"] - d["Average cost"]],
                    color=c, alpha=0.35, elinewidth=1, capsize=0, ls="none", zorder=2)
        ax.plot(x + off[k], d["Average cost"], color=c, lw=1.5, marker=RES_MARKERS[res], ms=7,
                label=tick, zorder=3)
    ax.set_xticks(x, [w.replace("_", " ") for w in windows], rotation=45, ha="right")
    ax.set_ylabel("Cost (USD)")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", color="#e5e4e0", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(title="Resolution", loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle(f"{scen}: sampling cost by window\n(point: average sample price; bars: min-max price)",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(out, f"{scen.replace(' ', '_')}_cost_by_window.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_cost(df, scen, out):
    periods, windows = periods_in(df), windows_in(df)
    metrics = ["rmse", "crps"]
    fig, axes = plt.subplots(len(metrics), len(periods), figsize=(2.6 * len(periods), 2.6 * len(metrics)),
                             sharex=True, sharey="row", squeeze=False)
    # res_markers = dict(zip(RES_ORDER, ["o", "s", "^", "D"]))
    res_markers = RES_MARKERS
    for j, period in enumerate(periods):
        d = df[df["period"] == period]
        for i, metric in enumerate(metrics):
            ax = axes[i, j]
            for _, r in d.iterrows():
                c, _, fill = style(r["Window"])
                # the min-max price range is the same ratio for every design, so it is drawn once, in
                # the last column, rather than repeated in every period
                if j == len(periods) - 1:
                    ax.errorbar(r["Average cost"], r[metric],
                                xerr=[[r["Average cost"] - r["Min cost"]], [r["Max cost"] - r["Average cost"]]],
                                color=c, alpha=0.35, lw=1, capsize=0, zorder=2)
                # ax.plot(r["Average cost"], r[metric], marker=res_markers[r["Resolution"]], ms=8, color=c,
                #         markeredgecolor="#fcfcfb", markeredgewidth=1, ls="none", zorder=3)
                ax.plot(r["Average cost"], r[metric], marker=res_markers[r["Resolution"]], ms=8, color=c,
                        fillstyle=fill, markeredgewidth=1.5, ls="none", zorder=3)
            ax.set_xscale("log")
            if i == 0:
                ax.set_title(period, fontsize=10)
            if j == 0:
                ax.set_ylabel(METRICS[metric])
            ax.grid(color="#e5e4e0", lw=0.8)
            ax.spines[["top", "right"]].set_visible(False)
        axes[-1, j].set_xlabel("Cost (USD)")
    handles = [plt.Line2D([], [], color="#52514e", marker=m, ls="none", ms=8, label=t)
               for m, t in zip(res_markers.values(), RES_TICKS)]
    # handles += [plt.Line2D([], [], color=LENGTH_COLORS[w.split("_")[0]], lw=6, label=w.split("_")[0])
    #             for w in dict.fromkeys(w.split("_")[0] + "_" for w in windows)]
    handles += [plt.Line2D([], [], color=style(w)[0], marker="o", fillstyle=style(w)[2], ms=8,
                           markeredgewidth=1.5, ls="none", label=w.replace("_", " ")) for w in windows]
    fig.legend(handles=handles, title="Resolution / window", loc="center left",
               bbox_to_anchor=(1.0, 0.5), frameon=False)
    # fig.suptitle(f"{scen}: accuracy vs cost (point: average sample price; bar: min-max price)",
    #              fontsize=11)
    fig.suptitle(f"{scen}: accuracy vs cost (point: average sample price; bars in last column: "
                 f"min-max price)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out, f"{scen.replace(' ', '_')}_cost_tradeoff.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# Marker AREA proportional to total KLD (sum over the physical parameters, nats), anchored at
# 0 nats = 0 area with a fixed constant (not min-max stretched), as in the old Ch1 script: equal
# information gain gives equal size. The sum of marginal KLDs is a lower bound on the joint
# information gained (independent priors).
AREA_PER_NAT = 12.0
SIZE_LEGEND_NATS = [5, 10, 20]


def fig_cost_kld(df, scen, out):
    """fig_cost with marker area = total KLD: accuracy vs cost vs identifiability in one figure."""
    periods, windows = periods_in(df), windows_in(df)
    metrics = ["rmse", "crps"]
    fig, axes = plt.subplots(len(metrics), len(periods), figsize=(2.6 * len(periods), 2.8 * len(metrics)),
                             sharex=True, sharey="row", squeeze=False)
    total = df[[f"{p} KLD" for p in PHYSICAL]].sum(axis=1, min_count=len(PHYSICAL))
    df = df.assign(total_kld=total)
    for j, period in enumerate(periods):
        d = df[df["period"] == period].sort_values("total_kld", ascending=False)   # small on top
        for i, metric in enumerate(metrics):
            ax = axes[i, j]
            for _, r in d.iterrows():
                c, _, fill = style(r["Window"])
                # min-max price bars left out: on the linear cost axis they run to ~3x the average
                # cost (off the panel); fig_cost shows them
                # if j == len(periods) - 1:      # min-max price range, drawn once
                #     ax.errorbar(r["Average cost"], r[metric],
                #                 xerr=[[r["Average cost"] - r["Min cost"]], [r["Max cost"] - r["Average cost"]]],
                #                 color=c, alpha=0.35, lw=1, capsize=0, zorder=2)
                ax.plot(r["Average cost"], r[metric], marker=RES_MARKERS[r["Resolution"]],
                        ms=np.sqrt(AREA_PER_NAT * max(r["total_kld"], 0)), color=c, fillstyle=fill,
                        markeredgewidth=1.5, alpha=0.85, ls="none", zorder=3)
            # ax.set_xscale("log")
            # ax.set_xlim(left=0)
            ax.set_xlim(0, 1.12 * df["Average cost"].max())   # room for the largest markers
            if i == 0:
                ax.set_title(period, fontsize=10)
            if j == 0:
                ax.set_ylabel(METRICS[metric])
            ax.grid(color="#e5e4e0", lw=0.8)
            ax.spines[["top", "right"]].set_visible(False)
        axes[-1, j].set_xlabel("Cost (USD)")
    handles = [plt.Line2D([], [], color="#52514e", marker=RES_MARKERS[r], ls="none", ms=8, label=t)
               for r, t in zip(RES_ORDER, RES_TICKS)]
    handles += [plt.Line2D([], [], color=style(w)[0], marker="o", fillstyle=style(w)[2], ms=8,
                           markeredgewidth=1.5, ls="none", label=w.replace("_", " ")) for w in windows]
    handles += [plt.Line2D([], [], color="#8f8d86", marker="o", ls="none", ms=np.sqrt(AREA_PER_NAT * n),
                           label=f"{n} nats") for n in SIZE_LEGEND_NATS]
    fig.legend(handles=handles, title="Resolution / window /\ntotal KLD (marker area)", loc="center left",
               bbox_to_anchor=(1.0, 0.5), frameon=False, labelspacing=1.1)
    fig.suptitle(f"{scen}: accuracy vs cost vs identifiability (marker area: KLD summed over "
                 f"{len(PHYSICAL)} parameters; cost at the average sample price)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out, f"{scen.replace(' ', '_')}_cost_kld_tradeoff.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_grids(df, scen, out):
    """Resolution x window grids of the mean-of-periods scores and the cost, values printed."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#104281"])   # one hue, light -> dark
    d = df[df["period"] == "mean of periods"]
    windows = windows_in(d)
    panels = {"rmse": ("RMSE (‰)", "{:.2f}"), "crps": ("CRPS (‰)", "{:.2f}"),
              "cover90": ("90% coverage", "{:.2f}"), "Average cost": ("Cost (USD, average price)", "{:,.0f}")}
    fig, axes = plt.subplots(1, len(panels), figsize=(1.2 * len(windows) * len(panels) + 1.5, 3.4))
    for ax, (col, (title, fmt)) in zip(axes, panels.items()):
        grid = d.pivot_table(index="Resolution", columns="Window", values=col).reindex(
            index=RES_ORDER, columns=windows)
        v = grid.to_numpy(float)
        ax.imshow(v, cmap=cmap, aspect="auto")
        lo, hi = np.nanmin(v), np.nanmax(v)
        for (a, b), val in np.ndenumerate(v):
            if np.isfinite(val):
                dark = hi > lo and (val - lo) / (hi - lo) > 0.55
                ax.text(b, a, fmt.format(val), ha="center", va="center", fontsize=8,
                        color="#fcfcfb" if dark else "#0b0b0b")
        ax.set_xticks(range(len(windows)), [w.replace("_", " ") for w in windows], fontsize=8)
        ax.set_yticks(range(len(RES_ORDER)), RES_TICKS)
        ax.set_title(title, fontsize=10)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(f"{scen}: mean of validation periods and cost (darker = larger value)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out, f"{scen.replace(' ', '_')}_grids.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser(description="Ch1 evaluation tables and figures")
    p.add_argument("--validation", required=True, help="validation_scores.csv from validate.py")
    p.add_argument("--runs-summary", required=True, help="run_summary.csv from compile_runs.py")
    p.add_argument("--workbook", required=True, help="cost workbook (.xlsx)")
    p.add_argument("--sheet", default="Evaluation Table")
    p.add_argument("--predictive-rhat", default=None,
                   help="optional CSV from predictive_by_chain.py --out (runs it was not run on are blank)")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load(args)
    path = os.path.join(args.out_dir, "evaluation_table.csv")
    df.to_csv(path, index=False)
    print(f"[info] wrote {path} ({df['tag'].nunique()} runs, {len(df)} rows)")
    for scen in sorted(df["Scenario"].unique()):
        d = df[df["Scenario"] == scen]
        runs = d.drop_duplicates("tag")
        # for f in (fig_validation(d, scen, args.out_dir), fig_kld(runs, scen, args.out_dir),
        #           fig_cost(d, scen, args.out_dir)):
        # for f in (fig_validation(d, scen, args.out_dir), fig_kld(runs, scen, args.out_dir),
        #           fig_rmse(d, scen, args.out_dir), fig_cost_by_res(d, scen, args.out_dir),
        #           fig_cost(d, scen, args.out_dir), fig_cost_kld(d, scen, args.out_dir),
        #           fig_grids(d, scen, args.out_dir)):
        for f in (fig_validation(d, scen, args.out_dir), fig_kld(runs, scen, args.out_dir),
                  fig_rhat(runs, scen, args.out_dir), fig_pred_rhat(runs, scen, args.out_dir),
                  fig_rmse(d, scen, args.out_dir),
                  fig_cost(d, scen, args.out_dir), fig_cost_kld(d, scen, args.out_dir),
                  fig_grids(d, scen, args.out_dir)):
            if f:
                print(f"[info] wrote {f}")
    # cost only: every design in the cost sheet, whether or not it has a validated run
    cost = load_cost(args)
    for scen in sorted(cost["scenario"].unique()):
        c = cost[cost["scenario"] == scen]
        for f in (fig_cost_by_res(c, scen, args.out_dir), fig_cost_by_window(c, scen, args.out_dir)):
            print(f"[info] wrote {f}")


if __name__ == "__main__":
    main()
