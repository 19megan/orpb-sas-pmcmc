# Number of isotope samples in each calibration design, for the Ch1 cost worksheet.
#
# Counts come from the same dataset files the calibration reads (data_filename in
# ORPB_pmcmc_script_parallel.py), inside the calibration window only, so the spinup year is never
# counted (its precip 18O is the isoscape, not samples).
#   Precip samples           precip 18O values (mean_c) in the window: intervals with no
#                            precipitation have no sample
#   Stream samples           ORPB 18O values in the window, before the held-out months are masked
#   Stream samples calibrated  the same after mask_holdout, i.e. what the calibration saw
# Scenario 1: stream daily, precip at the resolution. Scenario 2: both at the resolution (stream
# values there are interval means of the grab samples, so one composite sample per interval).
#
# Usage:
#   python sample_counts.py --data-root <ORPB_resolution_datasets> --out <folder>/sample_counts.csv
#
# Created on: 2026-11-16
import argparse
import os
import sys

script_folder = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, script_folder)
# ORPB_cases stays with the run scripts in tests_mesas
sys.path.insert(0, os.path.join(os.path.dirname(script_folder), "tests_mesas"))
import pandas as pd   # noqa: E402

from ORPB_cases import mask_holdout   # noqa: E402

RESOLUTIONS = ["daily", "weekly", "biweekly", "monthly"]
# Window label -> (start, end), as in the run tags. Add the remaining windows when they are run.
WINDOWS = {"3M_winter": ("2015-01-01", "2015-03-31"), "3M_spring": ("2016-03-01", "2016-05-31"),
           "3M_summer": ("2017-06-01", "2017-08-31"), "3M_fall": ("2018-08-01", "2018-10-31"),
           "6M_cold": ("2015-10-01", "2016-03-31"), "6M_warm": ("2017-04-01", "2017-09-30"),
           "1Y_avg": ("2015-05-01", "2016-04-30"), "1Y_dry": ("2017-03-01", "2018-02-28"),
           "2Y": ("2015-01-01", "2016-12-31"),
           "4Y": ("2015-01-01", "2018-12-31")}   # check against the 4Y run when it exists
SCENARIOS = {False: "Scenario 1: stream daily, precip varies", True: "Scenario 2: stream = precip resolution"}


def filename(resolution, coarsen_output):
    # same as data_filename in ORPB_pmcmc_script_parallel.py
    if coarsen_output:
        return f"ORPB_isotope_data_precip 18O_{resolution}_ORPB 18O_{resolution}.csv"
    return f"ORPB_isotope_data_precip 18O_{resolution}.csv"


def main():
    p = argparse.ArgumentParser(description="Isotope sample counts per calibration design")
    p.add_argument("--data-root", default=os.environ.get(
        "MESAS_DATA_ROOT", r"C:\Users\simon\Desktop\ORPB_resolution_datasets"))
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = []
    for coarsen, scenario in SCENARIOS.items():
        for res in RESOLUTIONS:
            path = os.path.join(args.data_root, filename(res, coarsen and res != "daily"))
            df = pd.read_csv(path, index_col=0, parse_dates=[0])
            for window, (start, end) in WINDOWS.items():
                w = df.loc[start:end].copy()
                n_stream = int(w["ORPB 18O"].notna().sum())
                mask_holdout(w, res if coarsen else "daily")
                rows.append({"Scenario": scenario, "Resolution": res, "Window": window,
                             "Start": start, "End": end, "Days": len(pd.date_range(start, end)),
                             "Rain days": int(w.loc[w['rainfall (mm/hr)'] > 0, 'rainfall (mm/hr)'].notna().sum()),
                             "Precip samples": int(w["mean_c"].notna().sum()),
                             "Stream samples": n_stream,
                             "Stream samples calibrated": int(w["ORPB 18O"].notna().sum())})
    out = pd.DataFrame(rows)
    out["Total samples"] = out["Precip samples"] + out["Stream samples"]
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
