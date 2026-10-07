"""Run the full PPMI progression analysis.

    python run_all.py [--data-dir PATH] [--out-dir PATH]

Writes aggregate tables (CSV) and figures (PNG) only. The participant-level
analysis file is saved under output/derived/ for your own checks; it is
git-ignored and must not be shared or published (PPMI DUA).
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
import models_lmm
import models_survival
import plot_forest
from load import build_dataset


def table1(long: pd.DataFrame) -> pd.DataFrame:
    bl = long[long["EVENT_ID"] == "BL"]
    per = long.groupby("PATNO")
    fu = per["years"].max()
    visits = per.size()

    def mean_sd(x):
        x = x.dropna()
        return f"{x.mean():.1f} ({x.std():.1f})", len(x)

    def median_iqr(x):
        x = x.dropna()
        return f"{x.median():.1f} [{x.quantile(.25):.1f}, {x.quantile(.75):.1f}]", len(x)

    rows = [("Participants", f"{long['PATNO'].nunique()}", long["PATNO"].nunique()),
            ("Age at baseline, mean (SD)", *mean_sd(bl["age_bl"])),
            ("Male, n (%)", f"{int(bl['male'].sum())} ({100 * bl['male'].mean():.1f}%)", bl["male"].notna().sum()),
            ("Disease duration at baseline, y, median [IQR]", *median_iqr(bl["dur_bl"]))]
    if bl["genetic"].notna().any():
        rows.append(("Genetic enrolment, n (%)",
                     f"{int(bl['genetic'].sum())} ({100 * bl['genetic'].mean():.1f}%)", bl["genetic"].notna().sum()))
    for col, lab in [("updrs1", "MDS-UPDRS I"), ("updrs2", "MDS-UPDRS II"),
                     ("updrs3", "MDS-UPDRS III"), ("nhy", "Hoehn & Yahr"),
                     ("moca", "MoCA"), ("ledd", "LEDD, mg/day")]:
        if col == "ledd":
            treated = bl["ledd"] > 0
            rows.append(("Treated at baseline (LEDD > 0), n (%)",
                         f"{int(treated.sum())} ({100 * treated.mean():.1f}%)", bl["ledd"].notna().sum()))
        rows.append((f"Baseline {lab}, mean (SD)", *mean_sd(bl[col])))
        if col == "moca":
            n_sc = int(bl["moca_bl_from_sc"].sum())
            rows.append(("  of which taken from screening visit, n", f"{n_sc}", bl["moca"].notna().sum()))
    rows += [("Follow-up, y, median [IQR]", *median_iqr(fu)),
             ("Visits per participant, median [IQR]", *median_iqr(visits.astype(float)))]
    return pd.DataFrame(rows, columns=["characteristic", "value", "n_non_missing"])


def analyse(long: pd.DataFrame, out_dir: Path):
    print("Mixed-effects models ...")
    models_lmm.run(long, out_dir)
    print("Survival models ...")
    models_survival.run(long, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=cfg.DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=cfg.DEFAULT_OUT_DIR)
    args = ap.parse_args()
    for sub in ("tables", "figures", "derived"):
        (args.out_dir / sub).mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("Building analysis dataset ...")
    long = build_dataset(args.data_dir)
    long.to_csv(args.out_dir / "derived" / "analysis_long_PRIVATE.csv", index=False)
    print(f"  {long['PATNO'].nunique()} PD participants, {len(long)} visits")
    table1(long).to_csv(args.out_dir / "tables" / "table1_baseline.csv", index=False)

    analyse(long, args.out_dir)

    # Sensitivity: participants untreated at baseline (true de novo subset).
    bl = long[long["EVENT_ID"] == "BL"]
    ids = bl.loc[bl["ledd"].eq(0) & bl["med_state"].eq("untreated"), "PATNO"]
    sub = long[long["PATNO"].isin(ids)]
    sens_dir = args.out_dir / "sensitivity_untreated_bl"
    for d in ("tables", "figures"):
        (sens_dir / d).mkdir(parents=True, exist_ok=True)
    print(f"Sensitivity: untreated at baseline ({sub['PATNO'].nunique()} participants) ...")
    table1(sub).to_csv(sens_dir / "tables" / "table1_baseline.csv", index=False)
    analyse(sub, sens_dir)
    plot_forest.run(args.out_dir)
    print(f"Done. Results in {args.out_dir / 'tables'} and {args.out_dir / 'figures'}")


if __name__ == "__main__":
    main()
