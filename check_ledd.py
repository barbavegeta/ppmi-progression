"""Aggregate checks on baseline LEDD (no participant-level output).

    python check_ledd.py [--data-dir PATH] [--out-dir PATH]

Answers: is the baseline LEDD real treatment (e.g. genetic-cohort participants
enrolled on medication) or an artefact of the LEDD log (duplicate or
overlapping rows, implausible doses)? Writes output/tables/ledd_check.csv.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from load import active_log_rows, build_dataset, find_files, parse_month, read_table


def pct(x) -> str:
    return f"{100 * np.mean(x):.1f}%" if len(x) else "n/a"


def summary(x: pd.Series) -> str:
    x = x.dropna()
    if not len(x):
        return "n/a"
    q = x.quantile([.5, .75, .9, .99])
    return (f"mean {x.mean():.0f}, median {q[.5]:.0f}, P75 {q[.75]:.0f}, "
            f"P90 {q[.9]:.0f}, P99 {q[.99]:.0f} (n={len(x)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=cfg.DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=cfg.DEFAULT_OUT_DIR)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)

    long = build_dataset(args.data_dir)
    log = read_table(find_files(args.data_dir)["ledd"], "ledd")
    pd_ids = set(long["PATNO"])
    log = log[log["PATNO"].isin(pd_ids)].copy()

    bl = long[long["EVENT_ID"].eq("BL")].copy()
    treated = bl["ledd"] > 0
    rows = [
        ("PD participants", len(pd_ids)),
        ("  with any row in the LEDD log", pct(pd.Series(list(pd_ids)).isin(log["PATNO"]))),
        ("Baseline LEDD, all", summary(bl["ledd"])),
        ("Baseline LEDD > 0", pct(treated)),
        ("Baseline LEDD among those > 0", summary(bl.loc[treated, "ledd"])),
    ]
    if bl["genetic"].notna().any():
        for g, lab in [(0, "sporadic"), (1, "genetic enrolment")]:
            s = bl[bl["genetic"].eq(g)]
            rows.append((f"Baseline LEDD > 0, {lab}", f"{pct(s['ledd'] > 0)} of {len(s)}"))
    for lo, hi, lab in [(0, 1, "< 1 y"), (1, 2, "1-2 y"), (2, np.inf, ">= 2 y")]:
        s = bl[(bl["dur_bl"] >= lo) & (bl["dur_bl"] < hi)]
        rows.append((f"Baseline LEDD > 0, disease duration {lab}", f"{pct(s['ledd'] > 0)} of {len(s)}"))

    # Agreement with the Part III exam's medication state at baseline.
    st = bl["med_state"].fillna("missing")
    for state in ["untreated", "OFF", "ON", "unknown", "missing"]:
        s = bl[st.eq(state)]
        if len(s):
            rows.append((f"BL Part III state '{state}': n, LEDD > 0",
                         f"{len(s)}, {pct(s['ledd'] > 0)}"))

    # The log itself.
    raw = log["LEDD"].astype(str).str.strip()
    mg = pd.to_numeric(raw, errors="coerce")
    is_mult = raw.str.contains(r"(?i)LD\s*[x\*]", regex=True)
    start = parse_month(log["STARTDT"])
    stop = parse_month(log["STOPDT"])
    dup_cols = ["PATNO", "LEDTRT", "STARTDT", "STOPDT", "LEDD"]
    rows += [
        ("LEDD log rows (PD participants)", len(log)),
        ("  numeric dose", int(mg.notna().sum())),
        ("  COMT multiplier ('LD x ...')", int(is_mult.sum())),
        ("  neither (dropped)", int((mg.isna() & ~is_mult).sum())),
        ("  missing start date (dropped)", int(start.isna().sum())),
        ("  stop before start", int((stop < start).sum())),
        ("  exact duplicates (same drug, dates, dose)", int(log.duplicated(dup_cols).sum())),
        ("  single-row dose > 2000 mg", int((mg > 2000).sum())),
        ("Visit LEDD, all visits", summary(long["ledd"])),
        ("Visits with LEDD > 3000 mg", int((long["ledd"] > 3000).sum())),
    ]

    # Same drug active twice in one visit month (overlapping prescriptions).
    v = long[cfg.KEY + ["date"]].dropna()
    m = v.merge(log.assign(start=start, stop=stop)[["PATNO", "LEDTRT", "start", "stop"]], on="PATNO")
    m = m[(m["start"] <= m["date"]) & (m["stop"].isna() | (m["stop"] >= m["date"]))]
    m["drug"] = m["LEDTRT"].astype(str).str.lower().str.strip()
    over = m.duplicated(cfg.KEY + ["drug"], keep=False)
    rows.append(("Visits where one drug has > 1 active log row",
                 f"{m.loc[over, cfg.KEY].drop_duplicates().shape[0]} of {len(v)}"))
    _, mc = active_log_rows(log, long)
    over_c = mc.duplicated(["_vid", "drug"], keep=False)
    rows.append(("  after cleaning (duplicates, dose-change months)",
                 f"{mc.loc[over_c, '_vid'].nunique()} of {len(v)}"))

    out = pd.DataFrame(rows, columns=["check", "value"])
    (args.out_dir / "tables").mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_dir / "tables" / "ledd_check.csv", index=False)
    with pd.option_context("display.max_colwidth", 120, "display.width", 200):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
