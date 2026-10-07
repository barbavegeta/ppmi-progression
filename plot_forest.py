"""Forest plot: Cox hazard ratios in the full cohort vs participants untreated
at baseline, one panel per milestone.

    python plot_forest.py [--out-dir output]

Reads the aggregate tables written by run_all.py (tables/cox_hazard_ratios.csv
and sensitivity_untreated_bl/tables/cox_hazard_ratios.csv).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg

MILESTONES = ["Hoehn & Yahr >= 3", "Cognitive impairment (MCI/dementia)", "MoCA < 26, confirmed"]
TERMS = ["Age at baseline (per 10 y)", "Male sex", "Disease duration at baseline (per y)",
         "Baseline MDS-UPDRS III (per 10 points)", "Baseline MoCA (per point)"]
SERIES = [("Full cohort", "#2a78d6", "o", 0.17),
          ("Untreated at baseline", "#eb6834", "s", -0.17)]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def forest(main_csv: Path, sens_csv: Path, out_png: Path):
    tabs = [pd.read_csv(main_csv), pd.read_csv(sens_csv)]
    lo = min(t["HR_low"].min() for t in tabs)
    hi = max(t["HR_high"].max() for t in tabs)

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": INK})
    fig, axes = plt.subplots(1, len(MILESTONES), figsize=(13, 4.2), sharey=True, facecolor=SURFACE)
    y = np.arange(len(TERMS))[::-1]
    for ax, ms in zip(axes, MILESTONES):
        ax.set_facecolor(SURFACE)
        ax.axvline(1, color=MUTED, lw=1, ls="--", zorder=1)
        for (label, color, marker, dy), t in zip(SERIES, tabs):
            r = t[t["milestone"] == ms].set_index("term").reindex(TERMS)
            ok = r["HR"].notna().values
            yy = y[ok] + dy
            ax.hlines(yy, r["HR_low"][ok], r["HR_high"][ok], color=color, lw=2, zorder=2)
            ax.scatter(r["HR"][ok], yy, s=42, marker=marker, color=color, zorder=3,
                       edgecolor=SURFACE, linewidth=1.5, label=label)
        ax.set_xscale("log")
        ax.set_xlim(lo / 1.1, hi * 1.1)
        ticks = [t for t in (0.5, 0.75, 1, 1.5, 2, 3) if lo / 1.1 <= t <= hi * 1.1]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(ms.replace(">=", "≥"), fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Hazard ratio (95% CI, log scale)", color=MUTED)
    axes[0].set_yticks(y, [t.replace(" at baseline", "") for t in TERMS])
    axes[0].set_ylim(-0.6, len(TERMS) - 0.4)
    axes[0].legend(loc="upper left", bbox_to_anchor=(-0.02, 1.17), ncol=2, frameon=False, fontsize=9)
    fig.text(0.01, 0.01, "Missing marker: covariate not in that model. Models also adjust for "
             "genetic enrolment in the full cohort.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out_png, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def run(out_dir: Path):
    main_csv = out_dir / "tables" / "cox_hazard_ratios.csv"
    sens_csv = out_dir / "sensitivity_untreated_bl" / "tables" / "cox_hazard_ratios.csv"
    if main_csv.exists() and sens_csv.exists():
        forest(main_csv, sens_csv, out_dir / "figures" / "cox_forest_main_vs_untreated.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=cfg.DEFAULT_OUT_DIR)
    run(ap.parse_args().out_dir)
