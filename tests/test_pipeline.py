"""End-to-end run on synthetic data (no PPMI data needed)."""
import subprocess
import sys

import pandas as pd

from conftest import ROOT


def test_full_pipeline_on_synthetic_data(tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"

    def run(*args):
        subprocess.run([sys.executable, *args], cwd=ROOT, check=True, capture_output=True, text=True)

    run("make_synthetic_data.py", "--out", str(data), "--n", "250")
    run("run_all.py", "--data-dir", str(data), "--out-dir", str(out))

    tab = out / "tables"
    for f in ["table1_baseline", "lmm_annual_change", "lmm_piecewise_change", "lmm_diagnostics",
              "cox_hazard_ratios", "cox_ph_test", "km_event_free_by_year", "survival_counts"]:
        assert (tab / f"{f}.csv").exists(), f
    for f in ["lmm_trajectories.png", "km_milestones_by_age.png", "cox_forest_main_vs_untreated.png"]:
        assert (out / "figures" / f).exists(), f
    assert (out / "sensitivity_untreated_bl" / "tables" / "cox_hazard_ratios.csv").exists()

    assert len(pd.read_csv(tab / "lmm_piecewise_change.csv")) == 5 * 3
    assert pd.read_csv(tab / "lmm_diagnostics.csv")["converged"].all()
    assert len(pd.read_csv(tab / "survival_counts.csv")) == 4
