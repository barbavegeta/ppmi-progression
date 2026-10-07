"""Piecewise time trend and covariate rules for the mixed models."""
import numpy as np
import pandas as pd

import config as cfg
import models_lmm as ml


def test_time_basis():
    d = ml.add_time_basis(pd.DataFrame({"years": [0, 1, 2, 3.5, 6]}))
    assert d["t2"].tolist() == [0, 0, 0, 1.5, 4]
    assert d["t5"].tolist() == [0, 0, 0, 0, 1]


def test_segment_contrasts_turn_coefficients_into_slopes():
    beta = pd.Series({"Intercept": 10, "years": 1.0, "t2": 0.5, "t5": -1.0})
    slopes = {lab: float(L @ beta) for lab, L in ml.segment_contrasts(beta.index)}
    assert slopes == {"0-2y": 1.0, "2-5y": 1.5, ">=5y": 0.5}


def test_piecewise_model_recovers_simulated_slopes():
    rng = np.random.default_rng(3)
    true = {"0-2y": 2.0, "2-5y": 1.0, ">=5y": 0.3}
    rows = []
    for pid in range(400):
        u0, u1 = rng.normal(0, 3), rng.normal(0, 0.4)
        for yr in range(9):
            mean = 20 + 2.0 * min(yr, 2) + 1.0 * np.clip(yr - 2, 0, 3) + 0.3 * max(yr - 5, 0)
            rows.append({"PATNO": pid, "years": float(yr),
                         "y": mean + u0 + u1 * yr + rng.normal(0, 2)})
    d = ml.add_time_basis(pd.DataFrame(rows))
    res, _ = ml.fit(d, ml.piecewise("y ~ years"))
    est = {lab: float(L @ res.fe_params) for lab, L in ml.segment_contrasts(res.fe_params.index)}
    for k, v in true.items():
        assert abs(est[k] - v) < 0.2, (k, est[k])


def test_genetic_covariate_needs_both_groups(monkeypatch):
    monkeypatch.setattr(cfg, "MIN_GENETIC_GROUP", 50)
    mk = lambda n0, n1: pd.DataFrame({"PATNO": range(n0 + n1), "genetic": [0] * n0 + [1] * n1})
    assert ml.use_genetic(mk(1433, 356))
    assert not ml.use_genetic(mk(1146, 25))
