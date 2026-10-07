"""Milestone definitions, counting-process split, KM horizon, time-split Cox."""
import numpy as np
import pandas as pd

import config as cfg
import models_survival as ms


def _moca_long(scores: dict) -> pd.DataFrame:
    return pd.DataFrame([dict(PATNO=p, years=float(i), moca=v, age_bl=60, male=1, dur_bl=1,
                              updrs3_bl=20, moca_bl=27, genetic=0)
                         for p, vs in scores.items() for i, v in enumerate(vs)])


SCORES = {1: [27, 25, 27, 28],   # one-off dip
          2: [27, 25, 24, 23],   # sustained from year 1
          3: [27, 27, 27, 25],   # drop at the last visit
          4: [25, 24, 27, 27],   # below the cut-off at baseline
          5: [27, 25, 27, 25]}   # dip, recovery, last-visit drop


def _events(confirm):
    ev = ms.event_data(_moca_long(SCORES), "moca", lambda x: x < 26, confirm=confirm)
    return ev.set_index("PATNO")[["time", "event", "time_mid"]]


def test_single_visit_definition():
    ev = _events(False)
    assert 4 not in ev.index                       # excluded at baseline
    assert ev.loc[[1, 2, 5], "time"].tolist() == [1, 1, 1]
    assert ev["event"].eq(1).all()


def test_confirmed_definition(monkeypatch):
    monkeypatch.setattr(cfg, "MOCA_CONFIRM_LAST_VISIT", True)
    ev = _events(True)
    assert 4 not in ev.index                       # baseline uses the raw score
    assert ev.loc[1].tolist() == [3, 0, 3]         # dip not confirmed: censored
    assert ev.loc[2].tolist() == [1, 1, 0.5]
    assert ev.loc[3].tolist() == [3, 1, 2.5]
    assert ev.loc[5].tolist() == [3, 1, 2.5]       # midpoint uses the last visit before


def test_confirmed_last_visit_can_be_censored(monkeypatch):
    monkeypatch.setattr(cfg, "MOCA_CONFIRM_LAST_VISIT", False)
    ev = _events(True)
    assert ev.loc[[3, 5], "event"].tolist() == [0, 0]


def test_split_at():
    d = pd.DataFrame({"time": [5.0, 2.0], "event": [1, 1], "x": [1, 0]})
    sp = ms.split_at(d, "time", 3).sort_values(["id", "start"])
    assert sp[["id", "start", "stop", "event", "late"]].values.tolist() == [
        [0, 0.0, 3.0, 0, 0], [0, 3.0, 5.0, 1, 1], [1, 0.0, 2.0, 1, 0]]


def test_km_horizon(monkeypatch):
    monkeypatch.setattr(cfg, "KM_MIN_AT_RISK", 3)
    t = pd.Series([1, 2, 3, 4, 5, 6], dtype=float)
    assert ms.km_horizon(t) == 3                   # 3 participants remain after t = 3


def test_time_split_cox_recovers_changing_hazard_ratio(monkeypatch):
    """Simulate HR 2 before year 3 and HR 1 after; the split model should see both."""
    monkeypatch.setattr(cfg, "COX_TIME_SPLIT", 3)
    rng = np.random.default_rng(7)
    n, base = 4000, 0.15
    x = rng.binomial(1, 0.5, n)
    t_early = rng.exponential(1 / (base * np.where(x == 1, 2.0, 1.0)))
    t = np.where(t_early < 3, t_early, 3 + rng.exponential(1 / base, n))
    censor = 8.0
    d = pd.DataFrame({"time": np.minimum(t, censor), "event": (t < censor).astype(int), "x": x})
    out = ms.cox_time_split(d, "time", ["x"]).set_index("term")
    early, late = out.loc["x, 0-3 y"], out.loc["x, >3 y"]
    assert early["HR_low"] < 2 < early["HR_high"]
    assert late["HR_low"] < 1 < late["HR_high"]
    assert late["p_equal_HR"] < 0.001
