"""Cleaning helpers and LEDD computation."""
import numpy as np
import pandas as pd

import load


def test_parse_month_handles_ppmi_and_full_dates():
    out = load.parse_month(pd.Series(["03/2020", "2014-03-15", "", "nan"]))
    assert out.iloc[0] == pd.Timestamp("2020-03-01")
    assert out.iloc[1] == pd.Timestamp("2014-03-01")
    assert out.iloc[2:].isna().all()


def test_clean_score_drops_out_of_range_codes():
    df = pd.DataFrame({"MCATOT": [30, 0, 101, -1, "x"]})
    out = load.clean_score(df, "MCATOT")
    assert out.tolist()[:2] == [30, 0]
    assert out.iloc[2:].isna().all()


def _log(rows):
    return pd.DataFrame(rows, columns=["PATNO", "LEDTRT", "STARTDT", "STOPDT", "LEDD"])


def _visit(ids, month="2020-03-01"):
    return pd.DataFrame({"PATNO": ids, "EVENT_ID": "V06", "date": pd.Timestamp(month)})


def test_ledd_dose_change_month_counts_new_dose_only():
    log = _log([("1", "Sinemet", "01/2019", "03/2020", "300"),
                ("1", "Sinemet", "03/2020", "", "600")])
    assert load.compute_ledd(log, _visit(["1"])).iloc[0] == 600


def test_ledd_exact_duplicates_counted_once():
    log = _log([("2", "Ropinirole", "01/2019", "", "200")] * 2)
    assert load.compute_ledd(log, _visit(["2"])).iloc[0] == 200


def test_ledd_concurrent_rows_and_comt_multiplier():
    # IR + CR levodopa both active (500 mg); entacapone adds 0.33 x levodopa.
    log = _log([("3", "Sinemet", "01/2019", "", "300"),
                ("3", "Sinemet", "06/2019", "", "200"),
                ("3", "Entacapone", "06/2019", "", "LD x 0.33")])
    assert np.isclose(load.compute_ledd(log, _visit(["3"])).iloc[0], 665)


def test_ledd_drug_stopping_in_visit_month_still_counts():
    log = _log([("4", "Rasagiline", "01/2019", "03/2020", "150")])
    assert load.compute_ledd(log, _visit(["4"])).iloc[0] == 150


def test_ledd_zero_without_active_rows():
    log = _log([("5", "Sinemet", "01/2021", "", "300")])      # starts after the visit
    assert load.compute_ledd(log, _visit(["5", "6"])).tolist() == [0, 0]
