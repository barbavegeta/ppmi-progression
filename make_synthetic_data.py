"""Generate FAKE data in the PPMI file layout, for testing the pipeline.

Nothing here comes from PPMI: values are simulated from made-up parameters
and only the file names and column names mimic the public data dictionary.

    python make_synthetic_data.py [--out data/synthetic] [--n 400]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

VISITS = [("SC", -1), ("BL", 0), ("V04", 12), ("V06", 24), ("V08", 36),
          ("V10", 48), ("V12", 60), ("V13", 72), ("V14", 84)]
SUFFIX = "_01Jan2000.csv"


def mmYYYY(d):
    return d.strftime("%m/%Y")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)

    n_pd, n_hc = a.n, a.n // 4
    ids = np.arange(3000, 3000 + n_pd + n_hc).astype(str)
    cohort = np.r_[np.ones(n_pd, int), np.full(n_hc, 2)]
    status = pd.DataFrame({"PATNO": ids, "COHORT": cohort,
                           "COHORT_DEFINITION": np.where(cohort == 1, "Parkinson's Disease", "Healthy Control"),
                           "ENROLL_STATUS": "Enrolled",
                           "ENRLLRRK2": rng.binomial(1, .1, len(ids)),
                           "ENRLGBA": rng.binomial(1, .08, len(ids))})
    sex = rng.binomial(1, .62, len(ids))
    demo = pd.DataFrame({"PATNO": ids, "SEX": sex})
    age0 = rng.normal(62, 9, len(ids)).clip(35, 85)
    bl_date = pd.to_datetime("2011-01-01") + pd.to_timedelta(rng.integers(0, 3000, len(ids)), "D")
    bl_date = bl_date.to_period("M").to_timestamp()
    dur = rng.gamma(1.2, 0.6, len(ids))
    dx = pd.DataFrame({"PATNO": ids[:n_pd],
                       "PDDXDT": [mmYYYY(d - pd.DateOffset(months=int(m * 12)))
                                  for d, m in zip(bl_date[:n_pd], dur[:n_pd])],
                       "SXDT": ""})

    rows = {k: [] for k in ["p1", "p1p", "p2", "p3", "moca", "cog", "age"]}
    ledd_rows = []
    for i, pid in enumerate(ids):
        pd_ = cohort[i] == 1
        last = rng.choice([12, 24, 36, 60, 84], p=[.1, .15, .2, .25, .3])
        u = rng.normal(0, 1, 4)
        slope3 = (2.5 + 0.06 * (age0[i] - 62) + 1.2 * u[1]) if pd_ else 0.1
        start_tx = rng.choice([6, 12, 24, 999], p=[.3, .3, .2, .2]) if pd_ else 999
        moca_at_sc = rng.random() < .7   # as in PPMI, baseline MoCA is often done at screening
        if start_tx < 999:
            ledd_rows.append({"PATNO": pid, "EVENT_ID": "LOG", "LEDTRT": "Carbidopa/Levodopa",
                              "LEDDSTRMG": 100, "LEDD": 300,
                              "STARTDT": mmYYYY(bl_date[i] + pd.DateOffset(months=int(start_tx))),
                              "STOPDT": ""})
            if rng.random() < .3:
                ledd_rows.append({"PATNO": pid, "EVENT_ID": "LOG", "LEDTRT": "Entacapone",
                                  "LEDDSTRMG": 200, "LEDD": "LD x 0.33",
                                  "STARTDT": mmYYYY(bl_date[i] + pd.DateOffset(months=int(start_tx) + 24)),
                                  "STOPDT": ""})
            if rng.random() < .4:
                ledd_rows.append({"PATNO": pid, "EVENT_ID": "LOG", "LEDTRT": "Pramipexole",
                                  "LEDDSTRMG": 1, "LEDD": 100,
                                  "STARTDT": mmYYYY(bl_date[i] + pd.DateOffset(months=int(start_tx))),
                                  "STOPDT": mmYYYY(bl_date[i] + pd.DateOffset(months=int(start_tx) + 30))})
        for ev, m in VISITS:
            if m > last:
                break
            if rng.random() < .07 and ev not in ("SC", "BL"):
                continue
            date = bl_date[i] + pd.DateOffset(months=max(m, 0))
            t = max(m, 0) / 12
            key = {"PATNO": pid, "EVENT_ID": ev, "INFODT": mmYYYY(date)}
            rows["age"].append({"PATNO": pid, "EVENT_ID": ev, "AGE_AT_VISIT": round(age0[i] + t, 1)})
            if ev == "SC":
                if moca_at_sc:
                    sc = 27.5 + u[3] + rng.normal(0, 1.2)
                    rows["moca"].append({**key, "MCATOT": int(np.clip(round(sc), 0, 30))})
                continue
            p1 = (6 + 1.0 * t + 2 * u[0] + rng.normal(0, 2)) if pd_ else rng.normal(3, 2)
            p2 = (5 + 1.4 * t + 3 * u[0] + rng.normal(0, 2.5)) if pd_ else rng.normal(1, 1)
            p3 = (20 + slope3 * t + 7 * u[2] + rng.normal(0, 4)) if pd_ else rng.normal(1, 2)
            p1r = np.clip(round(p1 * .4), 0, 24)
            rows["p1"].append({**key, "NP1RTOT": p1r})
            rows["p1p"].append({**key, "NP1PTOT": np.clip(round(p1 * .6), 0, 28)})
            rows["p2"].append({**key, "NP2PTOT": np.clip(round(p2), 0, 52)})
            hy = 3 if p3 > 42 else 2 if p3 > 14 else 1 if pd_ else 0
            treated = m >= start_tx
            off = {**key, "NP3TOT": np.clip(round(p3), 0, 132), "NHY": hy,
                   "PDSTATE": "OFF" if treated else "", "PDTRTMNT": int(treated), "PDMEDYN": int(treated)}
            if rng.random() < .01:
                off["NHY"] = 101
            rows["p3"].append(off)
            if treated:
                rows["p3"].append({**off, "NP3TOT": np.clip(round(p3 * .7), 0, 132), "PDSTATE": "ON"})
            moca = 27.5 - (0.25 + 0.02 * (age0[i] - 62)) * t * (pd_ + .3) + u[3] + rng.normal(0, 1.2)
            moca = int(np.clip(round(moca), 0, 30))
            if not (ev == "BL" and moca_at_sc):
                rows["moca"].append({**key, "MCATOT": moca})
            if pd_:
                rows["cog"].append({**key, "COGSTATE": 1 if moca >= 25 else 2 if moca >= 20 else 3})

    write = lambda df, name: df.to_csv(a.out / f"{name}{SUFFIX}", index=False)
    write(pd.DataFrame(rows["p1"]), "MDS-UPDRS_Part_I")
    write(pd.DataFrame(rows["p1p"]), "MDS-UPDRS_Part_I_Patient_Questionnaire")
    write(pd.DataFrame(rows["p2"]), "MDS_UPDRS_Part_II__Patient_Questionnaire")
    write(pd.DataFrame(rows["p3"]), "MDS-UPDRS_Part_III")
    write(pd.DataFrame(rows["moca"]), "Montreal_Cognitive_Assessment__MoCA_")
    write(pd.DataFrame(rows["cog"]), "Cognitive_Categorization")
    write(pd.DataFrame(rows["age"]), "Age_at_visit")
    write(status, "Participant_Status")
    write(demo, "Demographics")
    write(dx, "PD_Diagnosis_History")
    write(pd.DataFrame(ledd_rows), "LEDD_Concomitant_Medication_Log")
    print(f"Synthetic data written to {a.out}")


if __name__ == "__main__":
    main()
