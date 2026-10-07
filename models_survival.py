"""Kaplan-Meier and Cox models for time to clinical milestones.

Milestones (first scheduled visit at which it is observed):
  * hy3      Hoehn & Yahr stage >= 3
  * cogimp   cognitive impairment (site cognitive categorisation MCI/dementia)
  * moca26   MoCA < 26 (secondary, test-based definition)
  * moca26c  MoCA < 26 confirmed at the next assessed visit, which removes
             single-visit dips (test-retest noise near the cut-off)

Participants who already meet the milestone at baseline are excluded.
Follow-up is censored at the last visit at which the milestone was assessed.
Because milestones are only seen at visits, the event time is interval
censored; the primary analysis uses the visit time and a sensitivity
analysis uses the midpoint between the last negative and first positive visit.

Covariates that fail the proportional-hazards test (p < cfg.PH_ALPHA) are
refitted with a step-function effect: one hazard ratio up to
cfg.COX_TIME_SPLIT years and another after, from a counting-process Cox model,
with a Wald test that the two are equal.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, CoxTimeVaryingFitter, KaplanMeierFitter
from scipy import stats
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test, proportional_hazard_test

import config as cfg
import models_lmm

MILESTONES = {
    "hy3": ("Hoehn & Yahr >= 3", "nhy", lambda x: x >= cfg.HY_EVENT),
    "cogimp": ("Cognitive impairment (MCI/dementia)", "cogstate", lambda x: x.isin(cfg.COG_EVENT_STATES)),
    "moca26": ("MoCA < 26", "moca", lambda x: x < cfg.MOCA_CUTOFF),
    "moca26c": ("MoCA < 26, confirmed", "moca", lambda x: x < cfg.MOCA_CUTOFF),
}
CONFIRMED = {"moca26c"}

COVARIATES = {
    "hy3": ["age10", "male", "dur_bl", "updrs3_bl10"],
    "cogimp": ["age10", "male", "dur_bl", "updrs3_bl10", "moca_bl"],
    "moca26": ["age10", "male", "dur_bl", "updrs3_bl10", "moca_bl"],
    "moca26c": ["age10", "male", "dur_bl", "updrs3_bl10", "moca_bl"],
}

LABELS = {"age10": "Age at baseline (per 10 y)", "male": "Male sex",
          "dur_bl": "Disease duration at baseline (per y)",
          "updrs3_bl10": "Baseline MDS-UPDRS III (per 10 points)",
          "moca_bl": "Baseline MoCA (per point)", "genetic": "Genetic enrolment"}


def event_data(long: pd.DataFrame, col: str, is_event, confirm: bool = False) -> pd.DataFrame:
    d = long.dropna(subset=[col])[["PATNO", "years", col]].sort_values(["PATNO", "years"])
    d["pos"] = d["raw_pos"] = is_event(d[col])
    if confirm:
        # A positive visit counts only if the next assessed visit is positive too
        # (or, optionally, if it is the last visit). Event time = first visit of the run.
        nxt = d.groupby("PATNO")["pos"].shift(-1, fill_value=cfg.MOCA_CONFIRM_LAST_VISIT)
        d["pos"] = d["pos"] & nxt.astype(bool)
    rows = []
    for pid, g in d.groupby("PATNO", sort=False):
        base = g[g["years"] == 0]
        if base.empty or base["raw_pos"].iloc[0]:
            continue                       # no baseline, or already reached
        fu = g[g["years"] > 0]
        if fu.empty:
            continue
        hit = fu[fu["pos"]]
        if len(hit):
            t = hit["years"].iloc[0]
            prev = g.loc[g["years"] < t, "years"].max()   # last visit before event
            rows.append((pid, t, 1, (prev + t) / 2))
        else:
            t = fu["years"].iloc[-1]
            rows.append((pid, t, 0, t))
    ev = pd.DataFrame(rows, columns=["PATNO", "time", "event", "time_mid"])
    base = long.drop_duplicates("PATNO")[["PATNO", "age_bl", "male", "dur_bl",
                                          "updrs3_bl", "moca_bl", "genetic"]]
    ev = ev.merge(base, on="PATNO", how="left")
    ev["age10"] = ev["age_bl"] / 10
    ev["updrs3_bl10"] = ev["updrs3_bl"] / 10
    ev["age_group"] = np.where(ev["age_bl"] >= 65, ">=65", "<65")
    return ev


def separated(d: pd.DataFrame, covs: list) -> list:
    """Binary covariates with a level that has no events (HR not estimable)."""
    return [c for c in covs if d[c].nunique() == 2
            and d.groupby(c)["event"].sum().min() == 0]


def cox(ev: pd.DataFrame, covs: list, time_col: str):
    d = ev.dropna(subset=covs)[[time_col, "event"] + covs]
    drop = separated(d, covs)
    if drop:
        print(f"    dropped {drop}: no events in one level")
        d = d.drop(columns=drop)
    cph = CoxPHFitter()
    cph.fit(d, duration_col=time_col, event_col="event")
    s = cph.summary.reset_index().rename(columns={"covariate": "term"})
    s = s.rename(columns={"exp(coef)": "HR", "exp(coef) lower 95%": "HR_low",
                          "exp(coef) upper 95%": "HR_high"})
    s["term"] = s["term"].map(LABELS).fillna(s["term"])
    return cph, d, s[["term", "HR", "HR_low", "HR_high", "p"]]


def split_at(d: pd.DataFrame, time_col: str, cut: float) -> pd.DataFrame:
    """Counting-process rows: follow-up beyond `cut` becomes a second row
    (start=cut) with late=1, so effects can differ before and after `cut`."""
    d = d.rename(columns={time_col: "stop"}).assign(id=d.index, start=0.0, late=0)
    early = d.copy()
    long_fu = early["stop"] > cut
    early.loc[long_fu, "stop"] = cut
    early.loc[long_fu, "event"] = 0
    late = d[long_fu].assign(start=float(cut), late=1)
    return pd.concat([early, late], ignore_index=True)


def cox_time_split(d: pd.DataFrame, time_col: str, nonph: list) -> pd.DataFrame:
    """Cox model with HRs for `nonph` covariates split at cfg.COX_TIME_SPLIT."""
    cut = cfg.COX_TIME_SPLIT
    sp = split_at(d, time_col, cut)
    for c in nonph:
        sp[f"{c}__early"] = sp[c] * (1 - sp["late"])
        sp[f"{c}__late"] = sp[c] * sp["late"]
    sp = sp.drop(columns=nonph + ["late"])
    ctv = CoxTimeVaryingFitter()
    ctv.fit(sp, id_col="id", start_col="start", stop_col="stop", event_col="event")
    s = ctv.summary
    names = ctv.params_.index           # variance_matrix_ comes back unlabelled
    V = pd.DataFrame(np.asarray(ctv.variance_matrix_), index=names, columns=names)
    rows = []
    for term, r in s.iterrows():
        base, _, period = term.partition("__")
        label = LABELS.get(base, base)
        if period:
            label += f", {'0-' + str(cut) if period == 'early' else '>' + str(cut)} y"
        row = {"term": label, "HR": r["exp(coef)"], "HR_low": r["exp(coef) lower 95%"],
               "HR_high": r["exp(coef) upper 95%"], "p": r["p"], "p_equal_HR": np.nan}
        if period == "late":
            e, l = f"{base}__early", term
            diff = s.loc[l, "coef"] - s.loc[e, "coef"]
            se = np.sqrt(V.loc[l, l] + V.loc[e, e] - 2 * V.loc[l, e])
            row["p_equal_HR"] = 2 * stats.norm.sf(abs(diff / se))
        rows.append(row)
    return pd.DataFrame(rows)


def km_horizon(times: pd.Series) -> float:
    """Latest time after which at least KM_MIN_AT_RISK participants remain at
    risk (same convention as the lifelines at-risk table under the plot)."""
    t = np.sort(times.unique())
    ok = [x for x in t if (times > x).sum() >= cfg.KM_MIN_AT_RISK]
    return ok[-1] if ok else 0.0


def plot_km(ax, kmf, horizon: float, label: str):
    """Step curve with 95% CI band up to `horizon`, last step extended to it
    (lifelines' own plot leaves a zero-width sliver when it ends on a drop)."""
    sf = kmf.survival_function_.iloc[:, 0]
    ci = kmf.confidence_interval_survival_function_
    t = np.append(sf.index[sf.index <= horizon], horizon)
    pick = lambda x: np.append(x[sf.index <= horizon], x[sf.index <= horizon][-1])
    line, = ax.step(t, pick(sf.values), where="post", label=label)
    ax.fill_between(t, pick(ci.iloc[:, 0].values), pick(ci.iloc[:, 1].values),
                    step="post", alpha=0.25, color=line.get_color(), lw=0)


def km_table(ev: pd.DataFrame, label: str, max_year: int) -> pd.DataFrame:
    rows = []
    for grp, g in [("All", ev)] + list(ev.groupby("age_group")):
        kmf = KaplanMeierFitter().fit(g["time"], g["event"])
        ci = kmf.confidence_interval_survival_function_
        for yr in range(0, max_year + 1):
            at_risk = int((g["time"] >= yr).sum())
            if at_risk < cfg.KM_MIN_AT_RISK:
                break
            rows.append({"milestone": label, "group": grp, "year": yr,
                         "at_risk": at_risk,
                         "event_free": float(kmf.predict(yr)),
                         "ci_low": float(ci.iloc[:, 0].asof(yr)),
                         "ci_high": float(ci.iloc[:, 1].asof(yr))})
    return pd.DataFrame(rows)


def run(long: pd.DataFrame, out_dir):
    tab, fig_dir = out_dir / "tables", out_dir / "figures"
    use_genetic = models_lmm.use_genetic(long)
    hr_all, sens_all, ph_all, split_all, km_all, counts = [], [], [], [], [], []
    fig, axes = plt.subplots(1, len(MILESTONES), figsize=(5 * len(MILESTONES), 5.6))

    for ax, (key, (label, col, is_event)) in zip(axes, MILESTONES.items()):
        ev = event_data(long, col, is_event, confirm=key in CONFIRMED)
        if ev["event"].sum() < 10:
            print(f"  {label}: {int(ev['event'].sum())} events, skipped")
            ax.axis("off")
            continue
        covs = COVARIATES[key] + (["genetic"] if use_genetic else [])
        counts.append({"milestone": label, "participants": len(ev),
                       "events": int(ev["event"].sum()),
                       "median_followup_years": ev["time"].median()})

        cph, d, s = cox(ev, covs, "time")
        counts[-1].update({"cox_participants": len(d), "cox_events": int(d["event"].sum()),
                           "cox_covariates": "; ".join(LABELS.get(c, c) for c in d.columns[2:])})
        s.insert(0, "milestone", label)
        hr_all.append(s)

        ph = proportional_hazard_test(cph, d, time_transform="rank").summary.reset_index()
        ph = ph.rename(columns={"index": "term"})
        ph["term"] = ph["term"].map(LABELS).fillna(ph["term"])
        ph.insert(0, "milestone", label)
        ph_all.append(ph[["milestone", "term", "test_statistic", "p"]])

        raw = proportional_hazard_test(cph, d, time_transform="rank").summary["p"]
        nonph = [c for c in raw.index if raw[c] < cfg.PH_ALPHA]
        if nonph:
            sp = cox_time_split(d, "time", nonph)
            sp.insert(0, "milestone", label)
            split_all.append(sp)

        _, _, s_mid = cox(ev, covs, "time_mid")
        s_mid.insert(0, "milestone", label)
        sens_all.append(s_mid)

        km_all.append(km_table(ev, label, int(np.ceil(ev["time"].max()))))
        fitters = []
        for grp, g in ev.groupby("age_group"):
            kmf = KaplanMeierFitter().fit(g["time"], g["event"], label=f"Age {grp}")
            plot_km(ax, kmf, km_horizon(g["time"]), f"Age {grp}")
            fitters.append(kmf)
        add_at_risk_counts(*fitters, ax=ax, rows_to_show=["At risk"], fontsize=7)
        a, b = ev[ev["age_group"] == "<65"], ev[ev["age_group"] == ">=65"]
        p_lr = logrank_test(a["time"], b["time"], a["event"], b["event"]).p_value
        ax.set_title(f"{label}\nlog-rank p = {p_lr:.3g}", fontsize=10)
        ax.set_xlabel("Years since baseline")
        ax.set_ylabel("Proportion event-free")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8)
        print(f"  {label}: {counts[-1]['events']} events / {len(ev)} participants")
        if nonph:
            print(f"    non-PH, HR split at {cfg.COX_TIME_SPLIT} y: "
                  + ", ".join(LABELS.get(c, c) for c in nonph))

    fig.text(0.01, 0.005, f"Curves stop where fewer than {cfg.KM_MIN_AT_RISK} participants "
             "remain at risk.", fontsize=7, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(fig_dir / "km_milestones_by_age.png", dpi=200)
    plt.close(fig)

    if hr_all:
        pd.DataFrame(counts).to_csv(tab / "survival_counts.csv", index=False)
        pd.concat(hr_all).to_csv(tab / "cox_hazard_ratios.csv", index=False)
        pd.concat(sens_all).to_csv(tab / "cox_sensitivity_midpoint_time.csv", index=False)
        pd.concat(ph_all).to_csv(tab / "cox_ph_test.csv", index=False)
        if split_all:
            pd.concat(split_all).to_csv(tab / "cox_hazard_ratios_time_split.csv", index=False)
        pd.concat(km_all).to_csv(tab / "km_event_free_by_year.csv", index=False)
