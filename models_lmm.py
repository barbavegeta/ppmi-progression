"""Linear mixed-effects models of MDS-UPDRS and MoCA trajectories.

Model (per outcome), random intercept and slope per participant:

    y_ij = b0 + b1*years + covariates + years x baseline covariates
           + time-varying LEDD (+ ON-state for Part III) + u0_i + u1_i*years + e_ij

A second, piecewise-linear model adds (years - k)+ for each knot in
cfg.TIME_KNOTS, so the mean slope can change at 2 and 5 years (scores in early
PD rise slowly while treatment starts, then faster, then level off). The two
are compared with a likelihood-ratio test on ML fits; the linear slope is kept
as a one-number summary.

Fixed effects are reported with 95% CIs. A pattern-mixture sensitivity
analysis lets the slope differ by length of follow-up, to show how much the
MAR assumption behind the ML fit matters for dropout.
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from patsy import dmatrix

import config as cfg

OUTCOMES = {
    "updrs1": "MDS-UPDRS Part I",
    "updrs2": "MDS-UPDRS Part II",
    "updrs3": "MDS-UPDRS Part III",
    "updrs_total": "MDS-UPDRS I-III total",
    "moca": "MoCA",
}


def prepare(long: pd.DataFrame, outcome: str) -> pd.DataFrame:
    d = long.dropna(subset=[outcome, "years", "age_bl", "male", "dur_bl"]).copy()
    # Centre baseline covariates so the `years` coefficient is the annual
    # change for a participant of average age and disease duration.
    d["age_c"] = (d["age_bl"] - d["age_bl"].mean()) / 10          # per decade
    d["dur_c"] = d["dur_bl"] - d["dur_bl"].mean()                   # per year
    d["ledd100"] = d["ledd"] / 100
    d["y"] = d[outcome]
    add_time_basis(d)
    fu = d.groupby("PATNO")["years"].transform("max")
    d["fu_group"] = pd.cut(fu, cfg.FOLLOWUP_BINS, labels=cfg.FOLLOWUP_LABELS, right=False)
    return d


def add_time_basis(d: pd.DataFrame) -> pd.DataFrame:
    """Add (years - k)+ columns t<k> for the piecewise-linear time trend."""
    for k in cfg.TIME_KNOTS:
        d[f"t{k}"] = np.clip(d["years"] - k, 0, None)
    return d


def piecewise(f: str) -> str:
    return f + "".join(f" + t{k}" for k in cfg.TIME_KNOTS)


def segment_contrasts(index) -> list:
    """(label, L) pairs: L @ beta is the mean annual change in each segment."""
    edges = [0] + list(cfg.TIME_KNOTS)
    out = []
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else None
        L = pd.Series(0.0, index=index)
        L["years"] = 1
        for k in cfg.TIME_KNOTS[:i]:
            L[f"t{k}"] = 1
        out.append((f"{lo}-{hi}y" if hi is not None else f">={lo}y", L))
    return out


def use_genetic(d: pd.DataFrame) -> bool:
    """Enough participants in both the genetic and sporadic groups."""
    g = d.drop_duplicates("PATNO")["genetic"]
    return g.notna().all() and g.value_counts().reindex([0, 1]).fillna(0).min() >= cfg.MIN_GENETIC_GROUP


def formula(outcome: str, d: pd.DataFrame) -> str:
    def varies(col, min_n=cfg.MIN_CELL):
        x = d[col]
        return x.notna().all() and x.nunique() > 1 and x.value_counts().min() >= min_n

    f = "y ~ years * (age_c + male + dur_c)"
    if varies("ledd100", 1):
        f += " + ledd100"
    if outcome in ("updrs3", "updrs_total") and varies("on_state"):
        f += " + on_state"
    if varies("genetic") and use_genetic(d):
        f += " + years * genetic"
    return f


def fit(d: pd.DataFrame, f: str, reml: bool = True):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = smf.mixedlm(f, d, groups=d["PATNO"], re_formula="~years")
        res, err = None, None
        for method in ("lbfgs", "powell", "nm"):
            try:
                res = model.fit(reml=reml, method=method)
                res.optimizer = method
                if res.converged:
                    break
            except (np.linalg.LinAlgError, ValueError) as e:
                err = e
    if res is None:
        raise RuntimeError(f"Mixed model failed to fit: {f} ({err})")
    notes = sorted({str(x.message).split("\n")[0] for x in w})
    return res, notes


def fixed_effects(res) -> pd.DataFrame:
    fe = res.fe_params.index
    ci = res.conf_int().loc[fe]
    return pd.DataFrame({
        "term": fe,
        "estimate": res.fe_params.values,
        "se": res.bse.loc[fe].values,
        "ci_low": ci[0].values,
        "ci_high": ci[1].values,
        "p": res.pvalues.loc[fe].values,
    })


def predict_mean(res, d: pd.DataFrame, years: np.ndarray) -> pd.DataFrame:
    """Population-average trajectory at mean covariates, with 95% CI."""
    new = pd.DataFrame({
        "years": years, "age_c": 0.0, "male": d["male"].mean(), "dur_c": 0.0,
        "ledd100": 0.0, "on_state": 0, "genetic": 0, "y": 0.0,
    })
    add_time_basis(new)
    X = dmatrix(res.model.formula.split("~", 1)[1], new, return_type="dataframe")
    X = X[list(res.fe_params.index)].values
    beta = res.fe_params.values
    V = res.cov_params().loc[res.fe_params.index, res.fe_params.index].values
    est = X @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", X, V, X))
    return pd.DataFrame({"years": years, "pred": est,
                         "lo": est - 1.96 * se, "hi": est + 1.96 * se})


def fitted_means(res, d: pd.DataFrame) -> pd.DataFrame:
    """Mean fixed-effect prediction per yearly bin over the visits actually
    observed, i.e. the model's counterpart of observed_means()."""
    X = dmatrix(res.model.formula.split("~", 1)[1], d, return_type="dataframe")
    fit_ = pd.Series(X[list(res.fe_params.index)].values @ res.fe_params.values, index=X.index)
    b = pd.DataFrame({"yr": d.loc[X.index, "years"].round(), "fit": fit_})
    b = b.groupby("yr")["fit"].agg(["mean", "count"])
    return b[b["count"] >= cfg.MIN_CELL].reset_index()


def observed_means(d: pd.DataFrame) -> pd.DataFrame:
    """Observed mean (95% CI) per yearly bin; bins with < MIN_CELL dropped."""
    b = d.assign(yr=d["years"].round()).groupby("yr")["y"].agg(["mean", "std", "count"])
    b = b[b["count"] >= cfg.MIN_CELL]
    b["half"] = 1.96 * b["std"] / np.sqrt(b["count"])
    return b.reset_index()


def pattern_mixture(d: pd.DataFrame, f: str) -> pd.DataFrame:
    """Annual slope by follow-up-length group (dropout sensitivity)."""
    d = d.dropna(subset=["fu_group"]).copy()
    d["fu_group"] = d["fu_group"].astype(str)
    groups = [g for g in cfg.FOLLOWUP_LABELS if g in set(d["fu_group"])]
    if len(groups) < 2:
        return pd.DataFrame()
    ref = groups[-1]
    f2 = f + f" + years * C(fu_group, Treatment('{ref}'))"
    res, _ = fit(d, f2)
    fe = res.fe_params
    V = res.cov_params().loc[fe.index, fe.index]
    rows = []
    for g in groups:
        L = pd.Series(0.0, index=fe.index)
        L["years"] = 1
        term = f"years:C(fu_group, Treatment('{ref}'))[T.{g}]"
        if term in L.index:
            L[term] = 1
        est = float(L @ fe)
        se = float(np.sqrt(L @ V @ L))
        rows.append({"followup_group": g,
                     "participants": d.loc[d["fu_group"] == g, "PATNO"].nunique(),
                     "annual_change": est, "ci_low": est - 1.96 * se, "ci_high": est + 1.96 * se})
    return pd.DataFrame(rows)


def run(long: pd.DataFrame, out_dir):
    tab, fig_dir = out_dir / "tables", out_dir / "figures"
    slopes, segs, pm_all, diag = [], [], [], []
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes = axes.ravel()

    for k, (outcome, label) in enumerate(OUTCOMES.items()):
        d = prepare(long, outcome)
        n_id = d["PATNO"].nunique()
        if n_id < 20:
            print(f"  {label}: only {n_id} participants, skipped")
            continue
        f = formula(outcome, d)
        res, notes = fit(d, f)
        fe = fixed_effects(res)
        fe.to_csv(tab / f"lmm_{outcome}_fixed_effects.csv", index=False)

        yr = fe.set_index("term").loc["years"]
        re_cov = res.cov_re
        slopes.append({
            "outcome": label, "participants": n_id, "observations": len(d),
            "annual_change": yr["estimate"], "ci_low": yr["ci_low"],
            "ci_high": yr["ci_high"], "p": yr["p"],
            "sd_random_intercept": np.sqrt(re_cov.iloc[0, 0]),
            "sd_random_slope": np.sqrt(re_cov.iloc[1, 1]),
            "residual_sd": np.sqrt(res.scale),
        })
        # Piecewise-linear time; LR test against linear on ML fits.
        f_pw = piecewise(f)
        res_pw, notes_pw = fit(d, f_pw)
        fixed_effects(res_pw).to_csv(tab / f"lmm_{outcome}_piecewise_fixed_effects.csv", index=False)
        ml_lin, _ = fit(d, f, reml=False)
        ml_pw, _ = fit(d, f_pw, reml=False)
        lr = 2 * (ml_pw.llf - ml_lin.llf)
        lr_p = stats.chi2.sf(lr, len(cfg.TIME_KNOTS))
        fe_pw = res_pw.fe_params
        V_pw = res_pw.cov_params().loc[fe_pw.index, fe_pw.index]
        for seg, L in segment_contrasts(fe_pw.index):
            est, se = float(L @ fe_pw), float(np.sqrt(L @ V_pw @ L))
            segs.append({"outcome": label, "segment": seg, "annual_change": est,
                         "ci_low": est - 1.96 * se, "ci_high": est + 1.96 * se,
                         "p": 2 * stats.norm.sf(abs(est / se))})

        diag.append({"outcome": label, "formula": f, "converged": res.converged,
                     "optimizer": res.optimizer,
                     "piecewise_converged": res_pw.converged,
                     "aic_linear_ml": ml_lin.aic, "aic_piecewise_ml": ml_pw.aic,
                     "lr_chi2_piecewise_vs_linear": lr, "lr_df": len(cfg.TIME_KNOTS),
                     "lr_p": lr_p,
                     "warnings_incl_failed_attempts": " | ".join(notes + notes_pw)})

        pm = pattern_mixture(d, f)
        if len(pm):
            pm.insert(0, "outcome", label)
            pm_all.append(pm)

        ax = axes[k]
        grid = np.linspace(0, np.floor(d["years"].quantile(0.95)), 50)
        p = predict_mean(res_pw, d, grid)
        fm = fitted_means(res_pw, d)
        ob = observed_means(d)
        ax.fill_between(p["years"], p["lo"], p["hi"], alpha=0.2, color="#2b6cb0")
        ax.plot(p["years"], p["pred"], "--", color="#2b6cb0",
                label="Model, reference participant*")
        ax.plot(fm["yr"], fm["mean"], "-", color="#c05621", lw=1.5,
                label="Model, averaged over observed visits")
        for k in cfg.TIME_KNOTS:
            ax.axvline(k, color="#bbb", lw=0.6, zorder=0)
        ax.errorbar(ob["yr"], ob["mean"], yerr=ob["half"], fmt="o", ms=4,
                    color="#444", capsize=2, label="Observed mean ± 95% CI")
        ax.set_title(label)
        ax.set_xlabel("Years since baseline")
        print(f"  {label}: {yr['estimate']:+.2f}/yr "
              f"[{yr['ci_low']:+.2f}, {yr['ci_high']:+.2f}], n={n_id}"
              + ("" if res.converged else "  (NOT CONVERGED)"))
        seg_txt = ", ".join(f"{r['segment']} {r['annual_change']:+.2f}"
                            for r in segs if r["outcome"] == label)
        print(f"    piecewise: {seg_txt}  (LR p={lr_p:.2g})")

    for ax in axes[len(OUTCOMES):]:
        ax.axis("off")
    axes[0].legend(fontsize=7)
    fig.text(0.01, 0.005, "*Reference participant: mean age, sex mix and disease duration; "
             "LEDD 0, OFF/untreated, non-genetic. Grey lines: knots of the piecewise time trend.",
             fontsize=7, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(fig_dir / "lmm_trajectories.png", dpi=200)
    plt.close(fig)

    pd.DataFrame(slopes).to_csv(tab / "lmm_annual_change.csv", index=False)
    pd.DataFrame(segs).to_csv(tab / "lmm_piecewise_change.csv", index=False)
    pd.DataFrame(diag).to_csv(tab / "lmm_diagnostics.csv", index=False)
    if pm_all:
        pd.concat(pm_all).to_csv(tab / "lmm_sensitivity_dropout_pattern.csv", index=False)
