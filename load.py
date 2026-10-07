"""Locate, read and clean the PPMI tables into analysis-ready data frames."""
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg


# ---------------------------------------------------------------- file lookup

def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower())


def find_files(data_dir: Path) -> dict:
    """Return {role: Path or None}. If several files match, the newest wins."""
    csvs = sorted(Path(data_dir).glob("*.csv"))
    found = {}
    for role, (inc, exc) in cfg.FILE_PATTERNS.items():
        hits = [p for p in csvs
                if re.search(inc, _normalise(p.name))
                and not (exc and re.search(exc, _normalise(p.name)))]
        if len(hits) > 1:
            hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            warnings.warn(f"{role}: {len(hits)} files match, using newest: {hits[0].name}")
        found[role] = hits[0] if hits else None
    return found


def missing_columns(role: str, columns) -> list:
    cols = set(columns)
    return [req for req in cfg.REQUIRED_COLUMNS[role]
            if not any(alt in cols for alt in req.split("|"))]


def read_table(path: Path, role: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    miss = missing_columns(role, df.columns)
    if miss:
        raise KeyError(f"{path.name} ({role}) is missing columns {miss}. "
                       "Run check_files.py and update config.REQUIRED_COLUMNS.")
    df["PATNO"] = df["PATNO"].astype(str).str.strip()
    if "EVENT_ID" in df:
        df["EVENT_ID"] = df["EVENT_ID"].astype(str).str.strip().str.upper()
    return df


# ---------------------------------------------------------------- helpers

def parse_month(s: pd.Series) -> pd.Series:
    """PPMI dates are MM/YYYY; return the first day of that month."""
    s = s.astype(str).str.strip()
    out = pd.to_datetime(s, format="%m/%Y", errors="coerce")
    # Fall back for any full dates (e.g. 2014-03-01) in older exports.
    rest = out.isna() & s.ne("nan") & s.ne("")
    if rest.any():
        out[rest] = pd.to_datetime(s[rest], errors="coerce").dt.to_period("M").dt.to_timestamp()
    return out


def months_between(a: pd.Series, b: pd.Series) -> pd.Series:
    return (b.dt.year - a.dt.year) * 12 + (b.dt.month - a.dt.month)


def clean_score(df: pd.DataFrame, col: str) -> pd.Series:
    x = pd.to_numeric(df[col], errors="coerce")
    lo, hi = cfg.VALID_RANGE[col]
    return x.where(x.between(lo, hi))


def scheduled(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["EVENT_ID"].str.match(cfg.VISIT_REGEX)]


def one_per_visit(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Keep the first complete-ish record per participant-visit."""
    df = df.assign(_n=df[cols].notna().sum(axis=1)).sort_values("_n", ascending=False)
    return df.drop_duplicates(cfg.KEY)[cfg.KEY + cols]


# ---------------------------------------------------------------- tables

def load_updrs(t: dict) -> pd.DataFrame:
    p1 = scheduled(t["updrs1"]).assign(p1r=lambda d: clean_score(d, "NP1RTOT"),
                                      date=lambda d: parse_month(d["INFODT"]))
    p1p = scheduled(t["updrs1p"]).assign(p1p=lambda d: clean_score(d, "NP1PTOT"))
    p2 = scheduled(t["updrs2"]).assign(updrs2=lambda d: clean_score(d, "NP2PTOT"),
                                      date=lambda d: parse_month(d["INFODT"]))
    p1 = one_per_visit(p1, ["p1r", "date"])
    p1p = one_per_visit(p1p, ["p1p"])
    p2 = one_per_visit(p2, ["updrs2", "date"])

    part1 = p1.merge(p1p, on=cfg.KEY, how="outer")
    part1["updrs1"] = part1["p1r"] + part1["p1p"]   # missing if either half is

    p3 = load_part3(t["updrs3"])
    out = part1[cfg.KEY + ["updrs1", "date"]].merge(
        p2.rename(columns={"date": "date2"}), on=cfg.KEY, how="outer").merge(
        p3.rename(columns={"date": "date3"}), on=cfg.KEY, how="outer")
    out["date"] = out["date"].fillna(out["date3"]).fillna(out["date2"])
    return out.drop(columns=["date2", "date3"])


def load_part3(df: pd.DataFrame) -> pd.DataFrame:
    """One Part III exam per visit, preferring OFF / untreated over ON.

    PPMI records separate OFF and ON exams for treated participants. The
    practically-defined OFF exam reflects disease severity with the least
    medication effect, so it is the primary measure; the medication state of
    the exam kept is returned as `on_state` for adjustment.
    """
    df = scheduled(df).copy()
    state = df["PDSTATE"].astype(str).str.strip().str.upper()
    treated = pd.Series(np.nan, index=df.index)
    for col in ("PDTRTMNT", "PDMEDYN"):
        if col in df:
            treated = treated.fillna(pd.to_numeric(df[col], errors="coerce"))
    df["med_state"] = np.select(
        [state.eq("OFF"), state.eq("ON"), treated.eq(0)],
        ["OFF", "ON", "untreated"], default="unknown")
    df["_rank"] = df["med_state"].map({"untreated": 0, "OFF": 0, "unknown": 1, "ON": 2})
    df["updrs3"] = clean_score(df, "NP3TOT")
    df["nhy"] = clean_score(df, "NHY")
    df["date"] = parse_month(df["INFODT"])
    df = df.sort_values(["PATNO", "EVENT_ID", "_rank", "updrs3"], na_position="last")
    df = df.drop_duplicates(cfg.KEY)
    df["on_state"] = df["med_state"].eq("ON").astype(int)
    return df[cfg.KEY + ["updrs3", "nhy", "med_state", "on_state", "date"]]


def load_moca(t: dict) -> pd.DataFrame:
    m = scheduled(t["moca"]).assign(moca=lambda d: clean_score(d, "MCATOT"),
                                    date_m=lambda d: parse_month(d["INFODT"]))
    return one_per_visit(m, ["moca", "date_m"])


def load_moca_sc(t: dict) -> pd.DataFrame:
    """Screening-visit MoCA, used as baseline when BL has none."""
    m = t["moca"][t["moca"]["EVENT_ID"].eq("SC")]
    m = m.assign(moca_sc=clean_score(m, "MCATOT"), date_sc=parse_month(m["INFODT"]))
    return m.dropna(subset=["moca_sc"]).drop_duplicates("PATNO")[["PATNO", "moca_sc", "date_sc"]]


def load_cogcat(t: dict) -> pd.DataFrame:
    c = scheduled(t["cogcat"]).assign(cogstate=lambda d: clean_score(d, "COGSTATE"))
    if "INFODT" in c:
        c["date_c"] = parse_month(c["INFODT"])
    else:
        c["date_c"] = pd.NaT
    return one_per_visit(c, ["cogstate", "date_c"])


def load_participants(t: dict) -> pd.DataFrame:
    """PD participants with sex, genetic-enrolment flag, diagnosis date."""
    s = t["status"].drop_duplicates("PATNO").copy()
    if "COHORT" in s:
        is_pd = pd.to_numeric(s["COHORT"], errors="coerce").eq(cfg.PD_COHORT_CODE)
    else:
        is_pd = s["COHORT_DEFINITION"].astype(str).str.contains("parkinson", case=False)
    s = s[is_pd]
    gen_cols = [c for c in cfg.GENETIC_ENROLL_COLS if c in s]
    if gen_cols:
        s["genetic"] = (s[gen_cols].apply(pd.to_numeric, errors="coerce").fillna(0) > 0).any(axis=1).astype(int)
    else:
        s["genetic"] = np.nan
    s = s[["PATNO", "genetic"]]

    d = t["demo"].drop_duplicates("PATNO").copy()
    if "SEX" in d:
        d["male"] = pd.to_numeric(d["SEX"], errors="coerce").map({0: 0, 1: 1})
    else:   # older exports: GENDER 0/1 = female, 2 = male
        d["male"] = pd.to_numeric(d["GENDER"], errors="coerce").map({0: 0, 1: 0, 2: 1})

    dx = t["dxhist"].copy()
    dx["dx_date"] = parse_month(dx["PDDXDT"])
    dx = dx.sort_values("dx_date").drop_duplicates("PATNO")

    return (s.merge(d[["PATNO", "male"]], on="PATNO", how="left")
             .merge(dx[["PATNO", "dx_date"]], on="PATNO", how="left"))


def active_log_rows(log: pd.DataFrame, visits: pd.DataFrame):
    """(visits with _vid, LEDD-log rows active in each visit month).

    Cleaning, because dates are month-only:
      * exact duplicate log rows are dropped;
      * when a drug's row stops in the visit month and another row of the same
        drug starts in that month (a dose change), only the new row counts, so
        the visit is not credited with both doses. Rows of the same drug that
        are both fully active (e.g. IR + CR levodopa) are kept.
    """
    log = log.drop_duplicates(["PATNO", "LEDTRT", "STARTDT", "STOPDT", "LEDD"]).copy()
    log["start"] = parse_month(log["STARTDT"])
    log["stop"] = parse_month(log["STOPDT"])
    log = log[log["start"].notna()]
    raw = log["LEDD"].astype(str).str.strip()
    log["ledd_mg"] = pd.to_numeric(raw, errors="coerce")
    log["ld_factor"] = pd.to_numeric(
        raw.str.extract(r"(?i)LD\s*[x\*]\s*([0-9.]+)")[0], errors="coerce")
    log["drug"] = log["LEDTRT"].astype(str).str.lower().str.strip()
    log["is_ld"] = log["drug"].str.contains("|".join(cfg.LEVODOPA_TERMS))

    v = visits[cfg.KEY + ["date"]].dropna(subset=["date"]).reset_index(drop=True)
    v["_vid"] = np.arange(len(v))
    m = v.merge(log[["PATNO", "drug", "start", "stop", "ledd_mg", "ld_factor", "is_ld"]], on="PATNO")
    m = m[(m["start"] <= m["date"]) & (m["stop"].isna() | (m["stop"] >= m["date"]))]

    starts_now = m["start"].eq(m["date"])
    stops_now = m["stop"].eq(m["date"]) & m["start"].lt(m["date"])
    switch = starts_now.groupby([m["_vid"], m["drug"]]).transform("any")
    return v, m[~(stops_now & switch)]


def compute_ledd(log: pd.DataFrame, visits: pd.DataFrame) -> pd.Series:
    """Total LEDD (mg/day) active at each visit month.

    Follows the PPMI LEDD method: the LEDD column holds a numeric dose for
    most drugs and a multiplier string ("LD x 0.33") for COMT inhibitors,
    which is applied to that visit's total levodopa dose.
    """
    v, m = active_log_rows(log, visits)
    g = m.groupby("_vid")
    total = g["ledd_mg"].sum(min_count=1).fillna(0)
    levodopa = m[m["is_ld"]].groupby("_vid")["ledd_mg"].sum()
    factor = g["ld_factor"].sum()
    total = total.add(factor.mul(levodopa.reindex(factor.index).fillna(0)), fill_value=0)

    out = pd.Series(0.0, index=v["_vid"])
    out.loc[total.index] = total.values
    return pd.Series(out.values, index=pd.MultiIndex.from_frame(v[cfg.KEY]))


# ---------------------------------------------------------------- assembly

def build_dataset(data_dir: Path) -> pd.DataFrame:
    """Long data: one row per PD participant x scheduled visit."""
    files = find_files(data_dir)
    absent = [r for r, p in files.items() if p is None]
    if absent:
        raise FileNotFoundError(f"No file found in {data_dir} for: {absent}. "
                                "Run check_files.py to see what was matched.")
    t = {role: read_table(p, role) for role, p in files.items()}

    long = load_updrs(t)
    long = long.merge(load_moca(t), on=cfg.KEY, how="outer").merge(
        load_cogcat(t), on=cfg.KEY, how="outer")
    long["date"] = long["date"].fillna(long["date_m"]).fillna(long["date_c"])
    long = long.drop(columns=["date_m", "date_c"])

    age = scheduled(t["age"]).copy()
    age["age"] = pd.to_numeric(age["AGE_AT_VISIT"], errors="coerce")
    long = long.merge(age.drop_duplicates(cfg.KEY)[cfg.KEY + ["age"]], on=cfg.KEY, how="left")

    ppl = load_participants(t)
    long = long.merge(ppl, on="PATNO", how="inner")

    # Time since baseline from visit month; baseline must exist.
    bl = long.loc[long["EVENT_ID"].eq("BL"), ["PATNO", "date", "age"]].rename(
        columns={"date": "bl_date", "age": "age_bl"})
    long = long.merge(bl.dropna(subset=["bl_date"]), on="PATNO", how="inner")
    long = long[long["date"].notna()]
    long["years"] = months_between(long["bl_date"], long["date"]) / 12
    long = long[long["years"] >= 0]
    long["dur_bl"] = months_between(long["dx_date"], long["bl_date"]) / 12
    long.loc[long["dur_bl"] < 0, "dur_bl"] = 0   # diagnosed in the baseline month

    long["ledd"] = compute_ledd(t["ledd"], long).reindex(
        pd.MultiIndex.from_frame(long[cfg.KEY])).values
    long["ledd"] = long["ledd"].fillna(0)

    # Total MDS-UPDRS I-III.
    long["updrs_total"] = long[["updrs1", "updrs2", "updrs3"]].sum(axis=1, min_count=3)

    # Fill a missing BL MoCA from the screening visit if SC is close enough.
    long = long.merge(load_moca_sc(t), on="PATNO", how="left")
    gap = months_between(long["date_sc"], long["bl_date"])
    use_sc = (long["EVENT_ID"].eq("BL") & long["moca"].isna()
              & gap.between(0, cfg.MOCA_SC_MAX_MONTHS))
    long.loc[use_sc, "moca"] = long.loc[use_sc, "moca_sc"]
    long["moca_bl_from_sc"] = long["PATNO"].isin(long.loc[use_sc, "PATNO"])
    long = long.drop(columns=["moca_sc", "date_sc"])

    # Baseline values of outcomes, used as survival covariates.
    for col in ["updrs3", "moca"]:
        b = long.loc[long["EVENT_ID"].eq("BL"), ["PATNO", col]].rename(columns={col: f"{col}_bl"})
        long = long.merge(b, on="PATNO", how="left")

    return long.sort_values(["PATNO", "years"]).reset_index(drop=True)
