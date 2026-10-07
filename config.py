"""File patterns, expected columns and analysis constants.

Column names follow the public PPMI Data Dictionary. If a download uses a
different name, run `python check_files.py` and adjust the lists below.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data" / "raw"
DEFAULT_OUT_DIR = ROOT / "output"

# Role -> (include regex, exclude regex) matched against the normalised file
# name (lower case, non-alphanumerics collapsed to "_"). PPMI appends the
# download date (e.g. _06Oct2026), so patterns never anchor on the end.
FILE_PATTERNS = {
    "updrs1":    (r"mds_updrs_part_i_", r"patient|archived"),
    "updrs1p":   (r"mds_updrs_part_i_.*patient", r"archived"),
    "updrs2":    (r"mds_updrs_part_ii_", r"archived"),
    "updrs3":    (r"mds_updrs_part_iii", r"archived"),
    "moca":      (r"montreal_cognitive|(^|_)moca(_|$)", r"archived"),
    "cogcat":    (r"cognitive_categorization", r"archived"),
    "status":    (r"participant_status", r"archived|online"),
    "demo":      (r"^demographics", r"archived|online"),
    "age":       (r"age_at_visit", r"archived"),
    "dxhist":    (r"pd_diagnosis_history", r"archived"),
    "ledd":      (r"ledd_concomitant", r"archived"),
}

KEY = ["PATNO", "EVENT_ID"]

# Columns each table must have. Alternatives are separated by "|".
REQUIRED_COLUMNS = {
    "updrs1":  ["PATNO", "EVENT_ID", "INFODT", "NP1RTOT"],
    "updrs1p": ["PATNO", "EVENT_ID", "INFODT", "NP1PTOT"],
    "updrs2":  ["PATNO", "EVENT_ID", "INFODT", "NP2PTOT"],
    "updrs3":  ["PATNO", "EVENT_ID", "INFODT", "NP3TOT", "NHY", "PDSTATE"],
    "moca":    ["PATNO", "EVENT_ID", "INFODT", "MCATOT"],
    "cogcat":  ["PATNO", "EVENT_ID", "COGSTATE"],
    "status":  ["PATNO", "COHORT|COHORT_DEFINITION"],
    "demo":    ["PATNO", "SEX|GENDER"],
    "age":     ["PATNO", "EVENT_ID", "AGE_AT_VISIT"],
    "dxhist":  ["PATNO", "PDDXDT"],
    "ledd":    ["PATNO", "LEDTRT", "LEDD", "STARTDT", "STOPDT"],
}

# Optional columns used when present.
OPTIONAL_COLUMNS = {
    "updrs3": ["PDTRTMNT", "PDMEDYN"],
    "status": ["ENROLL_STATUS", "ENRLLRRK2", "ENRLGBA", "ENRLSNCA", "ENRLPRKN", "ENRLPINK1"],
}

# Valid score ranges; anything outside (e.g. 101 = "unable to rate") -> missing.
VALID_RANGE = {
    "NP1RTOT": (0, 24),
    "NP1PTOT": (0, 28),
    "NP2PTOT": (0, 52),
    "NP3TOT": (0, 132),
    "NHY": (0, 5),
    "MCATOT": (0, 30),
    "COGSTATE": (1, 3),
}

# Scheduled in-person visits only (drops SC, ST, PW, U01, remote/log visits).
VISIT_REGEX = r"^(BL|V\d+)$"

# PPMI cohort code for Parkinson's disease in Participant Status.
PD_COHORT_CODE = 1
GENETIC_ENROLL_COLS = ["ENRLLRRK2", "ENRLGBA", "ENRLSNCA", "ENRLPRKN", "ENRLPINK1"]

# Drug names (LEDTRT, case-insensitive) counted as levodopa when applying the
# COMT-inhibitor multipliers ("LD x 0.33" etc.) in the LEDD log.
LEVODOPA_TERMS = [
    "levodopa", "carbidopa", "sinemet", "rytary", "stalevo", "madopar",
    "duopa", "duodopa", "inbrija", "parcopa", "benserazide", "dhivy", "crexont",
]

# Milestone thresholds.
HY_EVENT = 3            # Hoehn & Yahr stage >= 3 (postural instability)
COG_EVENT_STATES = {2, 3}  # COGSTATE: 1 normal, 2 MCI, 3 dementia
MOCA_CUTOFF = 26        # secondary cognitive definition: MoCA < 26
# Confirmed MoCA < 26: the next assessed visit must also be < 26. A drop at the
# participant's last MoCA visit cannot be confirmed; True counts it as an event.
MOCA_CONFIRM_LAST_VISIT = True

# PPMI often gives the baseline MoCA at screening (SC) rather than BL. Use the
# SC score as baseline when BL has none and SC is at most this many months earlier.
MOCA_SC_MAX_MONTHS = 6

# Knots (years since baseline) for the piecewise-linear time trend in the mixed
# models: separate slopes for 0-2, 2-5 and 5+ years.
TIME_KNOTS = [2, 5]

# Cox covariates whose proportional-hazards test has p < PH_ALPHA get a
# separate hazard ratio before and after COX_TIME_SPLIT years.
PH_ALPHA = 0.05
COX_TIME_SPLIT = 3

# Dropout-pattern groups for the pattern-mixture sensitivity analysis (years).
FOLLOWUP_BINS = [0, 2, 5, float("inf")]
FOLLOWUP_LABELS = ["<2y", "2-5y", ">=5y"]

# Kaplan-Meier curves and tables stop where fewer than this many are at risk.
KM_MIN_AT_RISK = 20

# Genetic enrolment is used as a covariate only if each group (genetic /
# sporadic) has at least this many participants.
MIN_GENETIC_GROUP = 50

# Minimum participants per yearly bin before an observed mean is plotted.
MIN_CELL = 10
