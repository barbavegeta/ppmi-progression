"""Check which PPMI files were found and whether the expected columns exist.

Prints file names and column names only — never participant values — so the
output is safe to share when asking for help.

    python check_files.py [--data-dir PATH]
"""
import argparse

import pandas as pd

import config as cfg
from load import find_files, missing_columns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=cfg.DEFAULT_DATA_DIR)
    args = ap.parse_args()

    print(f"Looking in: {args.data_dir}\n")
    ok = True
    for role, path in find_files(args.data_dir).items():
        if path is None:
            print(f"[MISSING] {role:8s} no matching file (pattern: {cfg.FILE_PATTERNS[role][0]})")
            ok = False
            continue
        cols = [c.strip().upper() for c in pd.read_csv(path, nrows=0).columns]
        miss = missing_columns(role, cols)
        opt = [c for c in cfg.OPTIONAL_COLUMNS.get(role, []) if c not in cols]
        status = "OK" if not miss else "COLUMNS?"
        ok &= not miss
        print(f"[{status:8s}] {role:8s} {path.name}")
        if miss:
            print(f"           missing required: {miss}")
            print(f"           columns present:  {cols}")
        if opt:
            print(f"           optional not found (fine): {opt}")
    print("\nAll good — run: python run_all.py" if ok else "\nFix the items above first.")


if __name__ == "__main__":
    main()
