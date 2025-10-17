"""CLI entry-point that applies robust IQR-based winsorisation to the dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure the package modules are importable when the script is executed directly.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from features.outlier_handling import (  # noqa: E402
    select_numeric_continuous,
    compute_iqr_thresholds,
    winsorize_with_flags,
    zscore_diagnostics,
    save_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Outlier handling (IQR winsorisation).")
    parser.add_argument("--in", dest="in_path", default="data/processed/activities_clean.csv")
    parser.add_argument(
        "--out",
        dest="out_path",
        default="data/processed/activities_outliers_winsorized_v1.csv",
    )
    parser.add_argument(
        "--report",
        dest="report_path",
        default="reports/outlier_report_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = ROOT / args.in_path
    out_path = ROOT / args.out_path
    rpt_path = ROOT / args.report_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rpt_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    num_cols, binary_like = select_numeric_continuous(df)
    print("Numeric (continuous):", num_cols)
    print("Skipped binary-like :", binary_like)

    _, q1, q3, iqr, lower, upper = compute_iqr_thresholds(df, num_cols)
    df = winsorize_with_flags(df, num_cols, lower, upper)
    z_rate = zscore_diagnostics(df, num_cols)

    df.to_csv(out_path, index=False)
    save_report(
        rpt_path,
        num_cols,
        q1,
        q3,
        iqr,
        lower,
        upper,
        z_rate,
        source=str(in_path),
        output=str(out_path),
    )

    flag_cols = [f"{c}_outlier_flag" for c in num_cols]
    rates = {c: float(df[c].mean()) for c in flag_cols}
    print("Outlier flag rates:", rates)
    if num_cols:
        first = num_cols[0]
        print(f"[{first}] raw min/max:", df[f"{first}_raw"].min(), df[f"{first}_raw"].max())
        print(f"[{first}] winsorized min/max:", df[first].min(), df[first].max())

    print("Saved data to:", out_path)
    print("Saved report to:", rpt_path)


if __name__ == "__main__":
    main()
