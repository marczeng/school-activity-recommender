"""Basic data cleaning for the extracurricular activities dataset.

This script performs a minimal set of sanity checks so that the downstream
pipeline can assume consistent data types and ranges.  It intentionally keeps
the logic lightweight because the demo project ships with a small toy dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_IN = "data/raw/activities_sample.csv"
DEFAULT_OUT = "data/processed/activities_clean.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean the raw student activity data.")
    parser.add_argument("--in", dest="input_path", default=DEFAULT_IN, help="Path to the raw CSV file.")
    parser.add_argument(
        "--out",
        dest="output_path",
        default=DEFAULT_OUT,
        help="Where the cleaned CSV should be written.",
    )
    return parser.parse_args()


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert a collection of columns to numeric, coercing invalid values to NaN."""

    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clip_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Clip key numeric columns into sensible ranges for the demo dataset."""

    ranges: dict[str, tuple[float, float]] = {
        "gpa": (0, 4),
        "attendance_rate": (0, 1),
        "club_interest": (0, 5),
        "grade_level": (9, 12),
        "participated": (0, 1),
    }
    for col, (lower, upper) in ranges.items():
        if col in df.columns:
            df[col] = df[col].clip(lower, upper)
    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric columns with their median and enforce integer dtypes where appropriate."""

    fill_map: dict[str, float] = {}
    for col in ["gpa", "attendance_rate", "club_interest"]:
        if col in df.columns:
            fill_map[col] = df[col].median()
    if fill_map:
        df = df.fillna(value=fill_map)

    astype_map: dict[str, str] = {}
    for col in ["grade_level", "club_interest", "participated"]:
        if col in df.columns:
            astype_map[col] = "int64"
    if astype_map:
        df = df.astype(astype_map)
    return df


def main() -> None:
    args = parse_args()
    raw_path = Path(args.input_path)
    out_path = Path(args.output_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {raw_path}")

    df = pd.read_csv(raw_path)
    print("Before:", df.shape)

    df.columns = [c.strip().lower() for c in df.columns]
    df = coerce_numeric(
        df,
        [
            "student_id",
            "grade_level",
            "gpa",
            "attendance_rate",
            "club_interest",
            "participated",
        ],
    )

    df = df.dropna(subset=["student_id"])
    df["student_id"] = df["student_id"].astype("int64")

    df = clip_ranges(df)
    df = fill_missing(df)

    print("After:", df.shape)
    print("Missing ratio:\n", df.isna().mean().sort_values(ascending=False).head(10))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print("Saved to:", out_path)


if __name__ == "__main__":
    main()
