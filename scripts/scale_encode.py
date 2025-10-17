# scripts/scale_encode.py
import sys
from pathlib import Path
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from features.scaling_encoding import (
    infer_feature_groups,
    build_preprocessor,
    fit_transform_export,
)

def parse_args():
    p = argparse.ArgumentParser(description="Feature scaling and encoding.")
    p.add_argument("--in", dest="in_path",
                   default="data/processed/activities_outliers_winsorized_v1.csv")
    p.add_argument("--out", dest="out_parquet",
                   default="data/processed/activities_scaled_encoded_v1.parquet")
    p.add_argument("--map", dest="map_json",
                   default="reports/encoding_map_v1.json")
    p.add_argument("--preproc", dest="preproc_path",
                   default="models/preprocessor_v1.joblib")
    return p.parse_args()

def main():
    args = parse_args()
    in_path, out_parquet, map_json, preproc_path = map(Path, [args.in_path, args.out_parquet, args.map_json, args.preproc_path])

    df = pd.read_csv(in_path)
    numeric_cols, binary_cols, categorical_cols, dropped_cols = infer_feature_groups(df)

    print("Numeric columns:", numeric_cols)
    print("Binary columns:", binary_cols)
    print("Categorical columns:", categorical_cols)
    if dropped_cols:
        print("Dropped (id-like):", dropped_cols)

    preproc = build_preprocessor(numeric_cols, categorical_cols, binary_cols)
    fit_transform_export(df, preproc, out_parquet, map_json, preproc_path)

    print(f"Saved scaled+encoded data to: {out_parquet}")
    print(f"Saved encoding map to: {map_json}")
    print(f"Saved preprocessor to: {preproc_path}")

if __name__ == "__main__":
    main()

