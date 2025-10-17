# scripts/predict_batch.py  ← 新增脚本（批量推理）
import sys
from pathlib import Path
import argparse
import pandas as pd

# —— module discovery ——
ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from inference.predict import (  # noqa: E402
    load_artifacts, prepare_raw_X, transform_to_features,
    predict_proba_and_label, save_predictions
)

def parse_args():
    p = argparse.ArgumentParser(description="Batch inference with saved preprocessor & model.")
    p.add_argument("--in",   dest="in_csv",  default="data/processed/activities_outliers_winsorized_v1.csv")
    p.add_argument("--out",  dest="out_csv", default="data/predictions/predictions_v1.csv")
    p.add_argument("--pre",  dest="preproc", default="models/preprocessor_v1.joblib")
    p.add_argument("--mdl",  dest="model",   default="models/baseline_logreg_v1.joblib")
    p.add_argument("--map",  dest="emap",    default="reports/encoding_map_v1.json")
    p.add_argument("--th",   dest="thresh",  type=float, default=0.5)  # 决策阈值
    p.add_argument("--target", dest="target_col", default="participated")
    return p.parse_args()

def main():
    args = parse_args()
    in_csv   = ROOT / args.in_csv
    out_csv  = ROOT / args.out_csv
    pre_path = ROOT / args.preproc
    mdl_path = ROOT / args.model
    map_path = ROOT / args.emap

    print("✓ Loading artifacts ...")
    preproc, model, feature_names = load_artifacts(pre_path, mdl_path, map_path)

    print("✓ Reading input CSV ...", in_csv)
    df_raw = pd.read_csv(in_csv)
    print("  input shape:", df_raw.shape)

    print("✓ Preparing raw X (drop target if present) ...")
    X_raw = prepare_raw_X(df_raw, target_col=args.target_col)

    print("✓ Transforming with saved preprocessor ...")
    X = transform_to_features(X_raw, preproc, feature_names)
    print("  features shape:", X.shape)

    print(f"✓ Predicting (threshold = {args.thresh}) ...")
    pred_df = predict_proba_and_label(model, X, threshold=args.thresh)

    # 如果有明显 ID 列，把它拼回结果，便于对照（可选）
    id_cols = [c for c in df_raw.columns if "id" in c.lower()]
    if id_cols:
        pred_df = pd.concat([df_raw[id_cols].reset_index(drop=True),
                             pred_df.reset_index(drop=True)], axis=1)

    print("✓ Saving predictions to:", out_csv)
    save_predictions(pred_df, out_csv)

    # 简单统计：阳性数/阴性数
    pos = int((pred_df["pred"] == 1).sum())
    neg = int((pred_df["pred"] == 0).sum())
    print(f"Summary → positives: {pos}, negatives: {neg}, total: {len(pred_df)}")
    print(" Done.")

if __name__ == "__main__":
    main()
