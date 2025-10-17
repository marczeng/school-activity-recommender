import sys
from pathlib import Path
import argparse
import pandas as pd

# —— module discovery —— 
ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from models.train_eval import (   # noqa: E402
    load_xy, split_data, build_baseline_logreg,
    cross_validate_report, train_and_evaluate,
    export_feature_importance, save_json, save_model
)

def parse_args():
    p = argparse.ArgumentParser(description="Train baseline model with CV and holdout evaluation.")
    p.add_argument("--X",  dest="features_parquet", default="data/processed/activities_scaled_encoded_v1.parquet")
    p.add_argument("--y",  dest="labels_csv",      default="data/processed/activities_outliers_winsorized_v1.csv")
    p.add_argument("--target", dest="target_col",  default="participated")
    p.add_argument("--cv-report",  dest="cv_report",  default="reports/model_cv_report_v1.json")
    p.add_argument("--ho-report",  dest="ho_report",  default="reports/model_holdout_report_v1.json")
    p.add_argument("--feat-imp",   dest="feat_imp",   default="reports/feature_importance_v1.csv")
    p.add_argument("--model-out",  dest="model_out",  default="models/baseline_logreg_v1.joblib")
    return p.parse_args()

def main():
    args = parse_args()
    X_path = ROOT / args.features_parquet
    y_path = ROOT / args.labels_csv

    print(" Loading X & y ...")
    X, y = load_xy(X_path, y_path, target_col=args.target_col)
    print("  shapes:", X.shape, y.shape)

    print(" Split train/valid ...")
    X_train, X_valid, y_train, y_valid = split_data(X, y, test_size=0.2, seed=42)
    print("  train:", X_train.shape, "valid:", X_valid.shape)

    print(" Build baseline model (LogisticRegression) ...")
    model = build_baseline_logreg(seed=42)

    print(" Cross-validation ...")
    cv_rep = cross_validate_report(model, X_train, y_train, cv_splits=5)
    save_json(cv_rep, ROOT / args.cv_report)
    print("  CV report saved to:", ROOT / args.cv_report)

    print(" Train & evaluate on holdout ...")
    ho_metrics = train_and_evaluate(model, X_train, y_train, X_valid, y_valid)
    save_json(ho_metrics, ROOT / args.ho_report)
    print("  Holdout report saved to:", ROOT / args.ho_report)

    print("Save model & feature importance ...")
    save_model(model, ROOT / args.model_out)
    export_feature_importance(model, X.columns.tolist(), ROOT / args.feat_imp)
    print("  Model saved to:", ROOT / args.model_out)
    print("  Feature importance saved to:", ROOT / args.feat_imp)

    print("Done.")

if __name__ == "__main__":
    main()
