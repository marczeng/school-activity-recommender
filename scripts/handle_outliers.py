import sys                             
from pathlib import Path              
import argparse                        
import pandas as pd                     

# module discovery
ROOT = Path(__file__).resolve().parents[1]   
SRC  = ROOT / "src"
if str(SRC) not in sys.path:                 
    sys.path.append(str(SRC))

from features.outlier_handling import (     
    select_numeric_continuous,
    compute_iqr_thresholds,
    winsorize_with_flags,
    zscore_diagnostics,
    save_report,
)

def parse_args():
    p = argparse.ArgumentParser(description="Outlier handling (IQR winsorization).")

    p.add_argument("--in",  dest="in_path",
                   default="data/processed/activities_clean.csv")
  
    p.add_argument("--out", dest="out_path",
                   default="data/processed/activities_outliers_winsorized_v1.csv")
   
    p.add_argument("--report", dest="report_path",
                   default="reports/outlier_report_v1.json")
    return p.parse_args()

def main():
    args = parse_args()
    in_path  = ROOT / args.in_path
    out_path = ROOT / args.out_path
    rpt_path = ROOT / args.report_path

    # 确保输出目录存在
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rpt_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) 读取数据
    df = pd.read_csv(in_path)

    # 2) 选择连续数值列（排除 0/1 二元列）
    num_cols, binary_like = select_numeric_continuous(df)
    print("Numeric (continuous):", num_cols)
    print("Skipped binary-like :", binary_like)

    # 3) 计算 IQR 阈值（向量化）
    (iqr_dict, q1, q3, iqr, lower, upper) = compute_iqr_thresholds(df, num_cols)

    # 4) 先标记，再温莎化（并保留 *_raw）
    df = winsorize_with_flags(df, num_cols, lower, upper)

    # 5) Z 分数诊断（|Z|>3 的列级占比）
    z_rate = zscore_diagnostics(df, num_cols)

    # 6) 写出数据与报告
    df.to_csv(out_path, index=False)
    save_report(rpt_path, num_cols, q1, q3, iqr, lower, upper, z_rate,
                source=str(in_path), output=str(out_path))

    # 7) 快速自检：每列异常标记占比 & 抽一列看 min/max 前后对比
    flag_cols = [f"{c}_outlier_flag" for c in num_cols]
    rates = {c: float(df[c].mean()) for c in flag_cols}   # 布尔均值=占比
    print("Outlier flag rates:", rates)
    if num_cols:
        c = num_cols[0]
        print(f"[{c}] raw min/max:", df[f"{c}_raw"].min(), df[f"{c}_raw"].max())
        print(f"[{c}] winsorized min/max:", df[c].min(), df[c].max())

    print("Saved data to:", out_path)
    print("Saved report to:", rpt_path)

if __name__ == "__main__":
    main()  