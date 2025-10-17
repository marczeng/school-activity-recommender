from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def select_numeric_continuous(df: pd.DataFrame) -> Tuple[List[str],List[str]]:
    #only int/float
    num_all = df.select_dtypes(include = "number").columns.tolist()

    technical_suffixes = ("_raw", "_outlier_flag")
    num_no_tech = [c for c in num_all if not c.endswith(technical_suffixes)]

    def is_all_integers(s: pd.Series) -> bool:
        s = s.dropna()
        if s.empty:
            return False  # 空列不当作 ID
        # 本身是整型 dtype
        if pd.api.types.is_integer_dtype(s):
            return True
        # 浮点：检查是否都是整数值（向量化判断）
        if pd.api.types.is_float_dtype(s):
            # 用 isclose 抗浮点误差
            return np.isclose(s % 1, 0).all()
        return False

    maybe_id = []
    for c in num_no_tech:
        name_has_id = "id" in c.lower()
        if name_has_id and is_all_integers(df[c]):
            maybe_id.append(c)

    #check for binary
    binary_like = [c for c in num_all if df[c].dropna ().isin([0,1]).all()]
    num_cols = [i for i in num_all if i not in set(binary_like) | set(maybe_id)]
    return list(num_cols), list(binary_like)

def compute_iqr_thresholds(df: pd.DataFrame, cols:List[str], iqr_eps: float = 1e-9) -> Tuple[Dict[str,Dict[str,float]], pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    if not cols:    #none
        empty = pd.Series(dtype=float)
        return {},empty,empty,empty,empty,empty
    
    q = df[cols].quantile([0.25,0.75])
    q1,q3 = q.loc[0.25], q.loc[0.75]
    iqr = (q3 - q1).replace(0,iqr_eps)

    lower = q1 - 1.5* iqr
    upper = q3 + 1.5* iqr

    iqr_dict = {
        c:{"q1":float(q1[c]), "q3":float(q3[c]), "iqr": float(iqr[c]), "lower": float(lower[c]), "upper": float(upper[c])}
        for c in cols
    }
    return iqr_dict, q1, q3, iqr, lower, upper

def winsorize_with_flags(df: pd.DataFrame, cols: List[str], lower: pd.Series, upper:pd.Series) -> pd.DataFrame:

    if not cols:
        return df
    
    #provenance
    raw_cols = [f"{c}_raw" for c in cols]
    df[raw_cols] = df[cols].copy()

    flags = df[cols].lt(lower) | df[cols].gt(upper)
    df[[f"{c}_outlier_flag" for c in cols]] = flags.astype(int)  # True/False → 1/0
    #winsorize
    df[cols] = df[cols].clip(lower= lower, upper=upper, axis=1)
    return df
#calc the proportion where zscore > 3
def zscore_diagnostics (df: pd.DataFrame, cols: List[str]) -> Dict[str,float]:
    if not cols:
        return {}

    result: Dict[str, float] = {}
    for c in cols:
        raw_name = f"{c}_raw"
        base = df[raw_name] if raw_name in df.columns else df[c]  # 安全回退

        mean = base.mean()
        std  = base.std(ddof=0)
        if std and not pd.isna(std):
            z = (base - mean) / std
            rate = float((z.abs() > 3).mean())  # 布尔均值=占比
        else:
            rate = 0.0
        result[c] = rate

    return result
    
def save_report(path: Path, num_cols: List[str],
                q1: pd.Series, q3: pd.Series, iqr: pd.Series,
                lower: pd.Series, upper: pd.Series,
                z_rate: Dict[str, float],
                source: str, output: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    #json report
    report = {
        "method": "IQR winsorization with binary flags (vectorized)",
        "num_cols": num_cols,
        "iqr_thresholds": {  #convert series into dict 
            "q1": q1.to_dict(),
            "q3": q3.to_dict(),
            "iqr": iqr.to_dict(),
            "lower": lower.to_dict(),
            "upper": upper.to_dict()
        },
        "zscore_diagnostics": z_rate,
        "source": source,
        "output": output
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)