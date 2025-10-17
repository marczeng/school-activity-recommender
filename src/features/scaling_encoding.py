# src/features/scaling_encoding.py  
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib  # 保存预处理器

# 自动列分类
def infer_feature_groups(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    determine each col is  numeric / binary / categorical / dropped（id-like）
    """
    cols = df.columns.tolist()
    tech_suffix = ("_raw", "_outlier_flag") 
    base_cols = [c for c in cols if not c.endswith(tech_suffix)]  # 去掉技术列

    def is_all_integers(s: pd.Series) -> bool:
        """see if is int"""
        s = s.dropna()
        if s.empty:
            return False
        if pd.api.types.is_integer_dtype(s):
            return True
        if pd.api.types.is_float_dtype(s):
            return np.isclose(s % 1, 0).all()
        return False

    #  明显 ID 列（名字含 id 且值为整数）
    dropped_cols = [c for c in base_cols if "id" in c.lower() and is_all_integers(df[c])]
    remain = [c for c in base_cols if c not in dropped_cols]

    #  二元列：仅含 0/1
    binary_cols = [c for c in remain if df[c].dropna().isin([0, 1]).all()]
    remain = [c for c in remain if c not in binary_cols]

    #  类别列（字符串或唯一值较少）
    categorical_cols = []
    for c in remain:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]):
            categorical_cols.append(c)
        else:
            nunique = df[c].nunique(dropna=True)
            if nunique <= max(20, int(0.05 * len(df))):
                categorical_cols.append(c)
    remain = [c for c in remain if c not in categorical_cols]

    # (4) 剩余为数值列
    numeric_cols = [c for c in remain if pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols, binary_cols, categorical_cols, dropped_cols

# 2 构建预处理器
def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str], binary_cols: List[str]) -> ColumnTransformer:
    """
    返回
      数值列 → RobustScaler()
      类别列 → OneHotEncoder(handle_unknown='ignore')
      二元列 → 直接 passthrough
    """
    num_pipe = Pipeline([("scaler", RobustScaler())])
    cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse=True))])

    transformers = []
    if numeric_cols:
        transformers.append(("num", num_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", cat_pipe, categorical_cols))
    if binary_cols:
        transformers.append(("bin", "passthrough", binary_cols))

    preproc = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)
    return preproc

# 3 拟合+导出
def fit_transform_export(df: pd.DataFrame, preprocessor: ColumnTransformer,
                         out_parquet: Path, map_json: Path, save_pipeline_to: Path) -> None:
    """
    拟合、变换、恢复列名、保存 parquet + map + pipeline
    """
    X = preprocessor.fit_transform(df)
    feature_names = []

    # 生成列名（对 OHE 需要特征展开名）
    for name, trans, cols in preprocessor.transformers_:
        if name == "cat":
            ohe = trans.named_steps["ohe"]
            ohe_names = ohe.get_feature_names_out(cols).tolist()
            feature_names.extend(ohe_names)
        elif name in ["num", "bin"]:
            feature_names.extend(cols)

    # 将稀疏矩阵转为稠密（教学方便，可改为保持稀疏）
    if hasattr(X, "toarray"):
        X_dense = X.toarray()
    else:
        X_dense = X
    out_df = pd.DataFrame(X_dense, columns=feature_names, index=df.index)

    # 保存 parquet
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_parquet, index=False)

    # 保存类别映射 JSON
    map_json.parent.mkdir(parents=True, exist_ok=True)
    mapping = {"feature_names": feature_names}
    with open(map_json, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    # 保存 pipeline 对象
    save_pipeline_to.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, save_pipeline_to)

