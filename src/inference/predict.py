# src/inference/predict.py  ← 新增模块
from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import joblib   # 加载预处理器与模型

# 1) 载入工件：预处理器、模型、特征名映射
def load_artifacts(preproc_path: Path, model_path: Path, map_json_path: Path):
    """
    返回: (preprocessor, model, feature_names)
    """
    # joblib.load：反序列化对象（高效、常用于sklearn模型）
    preproc = joblib.load(preproc_path)   # 预处理器（ColumnTransformer）
    model   = joblib.load(model_path)     # 训练好的模型（如LogisticRegression）

    # json.load：读取JSON文件到Python字典
    with open(map_json_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    feature_names = mapping.get("feature_names", [])
    if not feature_names:
        raise ValueError("feature_names missing in encoding map JSON.")

    return preproc, model, feature_names

# 2) 准备输入X：删除目标列，选取训练时使用的原始列集合
def prepare_raw_X(df: pd.DataFrame, target_col: str = "participated") -> pd.DataFrame:
    """
    输入数据应与第四步构建预处理器时的表结构一致（未缩放、未编码）。
    若包含目标列，则删除，避免目标泄漏。
    """
    if target_col in df.columns:
        # DataFrame.drop(columns=[...])：按列名删除列，返回新DataFrame
        df = df.drop(columns=[target_col])
    return df

# 3) 变换到模型输入：transform → DataFrame（带列名）
def transform_to_features(df_raw: pd.DataFrame, preproc, feature_names: List[str]) -> pd.DataFrame:
    """
    用训练期保存的 preprocessor 对原始表做 transform。
    输出带有训练期同顺序的列名。
    """
    # Pipeline/ColumnTransformer.transform：使用“训练期拟合到的统计量/规则”做转换
    X_mat = preproc.transform(df_raw)  # 可能是稀疏矩阵

    # 稀疏矩阵转数组：hasattr(X_mat,"toarray") 判断是否支持 toarray()
    X_dense = X_mat.toarray() if hasattr(X_mat, "toarray") else X_mat

    # 用训练时保存的 feature_names 作为列顺序，构造 DataFrame
    X = pd.DataFrame(X_dense, columns=feature_names, index=df_raw.index)
    return X

# 4) 预测：概率 + 标签
def predict_proba_and_label(model, X: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    predict_proba：返回每个类别的概率（n_samples, 2），[:,1] 是“正类”概率
    np.where：根据阈值把概率转为 0/1 标签
    """
    proba = model.predict_proba(X)[:, 1]                 # 正类概率
    pred  = np.where(proba >= threshold, 1, 0).astype(int)
    out = pd.DataFrame({"proba": proba, "pred": pred}, index=X.index)
    return out

# 5) 保存推理结果
def save_predictions(df_pred: pd.DataFrame, out_path: Path) -> None:
    """
    DataFrame.to_csv(index=False)：保存为CSV，不包含行索引
    Path.parent.mkdir(..., exist_ok=True)：确保目录存在
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(out_path, index=False)
