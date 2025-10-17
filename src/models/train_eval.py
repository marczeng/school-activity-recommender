from pathlib import Path                 # 路径处理（跨平台）
import json                              # 写 JSON 报告
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)

import joblib                            # 保存/加载模型

# 1) 载入 X/y，并防止目标泄漏（target leakage）
def load_xy(features_parquet: Path, labels_csv: Path, target_col: str = "participated") -> Tuple[pd.DataFrame, pd.Series]:
    """
    读取特征矩阵 X（parquet）与标签 y（csv），并对齐索引。
    若 X 中包含 target_col（如预处理误保留），将其删除以防目标泄漏。
    """
    X = pd.read_parquet(features_parquet)         # 读取 parquet（高效二进制）
    y_df = pd.read_csv(labels_csv)                # 读取 csv（含 target 列）
    if target_col not in y_df.columns:
        raise KeyError(f"Target column '{target_col}' not found in {labels_csv}")

    y = y_df[target_col]                          # 取出标签列（Series）

    # 防御式：如果 X 里有目标列，删除它（避免模型偷看答案）
    if target_col in X.columns:
        X = X.drop(columns=[target_col])

    # 对齐索引长度（常规场景两者行数一致；若不一致则抛错或重置）
    if len(X) != len(y):
        raise ValueError(f"Length mismatch: X={len(X)} vs y={len(y)}")

    return X, y

# 2) 划分训练/验证（留出集）
def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed: int = 42):
    """
    使用分层抽样（stratify=y）划分训练/验证，保持正负比例稳定。
    返回：X_train, X_valid, y_train, y_valid
    """
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_valid, y_train, y_valid

# 3) 构建基线模型（Logistic Regression）
def build_baseline_logreg(seed: int = 42) -> LogisticRegression:
    """
    逻辑回归（LogisticRegression）基线模型：
    - class_weight='balanced'：类别不平衡时自动调权重
    - solver='liblinear'：稳定的二分类解算器
    - max_iter=1000：充足迭代避免未收敛警告
    """
    return LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=seed,
    )

# 4) 交叉验证（StratifiedKFold + cross_val_score）
def cross_validate_report(model, X: pd.DataFrame, y: pd.Series, cv_splits: int = 5) -> Dict:
    """
    使用分层K折交叉验证评估 Accuracy/F1/ROC-AUC 的均值与标准差。
    返回 dict 便于写 JSON 报告。
    """
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    # cross_val_score：对每个折次返回一个分数数组
    acc_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    f1_scores  = cross_val_score(model, X, y, cv=skf, scoring="f1")
    auc_scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")

    report = {
        "cv_splits": cv_splits,
        "accuracy": {"mean": float(acc_scores.mean()), "std": float(acc_scores.std())},
        "f1":       {"mean": float(f1_scores.mean()),  "std": float(f1_scores.std())},
        "roc_auc":  {"mean": float(auc_scores.mean()), "std": float(auc_scores.std())},
    }
    return report

# 5) 训练最终模型并在留出集评估
def train_and_evaluate(model, X_train, y_train, X_valid, y_valid) -> Dict:
    """
    拟合模型（fit）→ 预测标签（predict）与概率（predict_proba）→ 计算多种指标。
    """
    model.fit(X_train, y_train)                         # 拟合参数（梯度/极大似然等）
    y_pred = model.predict(X_valid)                     # 离散预测（0/1）
    # predict_proba[:,1]：取正类的概率，用于 ROC-AUC
    y_prob = model.predict_proba(X_valid)[:, 1]

    metrics = {
        "accuracy":  float(accuracy_score(y_valid, y_pred)),
        "precision": float(precision_score(y_valid, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_valid, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_valid, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_valid, y_prob)),
        "classification_report": classification_report(y_valid, y_pred, output_dict=True),
    }
    return metrics

# 6) 线性模型的“特征重要性”（系数）导出
def export_feature_importance(model, feature_names: List[str], path: Path) -> None:
    """
    对 LogisticRegression：model.coef_[0] 即每个特征的权重（正→提高预测为1的倾向）。
    保存为 CSV：feature, weight
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = model.coef_[0] if hasattr(model, "coef_") else np.zeros(len(feature_names))
    out = pd.DataFrame({"feature": feature_names, "weight": weights})
    out.sort_values("weight", ascending=False, inplace=True)
    out.to_csv(path, index=False)

# 7) 保存模型与报告
def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
