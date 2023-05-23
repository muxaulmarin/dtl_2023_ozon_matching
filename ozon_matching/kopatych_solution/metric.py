import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import auc, precision_recall_curve


def pr_auc_macro(
    target_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    prec_level: float = 0.75,
    cat_column: str = "cat3_grouped",
) -> float:

    df = target_df.merge(predictions_df, on=["variantid1", "variantid2"])

    y_true = df["target"].values
    y_pred = df["scores"].values
    categories = df[cat_column].values

    logger.info(f"df size - {df.shape}, n cat - {categories.size}")

    weights = []
    pr_aucs = []

    unique_cats, counts = np.unique(categories, return_counts=True)

    for i, category in enumerate(unique_cats):
        cat_idx = np.where(categories == category)[0]
        y_pred_cat = y_pred[cat_idx]
        y_true_cat = y_true[cat_idx]

        y, x, _ = precision_recall_curve(y_true_cat, y_pred_cat)
        gt_prec_level_idx = np.where(y >= prec_level)[0]

        try:
            pr_auc_prec_level = auc(x[gt_prec_level_idx], y[gt_prec_level_idx])
            if not np.isnan(pr_auc_prec_level):
                pr_aucs.append(pr_auc_prec_level)
                weights.append(counts[i] / len(categories))
        except ValueError:
            pr_aucs.append(0)
            weights.append(0)
    metric = float(np.average(pr_aucs, weights=weights))
    return metric


def pr_auc_macro_t(
    df: pd.DataFrame,
    prec_level: float = 0.75,
    cat_column: str = "cat3_grouped",
    t: int = 1000
) -> float:

    y_true = df["target"].values
    y_pred = df["scores"].values
    categories = df[cat_column].values

    unique_cats, counts = np.unique(categories, return_counts=True)
    cat_counts = {k: v for k, v in zip(unique_cats, counts)}
    rest_cat_id = categories.max() + 1

    for i in range(categories.size):
        if cat_counts[categories[i]] <= t:
            categories[i] = rest_cat_id

    unique_cats, counts = np.unique(categories, return_counts=True)

    if counts.min() < t:
        raise ValueError("")

    weights = []
    pr_aucs = []

    for i, category in enumerate(unique_cats):
        cat_idx = np.where(categories == category)[0]
        y_pred_cat = y_pred[cat_idx]
        y_true_cat = y_true[cat_idx]

        y, x, _ = precision_recall_curve(y_true_cat, y_pred_cat)
        gt_prec_level_idx = np.where(y >= prec_level)[0]

        try:
            pr_auc_prec_level = auc(x[gt_prec_level_idx], y[gt_prec_level_idx])
            if not np.isnan(pr_auc_prec_level):
                pr_aucs.append(pr_auc_prec_level)
                weights.append(counts[i] / len(categories))
        except ValueError:
            pr_aucs.append(0)
            weights.append(0)
    metric = float(np.average(pr_aucs, weights=weights))
    return metric
