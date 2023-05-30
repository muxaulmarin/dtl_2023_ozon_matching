from __future__ import annotations

from typing import Generator, Optional

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.model_selection import KFold


def kfold_split(
    x: pl.DataFrame, n_splits: int = 5, seed: Optional[int] = 777
) -> Generator:
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(x)


def manual_split(x: pl.DataFrame, folds: pl.DataFrame) -> Generator:
    folds = folds.with_columns(
        [pl.col("variantid1").cast(pl.UInt32), pl.col("variantid2").cast(pl.UInt32)]
    )
    for i in range(1, 6):
        variants = x.select(["variantid1", "variantid2"]).with_row_count()
        train_variants = folds.filter(pl.col(f"fold_{i}_train") == 1).select(
            ["variantid1", "variantid2", pl.lit(True).alias("included")]
        )
        val_variants = folds.filter(pl.col(f"fold_{i}_test") == 1).select(
            ["variantid1", "variantid2", pl.lit(True).alias("included")]
        )

        train_fold = variants.join(
            train_variants, how="left", on=["variantid1", "variantid2"]
        ).filter(pl.col("included"))
        val_fold = variants.join(
            val_variants, how="left", on=["variantid1", "variantid2"]
        ).filter(pl.col("included"))

        yield train_fold["row_nr"].to_list(), val_fold["row_nr"].to_list()


def calc_metrics(
    y_true: pd.Series, y_pred: pd.Series, categories: pd.Series
) -> dict[str, float]:
    return {
        "PR-AUC Macro (Precision >= 0.75)": pr_auc_macro(
            y_true, y_pred, categories, min_precision=0.75
        ),
        "ROC-AUC": roc_auc_score(y_true, y_pred),
    }


def pr_auc_macro(
    y_true: pd.Series,
    y_pred: pd.Series,
    categories: pd.Series,
    min_precision: float = 0.75,
) -> float:
    weights = []
    pr_aucs = []

    unique_cats, counts = np.unique(categories, return_counts=True)
    for i, category in enumerate(unique_cats):
        cat_idx = np.where(categories == category)[0]
        y_pred_cat = y_pred[cat_idx]
        y_true_cat = y_true[cat_idx]

        y, x, _ = precision_recall_curve(y_true_cat, y_pred_cat)
        gt_prec_level_idx = np.where(y >= min_precision)[0]

        try:
            pr_auc_prec_level = auc(x[gt_prec_level_idx], y[gt_prec_level_idx])
            if not np.isnan(pr_auc_prec_level):
                pr_aucs.append(pr_auc_prec_level)
                weights.append(counts[i] / len(categories))
        except ValueError as err:
            logger.error((category, err))
            pr_aucs.append(0)
            weights.append(0)
    return np.average(pr_aucs, weights=weights)
