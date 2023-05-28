import os

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from loguru import logger
from ozon_matching.kopatych_solution.metric import pr_auc_macro_t
from ozon_matching.kopatych_solution.utils import (
    log_cli,
    read_json,
    read_model,
    read_parquet,
    write_json,
    write_model,
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from typer import Option, Typer

cli = Typer()

NFOLDS = 5


@cli.command()
def dummy():
    pass


@cli.command()
@log_cli
def fit_model(data_dir: str = Option(...)):

    model_params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "is_unbalance": True,
        "n_estimators": 5000,
        "max_depth": -1,
        "num_leaves": 127,
        "min_data_in_leaf": 10,
        "learning_rate": 0.025,
        "lambda_l1": 1e-3,
        "lambda_l2": 1e-3,
        "max_bin": 255,
        "use_missing": True,
        "zero_as_missing": False,
        "random_state": 13,
    }

    dataset = read_parquet(os.path.join(data_dir, "train", "dataset.parquet"))
    cv = KFold(n_splits=NFOLDS, shuffle=True, random_state=13)
    X = dataset.drop(["variantid1", "variantid2", "target"]).to_pandas()
    y = dataset["target"].to_numpy()
    n = 1
    for train_index, valid_index in cv.split(X, y):
        X_train, y_train = X.iloc[train_index].copy(deep=True), y[train_index]
        X_valid, y_valid = X.iloc[valid_index].copy(deep=True), y[valid_index]
        model = LGBMClassifier(**model_params)
        model.fit(
            X=X_train,
            y=y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric=["auc"],
            early_stopping_rounds=50,
        )
        write_model(os.path.join(data_dir, f"lgbm_{n}.jbl"), model)
        n += 1


@cli.command()
@log_cli
def predict(data_dir: str = Option(...)):

    data = read_parquet(os.path.join(data_dir, "dataset.parquet"))
    data = data.to_pandas()
    scores = []
    for n in range(1, 1 + NFOLDS):
        model = read_model(os.path.join(os.path.dirname(data_dir), f"lgbm_{n}.jbl"))
        predict = model.predict_proba(data[model.feature_name_])[:, 1]
        scores.append(predict.reshape(-1, 1))
    data["scores"] = np.hstack(scores).mean(axis=1)
    data[["variantid1", "variantid2", "scores"]].to_csv(
        os.path.join(data_dir, "prediction.csv"), index=False
    )


@cli.command()
@log_cli
def evaluate(
    data_dir: str = Option(...),
    fold_type: str = Option(...),
):
    predictions = pl.read_csv(os.path.join(data_dir, fold_type, "prediction.csv"))
    target = read_parquet(
        os.path.join(os.path.dirname(data_dir), "common", "train", "pairs.parquet"),
        columns=["variantid1", "variantid2", "target"],
    )
    categories = read_parquet(
        os.path.join(os.path.dirname(data_dir), "common", "common_data.parquet"),
        columns=["variantid", "category_level_3"],
    )
    categories = categories.rename({"variantid": "variantid1"})

    df = (
        predictions.select(pl.col(["variantid1", "variantid2", "scores"]))
        .join(target, on=["variantid1", "variantid2"], how="left")
        .join(categories, on=["variantid1"])
        .with_columns([pl.col("target").fill_null(0)])
        .to_pandas()
    )
    logger.info(df.head())

    pr_auc_1000 = pr_auc_macro_t(df, cat_column="category_level_3", t=1000)
    pr_auc_50 = pr_auc_macro_t(df, cat_column="category_level_3", t=50)
    rocauc = roc_auc_score(df["target"].values, df["scores"].values)
    q95 = np.quantile(df["scores"].values, 0.95)

    write_json(
        {
            "pr_auc_1000": pr_auc_1000,
            "pr_auc_50": pr_auc_50,
            "rocauc": rocauc,
            "q95": q95,
        },
        os.path.join(data_dir, fold_type, "score.json"),
    )


@cli.command()
@log_cli
def cv_scores(
    data_dir: str = Option(...),
    n_folds: int = Option(...),
):
    metrics = {}
    for fold_type in ["train", "test"]:
        scores = {"pr_auc_1000": [], "pr_auc_50": [], "rocauc": [], "q95": []}
        for n in range(1, n_folds + 1):
            path = os.path.join(data_dir, f"cv_{n}", fold_type, "score.json")
            data = read_json(path)
            scores["pr_auc_1000"].append(data["pr_auc_1000"])
            scores["pr_auc_50"].append(data["pr_auc_50"])
            scores["rocauc"].append(data["rocauc"])
            scores["q95"].append(data["q95"])
        metrics[fold_type] = {
            "pr_auc_1000": scores["pr_auc_1000"],
            "avg_pr_auc_1000": np.mean(scores["pr_auc_1000"]),
            "std_pr_auc_1000": np.std(scores["pr_auc_1000"]),
            "pr_auc_50": scores["pr_auc_50"],
            "avg_pr_auc_50": np.mean(scores["pr_auc_50"]),
            "std_pr_auc_50": np.std(scores["pr_auc_50"]),
            "rocauc": scores["rocauc"],
            "avg_rocauc": np.mean(scores["rocauc"]),
            "std_rocauc": np.std(scores["rocauc"]),
            "q95": scores["q95"],
            "avg_q95": np.mean(scores["q95"]),
            "std_q95": np.std(scores["q95"]),
        }
    write_json(metrics, os.path.join(data_dir, "cv_result.json"))


@cli.command()
@log_cli
def oof_predict(
    data_dir: str = Option(...),
    n_folds: int = Option(...),
):
    dataset = read_parquet(
        os.path.join(data_dir, "test", "dataset.parquet")
    ).to_pandas()
    predicts = []
    for fold in range(1, n_folds + 1):
        for n in range(1, 1 + NFOLDS):
            model: LGBMClassifier = read_model(
                os.path.join(os.path.dirname(data_dir), f"cv_{fold}", f"lgbm_{n}.jbl")
            )
            predicts.append(
                model.predict_proba(dataset[model.feature_name_])[:, 1].reshape(-1, 1)
            )
    predict = np.mean(predicts, axis=0)
    dataset["target"] = predict
    dataset[["variantid1", "variantid2", "target"]].to_csv(
        os.path.join(data_dir, "submit_folds.csv"), index=False
    )


if __name__ == "__main__":
    cli()
