from typer import Option, Typer
from ozon_matching.kopatych_solution.utils import (
    extract_category_levels,
    get_and_create_dir,
    log_cli,
    read_json,
    read_model,
    read_parquet,
    write_json,
    write_model,
    write_parquet,
)
from ozon_matching.kopatych_solution.metric import pr_auc_macro_t
from sklearn.metrics import roc_auc_score
import numpy as np
from lightgbm import LGBMClassifier
from ozon_matching.kopatych_solution.config.model_cfg import lgbm_params
from sklearn.model_selection import KFold
import polars as pl
import os
from loguru import logger

cli = Typer()

NFOLDS = 5

@cli.command()
def dummy():
    pass


@cli.command()
@log_cli
def create_dataset(data_dir: str = Option(...), fold_type: str = Option(...), sf_oof_path: str = Option(...), n_folds: int = Option(...)):

    pairs = read_parquet(os.path.join(data_dir, fold_type, "pairs.parquet"))

    pic_sim_features = read_parquet(
        os.path.join(
            data_dir,
            fold_type,
            "similarity_features_main_pic_embeddings_resnet_v1.parquet",
        )
    )
    name_sim_features = read_parquet(
        os.path.join(data_dir, fold_type, "similarity_features_name_bert_64.parquet")
    )
    characteristics_features = read_parquet(
        os.path.join(data_dir, fold_type, "characteristics_features.parquet")
    )

    _ = read_parquet(
        os.path.join(os.path.dirname(data_dir), f"cv_{n_folds + 1}", "union_data.parquet"),
        columns=[
            "category_level_3_id",
            "category_level_4_id",
            "variantid",
        ],
    )

    sergey_oof_predictions = read_parquet(sf_oof_path).rename({'target': 'sergey_oof_predictions'}).with_columns(
        [pl.col('variantid1').cast(pl.Int64), pl.col('variantid2').cast(pl.Int64),]
    )

    pairs = pairs.join(pic_sim_features, on=["variantid1", "variantid2"])
    logger.info(f"Join pic_sim_features, Dataset size - {pairs.shape[0]}")
    pairs = pairs.join(name_sim_features, on=["variantid1", "variantid2"])
    logger.info(f"Join name_sim_features, Dataset size - {pairs.shape[0]}")
    pairs = pairs.join(characteristics_features, on=["variantid1", "variantid2"])
    logger.info(f"Join characteristics_features, Dataset size - {pairs.shape[0]}")
    pairs = pairs.join(sergey_oof_predictions, on=["variantid1", "variantid2"])
    logger.info(f"Join sergey_oof_predictions, Dataset size - {pairs.shape[0]}")

    write_parquet(pairs, os.path.join(data_dir, fold_type, "dataset.parquet"))


@cli.command()
@log_cli
def fit_model(data_dir: str = Option(...), params_version: str = Option(...)):
    dataset = read_parquet(os.path.join(data_dir, "train", "dataset.parquet"))
    cv = KFold(n_splits=NFOLDS, shuffle=True, random_state=13)
    X = dataset.drop(
        ["variantid1", "variantid2", "target"]
    ).to_pandas()
    y = dataset["target"].to_numpy()
    n = 1
    for train_index, valid_index in cv.split(X, y):
        X_train, y_train = X.iloc[train_index].copy(deep=True), y[train_index]
        X_valid, y_valid = X.iloc[valid_index].copy(deep=True), y[valid_index]
        model = LGBMClassifier(**lgbm_params[params_version])
        model.fit(
            X=X_train,
            y=y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric=['auc'],
            early_stopping_rounds=50
        )
        write_model(os.path.join(data_dir, f"lgbm_{params_version}_{n}.jbl"), model)
        n+=1


@cli.command()
@log_cli
def predict(data_dir: str = Option(...), params_version: str = Option(...), fold_type: str = Option(...)):

    data = read_parquet(os.path.join(data_dir, fold_type, "dataset.parquet"))
    data = data.to_pandas()
    scores = []
    for n in range(1, 1 + NFOLDS):
        model = read_model(os.path.join(data_dir, f"lgbm_{params_version}_{n}.jbl"))
        predict = model.predict_proba(data[model.feature_name_])[:, 1]
        scores.append(predict.reshape(-1, 1))
    data["scores"] = np.hstack(scores).mean(axis=1)
    data[["variantid1", "variantid2", "scores"]].to_csv(
        os.path.join(data_dir, fold_type, "prediction.csv"), index=False
    )


@cli.command()
@log_cli
def evaluate(
    data_dir: str = Option(...),
    fold_type: str = Option(...),
    n_folds: int = Option(...),
):
    target = read_parquet(
        os.path.join(
            os.path.dirname(data_dir), f"cv_{n_folds + 1}", "train", "pairs.parquet"
        ),
        columns=["variantid1", "variantid2", "target"],
    )
    predictions = pl.read_csv(os.path.join(data_dir, fold_type, "prediction.csv"))
    categories = read_parquet(
        os.path.join(
            os.path.dirname(data_dir), f"cv_{n_folds + 1}", "union_data.parquet"
        ),
        columns=["variantid", "category_level_3_id"],
    )
    categories = categories.rename({'variantid': 'variantid1'})

    df = (
        predictions.select(pl.col(["variantid1", "variantid2", "scores"]))
        .join(target, on=["variantid1", "variantid2"])
        .join(categories, on=["variantid1"])
        .to_pandas()
    )

    write_json(
        {
            "pr_auc_1000": pr_auc_macro_t(df, cat_column='category_level_3_id', t=1000), 
            "pr_auc_50": pr_auc_macro_t(df, cat_column='category_level_3_id', t=50), 
            "rocauc": roc_auc_score(df["target"].values, df["scores"].values)
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
    for fold_type in ["train", "test", 'adversarial']:
        scores = {"pr_auc_1000": [], "pr_auc_50": [],"rocauc": []}
        for n in range(1, n_folds + 1):
            path = os.path.join(data_dir, f"cv_{n}", fold_type, "score.json")
            data = read_json(path)
            scores["pr_auc_1000"].append(data["pr_auc_1000"])
            scores["pr_auc_50"].append(data["pr_auc_50"])
            scores["rocauc"].append(data["rocauc"])
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
        }
    write_json(metrics, os.path.join(data_dir, "cv_result.json"))


@cli.command()
@log_cli
def oof_predict(
    data_dir: str = Option(...),
    n_folds: int = Option(...),
    experiment_version: str = Option(...),
):
    dataset = read_parquet(
        os.path.join(data_dir, f"cv_{n_folds + 1}", "test", "dataset.parquet")
    ).to_pandas()
    predicts = []
    for fold in range(1, n_folds + 1):
        for n in range(1, 1 + NFOLDS):
            model: LGBMClassifier = read_model(
                os.path.join(data_dir, f"cv_{fold}", f"lgbm_{experiment_version}_{n}.jbl")
            )
            predicts.append(model.predict_proba(dataset[model.feature_name_])[:, 1].reshape(-1, 1))
    predict = np.mean(predicts, axis=0)
    dataset['target'] = predict
    dataset[["variantid1", "variantid2", 'target']].to_csv(
        os.path.join(data_dir, "submit_folds.csv"), index=False
    )


if __name__ == "__main__":
    cli()