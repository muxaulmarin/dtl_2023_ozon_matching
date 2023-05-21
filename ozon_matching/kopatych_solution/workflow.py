import os

import lightgbm
import numpy as np
import pandas as pd
import polars as pl
from ozon_matching.kopatych_solution.config.model_cfg import lgbm_params
from ozon_matching.kopatych_solution.cv import stratified_k_fold
from ozon_matching.kopatych_solution.features import (
    _create_characteristics_dict,
    _create_characteristics_keys,
    create_characteristic_features,
    create_sim_feature,
)
from ozon_matching.kopatych_solution.metric import pr_auc_macro
from ozon_matching.kopatych_solution.similarity import SimilarityEngine
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
from sklearn.metrics import roc_auc_score
from typer import Option, Typer

cli = Typer()


@cli.command()
@log_cli
def create_characteristics_dict(
    data_dir: str = Option(...), experiment: str = Option(...)
):
    train_data = read_parquet(
        os.path.join(data_dir, "data", "train", "data.parquet"),
        columns=["variantid", "characteristic_attributes_mapping", "categories"],
    )
    test_data = read_parquet(
        os.path.join(data_dir, "data", "test", "data.parquet"),
        columns=["variantid", "characteristic_attributes_mapping", "categories"],
    )
    data = pl.concat([train_data, test_data]).unique(subset=["variantid"])
    characteristics_dict = _create_characteristics_dict(data)
    write_json(
        characteristics_dict,
        os.path.join(data_dir, experiment, "characteristics_dict.json"),
    )

    train_pairs = read_parquet(
        os.path.join(data_dir, "data", "train", "pairs.parquet"),
        columns=["variantid1", "variantid2"],
    )
    test_pairs = read_parquet(
        os.path.join(data_dir, "data", "test", "pairs.parquet"),
        columns=["variantid1", "variantid2"],
    )
    pairs = pl.concat([train_pairs, test_pairs]).unique()
    characteristics_index = {
        characteristic: i
        for i, characteristic in enumerate(
            _create_characteristics_keys(pairs, characteristics_dict)
        )
    }
    write_json(
        characteristics_index,
        os.path.join(data_dir, experiment, "characteristics_index.json"),
    )

    data = extract_category_levels(data, [3], "categories")
    categories = (
        data.select(pl.col("category_level_3"))
        .unique()
        .with_row_count(name="category_level_3_id")
    )
    write_parquet(categories, os.path.join(data_dir, experiment, "categories.parquet"))


@cli.command()
@log_cli
def split_data_for_cv(
    data_dir: str = Option(...),
    n_folds: int = Option(...),
    experiment: str = Option(...),
):

    pairs = read_parquet(os.path.join(data_dir, "data", "train", "pairs.parquet"))
    pairs = pairs.with_columns([pl.col("target").cast(pl.Int8)])
    pairs = pairs.with_row_count(name="index")

    data = read_parquet(os.path.join(data_dir, "data", "train", "data.parquet"))
    data = extract_category_levels(data, [3], "categories")

    pairs = pairs.join(
        (
            data.select(
                [pl.col("variantid").alias("variantid1"), pl.col("category_level_3")]
            )
        ),
        on=["variantid1"],
    )
    cv_target = (
        pairs.select(pl.col(["target", "category_level_3"]))
        .unique()
        .sort(["category_level_3", "target"])
        .with_row_count(name="cv_target")
    )
    pairs = pairs.join(cv_target, on=["target", "category_level_3"])

    for n, train_pairs, test_pairs in stratified_k_fold(
        data=pairs, stratify_col="cv_target", k=int(n_folds)
    ):
        cv_folder = get_and_create_dir(os.path.join(data_dir, experiment, f"cv_{n}"))
        train_folder = get_and_create_dir(os.path.join(cv_folder, "train"))
        test_folder = get_and_create_dir(os.path.join(cv_folder, "test"))

        write_parquet(train_pairs, os.path.join(train_folder, "pairs.parquet"))
        write_parquet(test_pairs, os.path.join(test_folder, "pairs.parquet"))

        train_data = data.filter(
            pl.col("variantid").is_in(train_pairs["variantid1"])
            | pl.col("variantid").is_in(train_pairs["variantid2"])
        )
        write_parquet(train_data, os.path.join(train_folder, "data.parquet"))

        test_data = data.filter(
            pl.col("variantid").is_in(test_pairs["variantid1"])
            | pl.col("variantid").is_in(test_pairs["variantid2"])
        )
        write_parquet(test_data, os.path.join(test_folder, "data.parquet"))


@cli.command()
@log_cli
def fit_similarity_engine(data_dir: str = Option(...), vector_col: str = Option(...)):

    train_data = read_parquet(
        os.path.join(data_dir, "train", "data.parquet"),
        columns=["variantid", vector_col],
    )
    test_data = read_parquet(
        os.path.join(data_dir, "test", "data.parquet"),
        columns=["variantid", vector_col],
    )

    data = pl.concat([train_data, test_data])

    similarity_engine = SimilarityEngine(index_col="variantid", vector_col=vector_col)
    similarity_engine.fit(data)
    write_model(
        os.path.join(data_dir, "models", f"similarity_engine_{vector_col}.jbl"),
        similarity_engine,
    )


@cli.command()
@log_cli
def create_similarity_features(
    data_dir: str = Option(...), fold_type: str = Option(...)
):
    pairs = read_parquet(os.path.join(data_dir, fold_type, "pairs.parquet"))

    pic_similarity_engine: SimilarityEngine = read_model(
        os.path.join(
            data_dir, "models", "similarity_engine_main_pic_embeddings_resnet_v1.jbl"
        )
    )
    name_similarity_engine: SimilarityEngine = read_model(
        os.path.join(data_dir, "models", "similarity_engine_name_bert_64.jbl")
    )
    for engine in [pic_similarity_engine, name_similarity_engine]:
        feature = create_sim_feature(pairs, engine)
        fodler = get_and_create_dir(os.path.join(data_dir, fold_type, "features"))
        write_parquet(
            feature,
            os.path.join(fodler, f"similarity_features_{engine.vector_col}.parquet"),
        )


@cli.command()
@log_cli
def create_characteristics_features(
    data_dir: str = Option(...), fold_type: str = Option(...)
):
    pairs = read_parquet(os.path.join(data_dir, fold_type, "pairs.parquet"))

    characteristics_index = read_json(
        os.path.join(os.path.dirname(data_dir), "characteristics_index.json")
    )
    characteristics_dict = read_json(
        os.path.join(os.path.dirname(data_dir), "characteristics_dict.json")
    )
    feature = create_characteristic_features(
        pairs, characteristics_index, characteristics_dict
    )

    fodler = get_and_create_dir(os.path.join(data_dir, fold_type, "features"))
    write_parquet(feature, os.path.join(fodler, "characteristics_features.parquet"))


@cli.command()
@log_cli
def create_dataset(data_dir: str = Option(...), fold_type: str = Option(...)):

    pairs = read_parquet(os.path.join(data_dir, fold_type, "pairs.parquet"))
    pairs = pairs.drop(["index", "cv_target"])
    categories = read_parquet(
        os.path.join(os.path.dirname(data_dir), "categories.parquet")
    )
    pairs = pairs.join(categories, on=["category_level_3"]).drop("category_level_3")

    pic_sim_features = read_parquet(
        os.path.join(
            data_dir,
            fold_type,
            "features",
            "similarity_features_main_pic_embeddings_resnet_v1.parquet",
        )
    )
    name_sim_features = read_parquet(
        os.path.join(
            data_dir, fold_type, "features", "similarity_features_name_bert_64.parquet"
        )
    )
    characteristics_features = read_parquet(
        os.path.join(
            data_dir, fold_type, "features", "characteristics_features.parquet"
        )
    )

    dataset = (
        pairs.join(pic_sim_features, on=["variantid1", "variantid2"])
        .join(name_sim_features, on=["variantid1", "variantid2"])
        .join(characteristics_features, on=["variantid1", "variantid2"])
    )
    write_parquet(
        dataset, os.path.join(data_dir, fold_type, "features", "dataset.parquet")
    )

    dataset_pandas = dataset.to_pandas()

    lgbm_dataset = lightgbm.Dataset(
        data=dataset_pandas.drop(columns=["target", "variantid1", "variantid2"]),
        label=dataset_pandas["target"].values,
        reference=None
        if fold_type == "train"
        else read_model(os.path.join(data_dir, "models", "train_lgbm_dataset.jbl")),
        categorical_feature=["category_level_3_id"],
    )
    write_model(
        os.path.join(data_dir, "models", f"{fold_type}_lgbm_dataset.jbl"), lgbm_dataset
    )


@cli.command()
@log_cli
def fit_model(data_dir: str = Option(...), params_version: str = Option(...)):
    train_dataset = read_model(
        os.path.join(data_dir, "models", "train_lgbm_dataset.jbl")
    )
    valid_dataset = read_model(
        os.path.join(data_dir, "models", "test_lgbm_dataset.jbl")
    )
    model = lightgbm.train(
        params=lgbm_params[params_version],
        train_set=train_dataset,
        valid_sets=[valid_dataset],
        valid_names=["valid"],
        early_stopping_rounds=25,
        verbose_eval=25,
    )
    write_model(os.path.join(data_dir, "models", f"lgbm_{params_version}.jbl"), model)


@cli.command()
@log_cli
def predict(data_dir: str = Option(...), params_version: str = Option(...)):
    model = read_model(os.path.join(data_dir, "models", f"lgbm_{params_version}.jbl"))
    for fold_type in ["train", "test"]:
        data = read_parquet(
            os.path.join(data_dir, fold_type, "features", "dataset.parquet")
        )
        data = data.to_pandas()
        predict = model.predict(data[model.feature_name()], raw_score=True)
        predict = 1 / (1 + np.exp(-predict))
        data["scores"] = predict
        data[["variantid1", "variantid2", "scores"]].to_csv(
            os.path.join(data_dir, fold_type, "prediction.csv"), index=False
        )


@cli.command()
@log_cli
def evaluate(data_dir: str = Option(...), fold_type: str = Option(...)):
    target = (
        read_parquet(os.path.join(data_dir, fold_type, "pairs.parquet"))
        .drop(["index", "cv_target"])
        .to_pandas()
    )
    predictions = pd.read_csv(os.path.join(data_dir, fold_type, "prediction.csv"))
    score = pr_auc_macro(
        target, predictions, prec_level=0.75, cat_column="category_level_3"
    )
    roc_auc = roc_auc_score(target["target"].values, predictions["scores"].values)
    write_json(
        {"competition_metric": score, "rocauc": roc_auc},
        os.path.join(data_dir, fold_type, "score.json"),
    )


@cli.command()
@log_cli
def cv_scores(
    data_dir: str = Option(...),
    n_folds: int = Option(...),
    experiment: str = Option(...),
):
    metrics = {}
    for fold_type in ["train", "test"]:
        scores = {"competition_metric": [], "rocauc": []}
        for n in range(1, n_folds + 1):
            path = os.path.join(
                data_dir, experiment, f"cv_{n}", fold_type, "score.json"
            )
            data = read_json(path)
            scores["competition_metric"].append(data["competition_metric"])
            scores["rocauc"].append(data["rocauc"])
        metrics[fold_type] = {
            "competition_metric": scores["competition_metric"],
            "avg_competition_metric": np.mean(scores["competition_metric"]),
            "std_competition_metric": np.std(scores["competition_metric"]),
            "rocauc": scores["rocauc"],
            "avg_rocauc": np.mean(scores["rocauc"]),
            "std_rocauc": np.std(scores["rocauc"]),
        }
    write_json(metrics, os.path.join(data_dir, experiment, "cv_result.json"))


if __name__ == "__main__":
    cli()
