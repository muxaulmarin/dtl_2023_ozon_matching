import os

import lightgbm
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from ozon_matching.kopatych_solution.config.model_cfg import lgbm_params
from ozon_matching.kopatych_solution.cv import stratified_k_fold
from ozon_matching.kopatych_solution.features import (
    create_characteristic_features,
    create_sim_feature,
)
from ozon_matching.kopatych_solution.metric import pr_auc_macro
from ozon_matching.kopatych_solution.similarity import SimilarityEngine
from ozon_matching.kopatych_solution.utils import (
    extract_category_levels,
    get_and_create_dir,
    log_cli,
    read_model,
    read_parquet,
    write_json,
    write_model,
    write_parquet,
)
from typer import Option, Typer

cli = Typer()


@cli.command()
def dummpy_cli():
    logger.info("Dummy CLI")


@cli.command()
@log_cli
def split_data_for_cv(data_dir: str = Option(...), n_folds: int = Option(...)):

    pairs = read_parquet(os.path.join(data_dir, "data", "train_pairs.parquet"))
    pairs = pairs.with_columns([pl.col("target").cast(pl.Int8)])
    pairs = pairs.with_row_count(name="index")

    data = read_parquet(os.path.join(data_dir, "data", "train_data.parquet"))
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
        .with_row_count(name="cv_target")
    )
    pairs = pairs.join(cv_target, on=["target", "category_level_3"])

    for n, train_pairs, test_pairs in stratified_k_fold(
        data=pairs, stratify_col="cv_target", k=int(n_folds)
    ):
        cv_folder = get_and_create_dir(os.path.join(data_dir, f"cv_{n}"))
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
    train_data = read_parquet(
        os.path.join(data_dir, "train", "data.parquet"),
        columns=["variantid", "characteristic_attributes_mapping"],
    )
    test_data = read_parquet(
        os.path.join(data_dir, "test", "data.parquet"),
        columns=["variantid", "characteristic_attributes_mapping"],
    )
    data = pl.concat([train_data, test_data])
    feature = create_characteristic_features(data, pairs)
    fodler = get_and_create_dir(os.path.join(data_dir, fold_type, "features"))
    write_parquet(feature, os.path.join(fodler, "characteristics_features.parquet"))


@cli.command()
@log_cli
def create_dataset(data_dir: str = Option(...), fold_type: str = Option(...)):
    pairs = read_parquet(os.path.join(data_dir, fold_type, "pairs.parquet"))
    pairs = pairs.drop(["index", "cv_target"])
    categories = (
        pairs.select(pl.col("category_level_3"))
        .unique()
        .with_row_count(name="category_level_3_id")
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
        data=dataset_pandas.drop(columns=["target"]),
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
    write_json({"score": score}, os.path.join(data_dir, fold_type, "score.json"))


if __name__ == "__main__":
    cli()
