import os

import polars as pl
from loguru import logger
from ozon_matching.kopatych_solution.cv import stratified_k_fold
from ozon_matching.kopatych_solution.similarity import SimilarityEngine
from ozon_matching.kopatych_solution.utils import extract_category_levels, write_model
from typer import Option, Typer

cli = Typer()


@cli.command()
def dummpy_cli():
    logger.info("Dummy CLI")


@cli.command()
def split_data_for_cv(data_dir: str = Option(...)):
    pairs = pl.read_parquet(os.path.join(data_dir, "data", "train_pairs.parquet"))
    pairs = pairs.with_columns([pl.col("target").cast(pl.Int8)])
    pairs = pairs.with_row_count(name="index")

    data = pl.read_parquet(os.path.join(data_dir, "data", "train_data.parquet"))
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

    for n, train_pairs, valid_pairs in stratified_k_fold(
        data=pairs, stratify_col="cv_target", k=3
    ):
        cv_folder = os.path.join(data_dir, f"cv_{n}")
        if not os.path.exists(cv_folder):
            os.mkdir(cv_folder)
        logger.info("Write parquet")
        train_pairs.write_parquet(os.path.join(cv_folder, "train_pairs.parquet"))
        valid_pairs.write_parquet(os.path.join(cv_folder, "test_pairs.parquet"))

        train_data = data.filter(
            pl.col("variantid").is_in(train_pairs["variantid1"])
            | pl.col("variantid").is_in(train_pairs["variantid2"])  # type: ignore
        )
        (
            train_data.write_parquet(  # type:ignore
                os.path.join(cv_folder, "train_data.parquet")
            )
        )
        valid_data = data.filter(
            pl.col("variantid").is_in(valid_pairs["variantid1"])
            | pl.col("variantid").is_in(valid_pairs["variantid2"])  # type: ignore
        )
        (
            valid_data.write_parquet(  # type:ignore
                os.path.join(cv_folder, "test_data.parquet")
            )
        )


@cli.command()
def fit_similarity_engine(data_dir: str = Option(...), vertor_col: str = Option(...)):
    data = pl.read_parquet(
        os.path.join(data_dir, "train_data.parquet"), columns=["variantid", vertor_col]
    )
    similarity_engine = SimilarityEngine(index_col="variantid", vector_col=vertor_col)
    similarity_engine.fit(data)
    write_model(
        os.path.join(data_dir, "models", f"similarity_engine_{vertor_col}.jbl"),
        similarity_engine,
    )


@cli.command()
def join_features(data_dir: str = Option(...)):
    pairs = pl.read_parquet(os.path.join(data_dir, "train_pairs.parquet"))
    data = pl.read_parquet(os.path.join(data_dir, "train_data.parquet"))
    logger.info(pairs.shape, data.shape)


if __name__ == "__main__":
    cli()
