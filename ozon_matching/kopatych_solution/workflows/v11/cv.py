import os

import polars as pl
from ozon_matching.kopatych_solution.cv import stratified_k_fold
from ozon_matching.kopatych_solution.utils import log_cli, read_parquet, write_parquet
from typer import Option, Typer

cli = Typer()


@cli.command()
def dummy():
    pass


@cli.command()
@log_cli
def split_data_for_cv(
    data_dir: str = Option(...),
    n_folds: int = Option(...),
    chain_pairs: str = Option(...),
):

    original_pairs = read_parquet(
        os.path.join(data_dir, "common", "train", "pairs.parquet")
    )
    original_pairs = original_pairs.select(
        [
            pl.col("variantid1").cast(pl.Int64),
            pl.col("variantid2").cast(pl.Int64),
            pl.col("target").cast(pl.Int8),
            pl.lit(0).cast(pl.Int8).alias("is_chain"),
        ]
    )

    chain_pairs = read_parquet(chain_pairs).with_columns(
        [pl.col("target").cast(pl.Int8)]
    )
    chain_pairs = chain_pairs.select(
        [
            pl.col("variantid1").cast(pl.Int64),
            pl.col("variantid2").cast(pl.Int64),
            pl.col("target").cast(pl.Int8),
            pl.lit(1).cast(pl.Int8).alias("is_chain"),
        ]
    )

    pairs = pl.concat([original_pairs, chain_pairs])
    pairs = pairs.unique(subset=["variantid1", "variantid2"])
    pairs = pairs.with_row_count(name="index")
    pairs = pairs.join(
        pairs.select(pl.col(["target", "is_chain"]))
        .unique()
        .with_row_count(name="cv_target"),
        on=["target", "is_chain"],
    )

    titles_features = read_parquet(
        os.path.join(data_dir, "common", "train", "titles_features.parquet")
    )
    characteristics_features = read_parquet(
        os.path.join(data_dir, "common", "train", "characteristics_features.parquet")
    )
    main_pic_resnet_features = read_parquet(
        os.path.join(
            data_dir,
            "common",
            "train",
            "similarity_features_main_pic_embeddings_resnet_v1.parquet",
        )
    )
    name_bert_64_features = read_parquet(
        os.path.join(
            data_dir, "common", "train", "similarity_features_name_bert_64.parquet"
        )
    )

    features = (
        titles_features.join(characteristics_features, on=["variantid1", "variantid2"])
        .join(main_pic_resnet_features, on=["variantid1", "variantid2"])
        .join(name_bert_64_features, on=["variantid1", "variantid2"])
    )

    for n, train_pairs, test_pairs in stratified_k_fold(
        data=pairs, stratify_col="cv_target", k=int(n_folds)
    ):

        train_pairs = train_pairs.select(
            pl.col(["variantid1", "variantid2", "target"])
        ).join(features, on=["variantid1", "variantid2"])

        write_parquet(
            train_pairs,
            os.path.join(data_dir, f"cv_{n}", "train", "dataset.parquet"),
        )

        test_pairs = test_pairs.select(pl.col(["variantid1", "variantid2"])).join(
            features, on=["variantid1", "variantid2"]
        )

        write_parquet(
            test_pairs,
            os.path.join(data_dir, f"cv_{n}", "test", "dataset.parquet"),
        )


if __name__ == "__main__":
    cli()
