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
def split_data_for_cv(data_dir: str = Option(...), n_folds: int = Option(...)):

    pairs = read_parquet(os.path.join(data_dir, "common", "train", "pairs.parquet"))
    pairs = pairs.select(
        [
            pl.col("variantid1").cast(pl.Int64),
            pl.col("variantid2").cast(pl.Int64),
            pl.col("target").cast(pl.Int8),
        ]
    )

    pairs = pairs.unique(subset=["variantid1", "variantid2"])
    pairs = pairs.with_row_count(name="index")

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

    external_features = read_parquet("data/andrey_features/andrey_train.parquet")
    external_features = external_features.drop(
        [
            "cat3_grouped",
            "variant_id_diff",
            "category_level_2",
            "category_level_3",
            "category_level_4",
            "target",
        ]
    )
    mapping = {"BOTH": 0, "NONE": 1, "ONLY ONE": 2}
    external_features = external_features.with_columns(
        [
            pl.col("variantid1").cast(pl.Int64),
            pl.col("variantid2").cast(pl.Int64),
            pl.col("has_full_match_name").cast(pl.Int8),
            pl.col("has_full_match_name_norm").cast(pl.Int8),
            pl.col("has_full_match_name_tokens").cast(pl.Int8),
            pl.col("has_full_match_name_norm_tokens").cast(pl.Int8),
            (
                pl.col("pic_embeddings_resnet_v1_fillness")
                .apply(lambda x: mapping.get(x, -1))
                .cast(pl.Int8)
                .alias("pic_embeddings_resnet_v1_fillness")
            ),
            (
                pl.col("color_parsed_fillness")
                .apply(lambda x: mapping.get(x, -1))
                .cast(pl.Int8)
                .alias("color_parsed_fillness")
            ),
        ]
    )
    external_features = external_features.rename(
        {
            col: f"a__{col}"
            for col in external_features.columns
            if col not in ("variantid1", "variantid2")
        }
    )

    features = (
        titles_features.join(characteristics_features, on=["variantid1", "variantid2"])
        .join(main_pic_resnet_features, on=["variantid1", "variantid2"])
        .join(name_bert_64_features, on=["variantid1", "variantid2"])
        .join(external_features, on=["variantid1", "variantid2"])
    )

    for n, train_pairs, test_pairs in stratified_k_fold(
        data=pairs, stratify_col="target", k=int(n_folds)
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
