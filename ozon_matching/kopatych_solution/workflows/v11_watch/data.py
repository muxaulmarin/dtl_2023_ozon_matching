import os

import polars as pl
from ozon_matching.kopatych_solution.utils import (
    extract_category_levels,
    log_cli,
    read_parquet,
    write_parquet,
)
from typer import Option, Typer

cli = Typer()


@cli.command()
def dummy():
    pass


@cli.command()
@log_cli
def prepare_data(data_dir: str = Option(...)):
    data = pl.concat(
        [
            read_parquet(os.path.join(data_dir, "train", "data.parquet")),
            read_parquet(os.path.join(data_dir, "test", "data.parquet")),
        ]
    )
    data = data.unique(subset=["variantid"])
    data = extract_category_levels(data, [3, 4], category_col="categories")

    data = data.filter(pl.col('category_level_4') == 'Ремешок для смарт-часов')

    write_parquet(
        data,
        os.path.join(os.path.join(data_dir, "common_data.parquet")),
    )


@cli.command()
@log_cli
def prepare_data_submit(data_dir: str = Option(...)):
    pairs = read_parquet(os.path.join(data_dir, "pairs.parquet"))

    titles_features = read_parquet(os.path.join(data_dir, "titles_features.parquet"))
    characteristics_features = read_parquet(
        os.path.join(data_dir, "characteristics_features.parquet")
    )
    main_pic_resnet_features = read_parquet(
        os.path.join(
            data_dir, "similarity_features_main_pic_embeddings_resnet_v1.parquet"
        )
    )
    name_bert_64_features = read_parquet(
        os.path.join(data_dir, "similarity_features_name_bert_64.parquet")
    )

    dataset = (
        pairs.join(titles_features, on=["variantid1", "variantid2"])
        .join(characteristics_features, on=["variantid1", "variantid2"])
        .join(main_pic_resnet_features, on=["variantid1", "variantid2"])
        .join(name_bert_64_features, on=["variantid1", "variantid2"])
    )

    write_parquet(dataset, os.path.join(data_dir, "dataset.parquet"))


if __name__ == "__main__":
    cli()
