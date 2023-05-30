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

    external_features = read_parquet('data/andrey_features/andrey_test.parquet')
    for col in ['cat3_grouped', 'variant_id_diff', 'category_level_2', 'category_level_3', 'category_level_4', 'target']:
        if col in external_features.columns:
            external_features = external_features.drop(col)
    mapping = {
        'BOTH': 0, 
        'NONE': 1,
        'ONLY ONE': 2
    }
    external_features = external_features.with_columns(
        [
            pl.col('variantid1').cast(pl.Int64),
            pl.col('variantid2').cast(pl.Int64),
            pl.col('has_full_match_name').cast(pl.Int8),
            pl.col('has_full_match_name_norm').cast(pl.Int8),
            pl.col('has_full_match_name_tokens').cast(pl.Int8),
            pl.col('has_full_match_name_norm_tokens').cast(pl.Int8),
            (
                pl.col('pic_embeddings_resnet_v1_fillness')
                .apply(lambda x: mapping.get(x, -1))
                .cast(pl.Int8)
                .alias('pic_embeddings_resnet_v1_fillness')
            ),
            (
                pl.col('color_parsed_fillness')
                .apply(lambda x: mapping.get(x, -1))
                .cast(pl.Int8)
                .alias('color_parsed_fillness')
            )
        ]
    )
    external_features = external_features.rename(
        {
            col: f"a__{col}"
            for col in external_features.columns
            if col not in ('variantid1', 'variantid2')
        }
    )


    dataset = (
        pairs.join(titles_features, on=["variantid1", "variantid2"])
        .join(characteristics_features, on=["variantid1", "variantid2"])
        .join(main_pic_resnet_features, on=["variantid1", "variantid2"])
        .join(name_bert_64_features, on=["variantid1", "variantid2"])
        .join(external_features, on=["variantid1", "variantid2"])
    )

    write_parquet(dataset, os.path.join(data_dir, "dataset.parquet"))


@cli.command()
@log_cli
def oof_predict(data_dir: str = Option(...), n_folds: int = Option(...)):
    predict = []
    for fold in range(1, 1+n_folds):
        predict.append(
            pl.read_csv(os.path.join(data_dir, f'cv_{fold}', 'test', 'prediction.csv'))
            .rename({'scores': 'lgbm_oof_score'})
        )
    predict = pl.concat(predict)

    print(predict.shape)

    predict.write_parquet(
        os.path.join(data_dir, 'common', 'lgbm_oof_scores_train.parquet')
    )

if __name__ == "__main__":
    cli()
