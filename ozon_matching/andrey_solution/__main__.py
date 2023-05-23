from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

import polars as pl
from loguru import logger
from ozon_matching.andrey_solution.feature_engineering import (
    generate_categories_features,
    generate_characteristics_features,
    generate_colors_features,
    generate_names_features,
    generate_pictures_features,
    generate_variants_features,
)
from ozon_matching.andrey_solution.preprocessing import (
    preprocess_pairs,
    preprocess_products,
)
from ozon_matching.andrey_solution.utils import map_products, normalize
from typer import Option, Typer

cli = Typer()


class DataType(Enum):
    pairs = "pairs"
    products = "products"


class FeatureType(Enum):
    variants = "variants"
    names = "names"
    categories = "categories"
    colors = "colors"
    pictures = "pictures"
    characteristics = "characteristics"


@cli.command()
def preprocess(
    input_path: list[Path] = Option(...), data_type: DataType = Option(...)
) -> None:
    if data_type == DataType.pairs:
        preprocessor = preprocess_pairs
    elif data_type == DataType.products:
        preprocessor = preprocess_products
    else:
        raise ValueError(data_type)

    for input_path_ in input_path:
        output_dir = input_path_.parent.parent / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / input_path_.name

        logger.info(f"reading {input_path_}...")
        df = pl.read_parquet(input_path_)
        logger.info("preprocessing...")
        df_prep = preprocessor(df)
        logger.info(f"saving preprocessed to {output_path}")
        df_prep.write_parquet(output_path)


@cli.command()
def generate_features(
    pairs_path: list[Path] = Option(...),
    products_path: list[Path] = Option(...),
    feature_type: list[FeatureType] = Option(...),
) -> None:
    for pairs_path_, products_path_ in zip(pairs_path, products_path):
        logger.info(f"reading pairs from {pairs_path_}...")
        pairs = pl.read_parquet(pairs_path_)
        logger.info(f"reading products from {products_path_}...")
        products = pl.read_parquet(products_path_)
        logger.info("mapping products to pairs...")
        dataset = map_products(pairs, products)

        for feature_type_ in feature_type:
            if feature_type_ == FeatureType.variants:
                features_generator = generate_variants_features
            elif feature_type_ == FeatureType.names:
                features_generator = generate_names_features
            elif feature_type_ == FeatureType.categories:
                features_generator = generate_categories_features
            elif feature_type == FeatureType.colors:
                features_generator = generate_colors_features
            elif feature_type_ == FeatureType.pictures:
                features_generator = generate_pictures_features
            elif feature_type_ == FeatureType.characteristics:
                features_generator = generate_characteristics_features
            else:
                raise ValueError(feature_type_)

            logger.info("generating features...")
            features = features_generator(dataset)
            features_dir = pairs_path_.parent.parent / "features" / feature_type_.value
            features_dir.mkdir(parents=True, exist_ok=True)
            features_path = features_dir / (
                normalize(os.path.commonprefix([pairs_path_.name, products_path_.name]))
                + ".parquet"
            )
            logger.info(f"saving features to {features_path}...")
            features.write_parquet(features_path)


@cli.command()
def join_features(
    pairs_path: list[Path] = Option(...), features_path: Path = Option(...)
) -> pl.DataFrame:
    for pairs_path_ in pairs_path:
        logger.info(f"reading pairs from {pairs_path_}...")
        pairs = pl.read_parquet(pairs_path_)
        prefix = ""
        for feature_path in features_path.iterdir():
            parts = max(
                [
                    (p, len(os.path.commonprefix([pairs_path_.name, p.name])))
                    for p in feature_path.iterdir()
                ],
                key=lambda x: x[1],
            )
            if parts[1] == 0:
                raise ValueError(pairs_path_, feature_path)
            part = parts[0]
            prefix = part.name.split(".", maxsplit=2)[0]
            logger.info(f"joining features from {part}...")
            pairs = pairs.join(
                other=pl.read_parquet(part),
                how="left",
                on=["variantid1", "variantid2"],
            )
        dataset_dir = pairs_path_.parent.parent / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = dataset_dir / (prefix + ".parquet")
        logger.info(f"saving dataset to {dataset_path}...")
        pairs.write_parquet(dataset_path)


cli()
