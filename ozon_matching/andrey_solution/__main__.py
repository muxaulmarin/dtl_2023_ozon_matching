from __future__ import annotations

import os
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Optional

import joblib as jbl
import polars as pl
import scipy.sparse as sp
from loguru import logger
from ozon_matching.andrey_solution.chains import enrich_by_chains
from ozon_matching.andrey_solution.feature_engineering import (
    calc_similarity,
    fit_tfidf,
    generate_categories_features,
    generate_characteristics_features,
    generate_colors_features,
    generate_names_features,
    generate_pictures_features,
    generate_variants_features,
)
from ozon_matching.andrey_solution.modeling import CatBoostCV, kfold_split, manual_split
from ozon_matching.andrey_solution.modeling.config import catboost_params
from ozon_matching.andrey_solution.preprocessing import (
    add_cat3_grouped,
    preprocess_pairs,
    preprocess_products,
)
from ozon_matching.andrey_solution.utils import map_products, normalize
from typer import Option, Typer

cli = Typer()


class FeatureType(Enum):
    variants = "variants"
    names = "names"
    categories = "categories"
    colors = "colors"
    pictures = "pictures"
    characteristics = "characteristics"


@cli.command()
def preprocess(
    pairs_path: list[Path] = Option(...),
    products_path: list[Path] = Option(...),
) -> None:
    for pairs_path_, products_path_ in zip(pairs_path, products_path):
        logger.info(f"preprocessing pairs from {pairs_path_}...")
        pairs = preprocess_pairs(pl.read_parquet(pairs_path_))
        logger.info(f"preprocessing products from {products_path_}...")
        products = preprocess_products(pl.read_parquet(products_path_))

        if "target" in pairs.columns:
            logger.info("adding cat3_grouped to pairs...")
            pairs = add_cat3_grouped(pairs, products)

        pairs_output_dir = pairs_path_.parent.parent / "preprocessed"
        pairs_output_dir.mkdir(parents=True, exist_ok=True)
        pairs_output_path = pairs_output_dir / pairs_path_.name
        logger.info(f"saving preprocessed pairs to {pairs_output_path}...")
        pairs.write_parquet(pairs_output_path)

        products_output_dir = products_path_.parent.parent / "preprocessed"
        products_output_dir.mkdir(parents=True, exist_ok=True)
        products_output_path = products_output_dir / products_path_.name
        logger.info(f"saving preprocessed products to {products_output_path}...")
        products.write_parquet(products_output_path)


@cli.command()
def generate_features(
    pairs_path: list[Path] = Option(...),
    products_path: list[Path] = Option(...),
    feature_type: list[FeatureType] = Option(...),
    output_file: Optional[Path] = None,
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
            elif feature_type_ == FeatureType.colors:
                features_generator = generate_colors_features
            elif feature_type_ == FeatureType.pictures:
                features_generator = generate_pictures_features
            elif feature_type_ == FeatureType.characteristics:
                features_generator = generate_characteristics_features
            else:
                raise ValueError(feature_type_)

            logger.info(f"generating features on {feature_type_.value}...")
            features = features_generator(dataset)
            features_dir = pairs_path_.parent.parent / "features" / feature_type_.value
            features_dir.mkdir(parents=True, exist_ok=True)
            if output_file is None:
                features_path = features_dir / (
                    normalize(
                        os.path.commonprefix([pairs_path_.name, products_path_.name])
                    )
                    + ".parquet"
                )
            else:
                features_path = features_dir / output_file

            logger.info(f"saving features to {features_path}...")
            features.write_parquet(features_path)


@cli.command()
def join_features(
    pairs_path: list[Path] = Option(...),
    output_file: list[str] = Option(...),
    features_path: Path = Option(...),
    ignore: list[str] = None,
) -> pl.DataFrame:
    for pairs_path_, output_file_ in zip(pairs_path, output_file):
        logger.info(f"reading pairs from {pairs_path_}...")
        pairs = pl.read_parquet(pairs_path_)
        for feature_path in features_path.iterdir():
            if ignore and feature_path.name in ignore:
                logger.warning(f"ignore {feature_path}")
                continue
            part = feature_path / output_file_
            logger.info(f"joining features from {part}...")
            pairs = pairs.join(
                other=pl.read_parquet(part)
                .with_columns([pl.col(["variantid1", "variantid2"]).cast(pl.UInt32)])
                .select(
                    pl.exclude(
                        [
                            "__index_level_0__",
                            "target",
                            "color_parsed1",
                            "color_parsed2",
                        ]
                    )
                ),
                how="left",
                on=["variantid1", "variantid2"],
            )
        dataset_dir = pairs_path_.parent.parent / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = dataset_dir / output_file_
        logger.info(f"saving dataset to {dataset_path}...")
        pairs.write_parquet(dataset_path)


@cli.command()
def fit_catboost(
    train_path: Path = Option(...),
    experiment_path: Path = Option(...),
    folds_path: Path = None,
    chains_path: Path = None,
    seed: Optional[int] = 777,
) -> None:
    logger.info(f"reading train from {train_path}...")
    train = pl.read_parquet(train_path)

    if folds_path is not None:
        logger.info(f"reading folds from {folds_path}...")
        cv_splitter = partial(manual_split, folds=pl.read_csv(folds_path))
    else:
        logger.info(f"no folds given, falling back to simple kfold with {seed = }...")
        cv_splitter = partial(kfold_split, seed=seed)

    chains = None
    if chains_path is not None:
        logger.info(f"reading chains from {chains_path}...")
        chains = pl.read_parquet(chains_path)

    model = CatBoostCV(**catboost_params, splitter=cv_splitter)
    model.fit(train, chains=chains)
    logger.info(f"top-10 feature importances:\n{model.feature_importances.iloc[:10]}")

    logger.info(f"saving cv models with metrics to {experiment_path}...")
    model.save(experiment_path)

    oof_path = experiment_path / "oof.parquet"
    logger.info(f"saving oof predicts to {oof_path}...")
    model.predict_oof(train).to_parquet(oof_path)


@cli.command()
def prepare_submission(
    test_path: Path = Option(...), experiment_path: Path = Option(...)
) -> None:
    logger.info(f"reading test from {test_path}...")
    test = pl.read_parquet(test_path)
    logger.info(f"reading model from {experiment_path}...")
    model = CatBoostCV.from_snapshot(experiment_path, **catboost_params)

    submission_path = experiment_path / "submission.csv"
    logger.info(f"saving submission to {submission_path}...")
    model.predict(test).to_csv(submission_path, index=False)


@cli.command()
def extract_chains(
    pairs_path: Path = Option(...),
    products_path: Path = Option(...),
    max_nodes: int = 120,
    cutoff: Optional[int] = None,
) -> None:
    logger.info(f"reading pairs from {pairs_path}...")
    pairs = pl.read_parquet(pairs_path)
    logger.info(f"reading products from {pairs_path}...")
    products = pl.read_parquet(products_path)
    logger.info("generating chains...")
    chains = enrich_by_chains(pairs, max_nodes=max_nodes, cutoff=cutoff)

    logger.info("preprocessing chains pairs...")
    chains = preprocess_pairs(chains)
    logger.info("adding cat3_grouped to pairs...")
    chains = add_cat3_grouped(chains, products)

    chains_dir = pairs_path.parent.parent / "preprocessed"
    chains_dir.mkdir(parents=True, exist_ok=True)
    chains_path = chains_dir / "train_chains_pairs.parquet"
    logger.info(f"saving extracted chains to {chains_path}...")
    chains.write_parquet(chains_path)


@cli.command()
def create_tfidf_matrix(
    train_products_path: Path = Option(...),
    test_products_path: Path = Option(...),
    col_name: list[str] = Option(...),
) -> None:
    logger.info(f"reading products train from {train_products_path}...")
    products_train = pl.read_parquet(train_products_path)
    logger.info(f"reading products test from {test_products_path}...")
    products_test = pl.read_parquet(test_products_path)

    for col_name_ in col_name:
        logger.info(f"fitting tfidf for {col_name_}...")
        tfidf, variant_id_to_index, value_to_index = fit_tfidf(
            products_train, products_test, col_name=col_name_
        )

        tfidf_dir = train_products_path.parent.parent / "tfidf" / col_name_
        tfidf_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"saving tfidf matrix to {tfidf_dir}...")
        sp.save_npz(str(tfidf_dir / "matrix.npz"), tfidf)
        jbl.dump(variant_id_to_index, tfidf_dir / "variant_id_to_index.jbl")
        jbl.dump(value_to_index, tfidf_dir / "value_to_index.jbl")


@cli.command()
def create_tfidf_similarity_features(
    pairs_path: list[Path] = Option(...),
    output_file: list[Path] = Option(...),
    col_name: list[str] = Option(...),
) -> None:
    for col_name_ in col_name:
        tfidf_dir = pairs_path[0].parent.parent / "tfidf" / col_name_
        logger.info(f"loading tfidf for {col_name_} from {tfidf_dir}...")
        tfidf = sp.load_npz(str(tfidf_dir / "matrix.npz"))
        variant_id_to_index = jbl.load(tfidf_dir / "variant_id_to_index.jbl")

        for pairs_path_, output_file_ in zip(pairs_path, output_file):
            logger.info(f"reading pairs from {pairs_path_}...")
            pairs = pl.read_parquet(pairs_path_)

            logger.info(f"calculating similarity for {col_name_}...")
            similarity = calc_similarity(
                pairs=pairs,
                col_name=col_name_,
                tfidf_matrix=tfidf,
                variant_id_to_index=variant_id_to_index,
            )
            similarity_dir = tfidf_dir.parent.parent / "features" / f"tfidf-{col_name_}"
            similarity_dir.mkdir(parents=True, exist_ok=True)
            similarity_path = similarity_dir / output_file_
            logger.info(f"saving similarity for {col_name_} to {similarity_path}...")
            similarity.write_parquet(similarity_path)


cli()
