import json

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from ozon_matching.kopatych_solution.similarity import SimilarityEngine
from tqdm.auto import tqdm


def create_sim_feature(data: pl.DataFrame, engine: SimilarityEngine) -> pl.DataFrame:
    logger.info(f"Create cosine_similarity_feature with vector {engine.vector_col}")
    rows = data.select(pl.col(["variantid1", "variantid2"])).iter_rows()  # type: ignore
    feature = [engine.get_similarity(row[0], row[1]) for row in rows]
    df_feature = pl.DataFrame(
        data={
            f"cosine_similarity_{engine.vector_col}": feature,
        }
    )
    df_feature = df_feature.with_columns(
        [
            pl.lit(data["variantid1"]).alias("variantid1"),
            pl.lit(data["variantid2"]).alias("variantid2"),
        ]
    )
    logger.info(df_feature.head())
    return df_feature  # type: ignore


def _create_characteristics_dict(data: pl.DataFrame):
    logger.info("_create_characteristics_dict")
    return {
        variantid: {k: v[0] for k, v in json.loads(characteristic).items()}
        for variantid, characteristic in zip(
            data["variantid"].to_list(),
            data["characteristic_attributes_mapping"].to_list(),
        )
        if characteristic is not None
    }


def _create_characteristics_keys(data, characteristics):
    logger.info("_create_characteristics_keys")
    characteristics_keys = set()
    errors = 0
    for pair in tqdm(
        data.select(pl.col(["variantid1", "variantid2"])).iter_rows(),
        total=data.shape[0],
    ):
        try:
            item_a = characteristics[pair[0]]
            item_b = characteristics[pair[1]]
            characteristics_keys = characteristics_keys.union(
                set(item_a.keys()).intersection(item_b.keys())
            )
        except KeyError:
            errors += 1
            continue
    for _, characteristic in characteristics.items():
        characteristics_keys = characteristics_keys.union(characteristic.keys())
    logger.info(f"_create_characteristics_keys errors count - {errors}")
    return list(characteristics_keys)


def create_characteristic_features(data: pl.DataFrame, pairs: pl.DataFrame):

    logger.info("create_characteristic_features")

    characteristics = _create_characteristics_dict(data)
    characteristics_keys = _create_characteristics_keys(pairs, characteristics)
    characteristics_keys_indexes = {k: i for i, k in enumerate(characteristics_keys)}

    match_features_matrix = np.empty((pairs.shape[0], len(characteristics_keys)))
    match_features_matrix[:] = np.nan
    rows = pairs.select(
        pl.col(["variantid1", "variantid2"])
    ).iter_rows()  # type: ignore
    for n_row, pair in tqdm(
        enumerate(rows),
        total=pairs.shape[0],
    ):
        try:
            item_a = characteristics[pair[0]]
            item_b = characteristics[pair[1]]
        except KeyError:
            continue
        for k in set(item_a.keys()).intersection(item_b.keys()):
            n_col = characteristics_keys_indexes[k]
            value_a = item_a[k]
            value_b = item_b[k]
            if value_a == value_b:
                match_features_matrix[n_row, n_col] = 1
            else:
                match_features_matrix[n_row, n_col] = 0

    df_match_features_pandas = pd.DataFrame(
        match_features_matrix,
        columns=[f"match_feature_{i}" for i in range(len(characteristics_keys))],
    )
    match_features = pl.from_pandas(df_match_features_pandas)
    match_features = match_features.select(
        [pl.col(col).cast(pl.Int8).alias(col) for col in match_features.columns]  # type: ignore
    )
    match_features = match_features.with_columns(
        [
            pl.lit(pairs["variantid1"]).alias("variantid1"),
            pl.lit(pairs["variantid2"]).alias("variantid2"),
        ]
    )
    logger.info(match_features.head())
    return match_features
