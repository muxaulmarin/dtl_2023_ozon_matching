import json
import string
from typing import Dict

import numpy as np
import pandas as pd
import polars as pl
from Levenshtein import distance as levenshtein_distance
from loguru import logger
from ozon_matching.kopatych_solution.similarity import SimilarityEngine
from tqdm.auto import tqdm

PUNCTUATION = set(string.punctuation)


def create_sim_feature(data: pl.DataFrame, engine: SimilarityEngine) -> pl.DataFrame:
    logger.info(f"Create cosine_similarity_feature with vector {engine.vector_col}")
    rows = data.select(pl.col(["variantid1", "variantid2"])).iter_rows()
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
    return df_feature


def _create_characteristics_dict(data: pl.DataFrame):
    logger.info("_create_characteristics_dict")
    return {
        variantid: {k: v[0] for k, v in json.loads(characteristic).items()}
        for variantid, characteristic in tqdm(
            zip(
                data["variantid"].to_list(),
                data["characteristic_attributes_mapping"].to_list(),
            ),
            total=data.shape[0],
        )
        if characteristic is not None
    }


def prepare_characteristic(text: str):
    text = text.lower().replace(" ", "")
    new_text = ""
    for char in text:
        if char not in PUNCTUATION:
            new_text += char
    return new_text


def _create_characteristics_dict_v5(data: pl.DataFrame):
    characteristics = {
        variantid: {
            prepare_characteristic(k): prepare_characteristic(v[0])
            for k, v in json.loads(characteristic).items()
        }
        for variantid, characteristic in tqdm(
            zip(
                data["variantid"].to_list(),
                data["characteristic_attributes_mapping"].to_list(),
            ),
            total=data.shape[0],
        )
        if characteristic is not None
    }
    logger.info(f"_create_characteristics_dict_v5, len - {len(characteristics)}")
    return characteristics


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
    logger.info(f"_create_characteristics_keys errors count - {errors}")

    logger.info(f"len characteristics_keys after intersect {len(characteristics_keys)}")
    for _, characteristic in characteristics.items():
        characteristics_keys = characteristics_keys.union(characteristic.keys())
    logger.info(f"len characteristics_keys after union {len(characteristics_keys)}")

    return sorted(list(characteristics_keys))


def create_characteristic_features(
    pairs: pl.DataFrame,
    indexes: Dict[str, int],
    characteristics: Dict[str, Dict[str, str]],
):

    logger.info("create_characteristic_features")
    n_changes = 0
    match_features_matrix = np.empty((pairs.shape[0], len(indexes)))
    match_features_matrix[:] = np.nan
    rows = pairs.select(pl.col(["variantid1", "variantid2"])).iter_rows()
    for n_row, pair in tqdm(
        enumerate(rows),
        total=pairs.shape[0],
    ):
        try:
            item_a = characteristics[str(pair[0])]
            item_b = characteristics[str(pair[1])]
        except KeyError:
            continue
        for k in set(item_a.keys()).intersection(item_b.keys()):
            n_col = indexes[k]
            value_a = item_a[k]
            value_b = item_b[k]
            if value_a == value_b:
                match_features_matrix[n_row, n_col] = 1
            else:
                match_features_matrix[n_row, n_col] = 0
            n_changes += 1
    logger.info(f"n_changes - {n_changes}")

    df_match_features_pandas = pd.DataFrame(
        match_features_matrix,
        columns=[f"match_feature_{i}" for i in range(len(indexes))],
    )
    match_features = pl.from_pandas(df_match_features_pandas)
    match_features = match_features.select(
        [pl.col(col).cast(pl.Int8).alias(col) for col in match_features.columns]
    )
    match_features = match_features.with_columns(
        [
            pl.lit(pairs["variantid1"]).alias("variantid1"),
            pl.lit(pairs["variantid2"]).alias("variantid2"),
        ]
    )
    logger.info(match_features.head())
    return match_features


def create_characteristic_features_v5(
    pairs: pl.DataFrame,
    characteristics: Dict[str, Dict[str, str]],
):

    logger.info("create_characteristic_features")
    characteristic_features_v5 = []
    rows = pairs.select(pl.col(["variantid1", "variantid2"])).iter_rows()
    for row in tqdm(rows, total=pairs.shape[0]):
        f = []
        characteristic_a = characteristics.get(str(row[0]), {})
        characteristic_b = characteristics.get(str(row[1]), {})
        k_union = set(characteristic_a).union(characteristic_b)
        k_intersect = set(characteristic_a).intersection(characteristic_b)
        f.extend(
            [
                len(characteristic_a),
                len(characteristic_b),
                len(k_union),
                len(k_intersect),
            ]
        )
        n_match = 0
        distances = []
        for k in k_intersect:
            value_a = characteristic_a[k]
            value_b = characteristic_b[k]
            if value_a == value_b:
                n_match += 1
            else:
                distances.append(levenshtein_distance(value_a, value_b))
        f.extend([n_match, np.mean(distances) if distances else 0])
        characteristic_features_v5.append(f)

    characteristic_features_v5 = pl.DataFrame(characteristic_features_v5, orient="row")
    characteristic_features_v5 = characteristic_features_v5.select(
        [
            pairs["variantid1"],
            pairs["variantid2"],
            pl.col("column_0").alias("n_left_key"),
            pl.col("column_1").alias("n_right_key"),
            pl.col("column_2").alias("n_union_key"),
            pl.col("column_3").alias("n_intersect_key"),
            pl.col("column_4").alias("n_match"),
            pl.col("column_5").alias("levenshtein_distance"),
        ]
    )
    characteristic_features_v5 = characteristic_features_v5.with_columns(
        [
            (pl.col("n_match") / pl.col("n_left_key")).alias("left_match_ratio"),
            (pl.col("n_match") / pl.col("n_right_key")).alias("right_match_ratio"),
            (pl.col("n_match") / pl.col("n_union_key")).alias("union_match_ratio"),
            (pl.col("n_match") / pl.col("n_intersect_key")).alias(
                "intersect_match_ratio"
            ),
            ((pl.col("n_left_key") - pl.col("n_match")) / pl.col("n_left_key")).alias(
                "left_dismatch_ratio"
            ),
            ((pl.col("n_right_key") - pl.col("n_match")) / pl.col("n_right_key")).alias(
                "right_dismatch_ratio"
            ),
            ((pl.col("n_union_key") - pl.col("n_match")) / pl.col("n_union_key")).alias(
                "union_dismatch_ratio"
            ),
            (
                (pl.col("n_intersect_key") - pl.col("n_match"))
                / pl.col("n_intersect_key")
            ).alias("intersect_dismatch_ratio"),
        ]
    )
    return characteristic_features_v5
