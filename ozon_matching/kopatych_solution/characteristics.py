import json
import string

import numpy as np
import pandas as pd
import polars as pl
from Levenshtein import distance as levenshtein_distance
from loguru import logger
from tqdm.auto import tqdm
from typer import Option, Typer
from ozon_matching.kopatych_solution.utils import (
    extract_category_levels,
    get_and_create_dir,
    log_cli,
    read_json,
    read_model,
    read_parquet,
    write_json,
    write_model,
    write_parquet,
)
import os

cli = Typer()



class CharacteristicsModel:
    def __init__(self):
        logger.info("Init CharacteristicsModel")
        self.characteristics = {}
        self.indexes = {}
        self.punctuation = set(string.punctuation)

    def _prepare_characteristic(self, text: str):
        text = text.lower().replace(" ", "")
        new_text = ""
        for char in text:
            if char not in self.punctuation:
                new_text += char
        return new_text

    def fit(self, data: pl.DataFrame, pairs: pl.DataFrame):
        self._update_characteristics(data)
        self._update_indexes(pairs)

    def predict(self, data: pl.DataFrame) -> pl.DataFrame:
        return self._batch_predict_match(data).join(
            self._batch_predict_match_stats(data), on=["variantid1", "variantid2"]
        )

    def _update_characteristics(self, data: pl.DataFrame):
        logger.info("_update_characteristics")
        for variantid, characteristic in tqdm(
            zip(
                data["variantid"].to_list(),
                data["characteristic_attributes_mapping"].to_list(),
            ),
            total=data.shape[0],
        ):
            if characteristic is not None:
                self.characteristics[variantid] = {
                    self._prepare_characteristic(k): self._prepare_characteristic(v[0])
                    for k, v in json.loads(characteristic).items()
                }
        logger.info(f"characteristics dict size - {len(self.characteristics)}")

    def _update_indexes(self, data: pl.DataFrame):
        logger.info("_update_indexes")
        common_characteristics = set()
        for pair in tqdm(
            data.select(pl.col(["variantid1", "variantid2"])).iter_rows(),
            total=data.shape[0],
        ):
            item_a = self.characteristics.get(pair[0], {})
            item_b = self.characteristics.get(pair[1], {})
            common_characteristics = common_characteristics.union(
                set(item_a.keys()).intersection(item_b.keys())
            )
        for _, characteristic in tqdm(
            self.characteristics.items(), total=len(self.characteristics)
        ):
            common_characteristics = common_characteristics.union(characteristic.keys())

        for n, characteristic in tqdm(
            enumerate(sorted(list(common_characteristics))),
            total=len(common_characteristics),
        ):
            self.indexes[characteristic] = n

        logger.info(f"characteristics index size - {len(self.indexes)}")

    def _predict_match(self, variantid1: int, variantid2: int) -> np.ndarray:
        vector = np.empty(len(self.indexes))
        vector[:] = np.nan
        variantid1_characteristics = self.characteristics.get(variantid1, {})
        variantid2_characteristics = self.characteristics.get(variantid2, {})
        for characteristic in set(variantid1_characteristics.keys()).intersection(
            variantid2_characteristics.keys()
        ):
            n_col = self.indexes[characteristic]
            variantid1_characteristic_value = variantid1_characteristics[characteristic]
            variantid2_characteristic_value = variantid2_characteristics[characteristic]
            if variantid1_characteristic_value == variantid2_characteristic_value:
                vector[n_col] = 1
            else:
                vector[n_col] = 0
        return vector

    def _batch_predict_match(self, data: pl.DataFrame) -> pl.DataFrame:
        batch = []
        rows = data.select(pl.col(["variantid1", "variantid2"])).iter_rows()
        for pair in tqdm(rows, total=data.shape[0]):
            vector = self._predict_match(*pair)
            batch.append(vector)
        matrix = np.vstack(batch)
        df_match_features_pandas = pd.DataFrame(
            matrix,
            columns=[f"match_feature_{i}" for i in self.indexes.values()],
        )
        match_features = pl.from_pandas(df_match_features_pandas)
        match_features = match_features.select(
            [pl.col(col).cast(pl.Int8).alias(col) for col in match_features.columns]
        )
        match_features = match_features.with_columns(
            [
                pl.lit(data["variantid1"]).alias("variantid1"),
                pl.lit(data["variantid2"]).alias("variantid2"),
            ]
        )
        return match_features

    def _predict_match_stats(self, variantid1: int, variantid2: int) -> pl.DataFrame:
        variantid1_characteristics = self.characteristics.get(variantid1, {})
        variantid2_characteristics = self.characteristics.get(variantid2, {})
        union_characteristics = set(variantid1_characteristics).union(
            variantid2_characteristics
        )
        intersect_characteristics = set(variantid1_characteristics).intersection(
            variantid2_characteristics
        )

        n_match = 0
        distances = []

        for characteristic in intersect_characteristics:
            variantid1_characteristic_value = variantid1_characteristics[characteristic]
            variantid2_characteristic_value = variantid2_characteristics[characteristic]
            if variantid1_characteristic_value == variantid2_characteristic_value:
                n_match += 1
            else:
                distances.append(
                    levenshtein_distance(
                        variantid1_characteristic_value, variantid2_characteristic_value
                    )
                )
        match_stats = pl.DataFrame(
            [
                [
                    variantid1,
                    variantid2,
                    len(variantid1_characteristics),
                    len(variantid2_characteristics),
                    len(union_characteristics),
                    len(intersect_characteristics),
                    n_match,
                    np.mean(distances) if distances else 0,
                ]
            ],
            orient="row",
            schema={
                "variantid1": pl.Int64,
                "variantid2": pl.Int64,
                "n_left_key": pl.Int16,
                "n_right_key": pl.Int16,
                "n_union_key": pl.Int16,
                "n_intersect_key": pl.Int16,
                "n_match": pl.Int16,
                "levenshtein_distance": pl.Float32,
            },
        )
        return match_stats

    def _batch_predict_match_stats(self, data: pl.DataFrame) -> pl.DataFrame:
        batch = []
        rows = data.select(pl.col(["variantid1", "variantid2"])).iter_rows()
        for pair in tqdm(rows, total=data.shape[0]):
            batch.append(self._predict_match_stats(*pair))
        match_stats = pl.concat(batch)
        match_stats = match_stats.with_columns(
            [
                (pl.col("n_match") / pl.col("n_left_key")).alias("left_match_ratio"),
                (pl.col("n_match") / pl.col("n_right_key")).alias("right_match_ratio"),
                (pl.col("n_match") / pl.col("n_union_key")).alias("union_match_ratio"),
                (pl.col("n_match") / pl.col("n_intersect_key")).alias(
                    "intersect_match_ratio"
                ),
                (
                    (pl.col("n_left_key") - pl.col("n_match")) / pl.col("n_left_key")
                ).alias("left_dismatch_ratio"),
                (
                    (pl.col("n_right_key") - pl.col("n_match")) / pl.col("n_right_key")
                ).alias("right_dismatch_ratio"),
                (
                    (pl.col("n_union_key") - pl.col("n_match")) / pl.col("n_union_key")
                ).alias("union_dismatch_ratio"),
                (
                    (pl.col("n_intersect_key") - pl.col("n_match"))
                    / pl.col("n_intersect_key")
                ).alias("intersect_dismatch_ratio"),
            ]
        )
        return match_stats

@cli.command()
def dummy():
    pass



@cli.command()
@log_cli
def fit_characteristics_model(data_dir: str = Option(...)):
    data = read_parquet(
        os.path.join(data_dir, "union_data.parquet"),
        columns=["variantid", "characteristic_attributes_mapping"],
    )
    pairs = pl.concat(
        [
            read_parquet(
                os.path.join(data_dir, "train", "pairs.parquet"),
                columns=["variantid1", "variantid2"],
            ),
            read_parquet(
                os.path.join(data_dir, "test", "pairs.parquet"),
                columns=["variantid1", "variantid2"],
            ),
        ]
    )

    model = CharacteristicsModel()
    model.fit(data, pairs)

    write_model(
        os.path.join(data_dir, "characteristics_model.jbl"),
        model,
    )

@cli.command()
@log_cli
def create_characteristics_features(
    data_dir: str = Option(...),
    fold_type: str = Option(...),
    n_folds: int = Option(...),
):
    pairs = read_parquet(os.path.join(data_dir, fold_type, "pairs.parquet"))

    characteristics_model: CharacteristicsModel = read_model(
        os.path.join(
            os.path.dirname(data_dir), f"cv_{n_folds + 1}", "characteristics_model.jbl"
        )
    )
    feature = characteristics_model.predict(pairs)
    write_parquet(
        feature, os.path.join(data_dir, fold_type, "characteristics_features.parquet")
    )

if __name__ == "__main__":
    cli()