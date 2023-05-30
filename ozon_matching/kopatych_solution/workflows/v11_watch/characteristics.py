import json
import os
import string

import polars as pl
from Levenshtein import distance as levenshtein_distance
from loguru import logger
from ozon_matching.kopatych_solution.utils import (
    log_cli,
    read_model,
    read_parquet,
    write_model,
    write_parquet,
)
from ozon_matching.kopatych_solution.workflows.v11.nlp import (
    longest_common_prefix,
    longest_common_subsequence,
)
from tqdm.auto import tqdm
from typer import Option, Typer

cli = Typer()


class CharacteristicsModel:
    def __init__(self):
        logger.info("Init CharacteristicsModel")
        self.characteristics = {}
        self.indexes = {}
        self.punctuation = set(string.punctuation)

    def fit(self, data: pl.DataFrame):
        for variantid, characteristic in tqdm(
            zip(
                data["variantid"].to_list(),
                data["characteristic_attributes_mapping"].to_list(),
            ),
            total=data.shape[0],
        ):
            if characteristic is not None:
                self.characteristics[variantid] = {
                    k.lower().strip(): v[0].lower().strip()
                    for k, v in json.loads(characteristic).items()
                }

    def predict(self, data: pl.DataFrame) -> pl.DataFrame:
        return self._batch_predict_match_stats(data)

    def agg_stats(self, stats_dict):
        agg_stats = []
        for s in stats_dict.values():
            agg_stats.append(sum(s) / max(1, len(s)))
        return agg_stats

    def _predict_match_stats(self, variantid1: int, variantid2: int) -> pl.DataFrame:
        variantid1_characteristics = self.characteristics.get(variantid1, {})
        variantid2_characteristics = self.characteristics.get(variantid2, {})

        n_match = 0
        n_mismatch = 0
        stats = {
            "char_lev_full_a": [],
            "char_lev_full_n": [],
            "char_lev_filter_a": [],
            "char_lev_filter_n": [],
            "char_lcp_full_a": [],
            "char_lcp_full_n": [],
            "char_lcp_filter_a": [],
            "char_lcp_filter_n": [],
            "char_lcs_full_a": [],
            "char_lcs_full_n": [],
            "char_lcs_filter_a": [],
            "char_lcs_filter_n": [],
            "char_intersect_l_full_a": [],
            "char_intersect_l_full_n": [],
            "char_intersect_l_filter_a": [],
            "char_intersect_l_filter_n": [],
            "char_intersect_t_full_a": [],
            "char_intersect_t_full_n": [],
            "char_intersect_t_filter_a": [],
            "char_intersect_t_filter_n": [],
        }

        common_characteristics = set(variantid1_characteristics).intersection(
            variantid2_characteristics
        )
        for characteristic in common_characteristics:
            variantid1_value = variantid1_characteristics[characteristic]
            variantid2_value = variantid2_characteristics[characteristic]
            lenght = max(min(len(variantid1_value), len(variantid2_value)), 1)
            lev = levenshtein_distance(variantid1_value, variantid2_value)
            lcp = len(
                longest_common_prefix([variantid1_value, variantid2_value]).strip()
            )
            intersect_letter = len(set(variantid1_value).intersection(variantid2_value))
            intersect_tokens = len(
                set(variantid1_value.split(" ")).intersection(
                    variantid2_value.split(" ")
                )
            )
            lcs = longest_common_subsequence(variantid1_value, variantid2_value)

            if variantid1_value == variantid2_value:
                n_match += 1
            else:
                n_mismatch += 1
                stats["char_lev_filter_a"].append(lev)
                stats["char_lcp_filter_a"].append(lcp)
                stats["char_lcs_filter_a"].append(lcs)
                stats["char_intersect_l_filter_a"].append(intersect_letter)
                stats["char_intersect_t_filter_a"].append(intersect_tokens)
                stats["char_lev_filter_n"].append(lev / lenght)
                stats["char_lcp_filter_n"].append(lcp / lenght)
                stats["char_lcs_filter_n"].append(lcs / lenght)
                stats["char_intersect_l_filter_n"].append(intersect_letter / lenght)
                stats["char_intersect_t_filter_n"].append(intersect_tokens / lenght)

            stats["char_lev_full_a"].append(lev)
            stats["char_lcp_full_a"].append(lcp)
            stats["char_lcs_full_a"].append(lcs)
            stats["char_intersect_l_full_a"].append(intersect_letter)
            stats["char_intersect_t_full_a"].append(intersect_tokens)
            stats["char_lev_full_n"].append(lev / lenght)
            stats["char_lcp_full_n"].append(lcp / lenght)
            stats["char_lcs_full_n"].append(lcs / lenght)
            stats["char_intersect_l_full_n"].append(intersect_letter / lenght)
            stats["char_intersect_t_full_n"].append(intersect_tokens / lenght)

        match_stats = pl.DataFrame(
            [
                [
                    variantid1,
                    variantid2,
                    len(variantid1_characteristics),
                    len(variantid2_characteristics),
                    len(
                        set(variantid1_characteristics).union(
                            variantid2_characteristics
                        )
                    ),
                    len(common_characteristics),
                    n_match,
                    n_mismatch,
                    *self.agg_stats(stats),
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
                "n_mismatch": pl.Int16,
                "char_lev_full_a": pl.Float32,
                "char_lev_full_n": pl.Float32,
                "char_lev_filter_a": pl.Float32,
                "char_lev_filter_n": pl.Float32,
                "char_lcp_full_a": pl.Float32,
                "char_lcp_full_n": pl.Float32,
                "char_lcp_filter_a": pl.Float32,
                "char_lcp_filter_n": pl.Float32,
                "char_lcs_full_a": pl.Float32,
                "char_lcs_full_n": pl.Float32,
                "char_lcs_filter_a": pl.Float32,
                "char_lcs_filter_n": pl.Float32,
                "char_intersect_l_full_a": pl.Float32,
                "char_intersect_l_full_n": pl.Float32,
                "char_intersect_l_filter_a": pl.Float32,
                "char_intersect_l_filter_n": pl.Float32,
                "char_intersect_t_full_a": pl.Float32,
                "char_intersect_t_full_n": pl.Float32,
                "char_intersect_t_filter_a": pl.Float32,
                "char_intersect_t_filter_n": pl.Float32,
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
                (pl.col("n_mismatch") / pl.col("n_left_key")).alias(
                    "left_mismatch_ratio"
                ),
                (pl.col("n_mismatch") / pl.col("n_right_key")).alias(
                    "right_mismatch_ratio"
                ),
                (pl.col("n_mismatch") / pl.col("n_union_key")).alias(
                    "union_mismatch_ratio"
                ),
                (pl.col("n_mismatch") / pl.col("n_intersect_key")).alias(
                    "intersect_mismatch_ratio"
                ),
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
@log_cli
def fit_characteristics_model(data_dir: str = Option(...)):
    data = read_parquet(
        os.path.join(data_dir, "common_data.parquet"),
        columns=["variantid", "characteristic_attributes_mapping"],
    )
    model = CharacteristicsModel()
    model.fit(data)

    write_model(
        os.path.join(data_dir, "characteristics_model.jbl"),
        model,
    )


@cli.command()
@log_cli
def create_characteristics_features(
    data_dir: str = Option(...),
    fold: str = Option(...),
):
    pairs = read_parquet(os.path.join(data_dir, fold, "pairs.parquet"))

    characteristics_model: CharacteristicsModel = read_model(
        os.path.join(data_dir, "characteristics_model.jbl")
    )
    feature = characteristics_model.predict(pairs)
    write_parquet(
        feature, os.path.join(data_dir, fold, "characteristics_features.parquet")
    )


if __name__ == "__main__":
    cli()
