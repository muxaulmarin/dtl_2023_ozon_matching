import polars as pl
from ozon_matching.andrey_solution.feature_engineering.utils import (
    fillness,
    list_match,
    max_min,
)
from ozon_matching.andrey_solution.utils import merge_tables


def generate_features(pairs: pl.DataFrame) -> pl.DataFrame:
    return merge_tables(
        max_min(pairs, col_name="color_parsed"),
        fillness(pairs, col_name="color_parsed"),
        list_match(pairs, col_name="color_parsed"),
    )
