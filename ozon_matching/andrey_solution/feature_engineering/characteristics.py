import polars as pl
from ozon_matching.andrey_solution.feature_engineering.utils import list_match, max_min
from ozon_matching.andrey_solution.utils import merge_tables


def generate_features(pairs: pl.DataFrame) -> pl.DataFrame:
    return merge_tables(
        max_min(pairs, col_name="characteristics"),
        max_min(pairs, col_name="attributes_joined"),
        max_min(pairs, col_name="characteristics_attributes"),
        list_match(pairs, col_name="characteristics"),
        list_match(pairs, col_name="attributes_joined"),
        list_match(pairs, col_name="characteristics_attributes"),
    )
