from typing import List

import polars as pl


def extract_category_levels(data: pl.DataFrame, levels: List[int], category_col="categories"):
    if not set(levels).intersection([1, 2, 3, 4]):
        raise ValueError("")
    return data.with_columns(
        [
            pl.col(category_col).str.json_path_match(r"$." + str(level)).alias(f"category_level_{level}")
            for level in levels
        ]
    )
