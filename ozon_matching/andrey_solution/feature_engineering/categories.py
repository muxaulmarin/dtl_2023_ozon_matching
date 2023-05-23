from __future__ import annotations

from typing import Iterable

import polars as pl


def match_levels(pairs: pl.DataFrame, levels: list[int] | int) -> pl.DataFrame:
    if not isinstance(levels, Iterable):
        levels = [levels]
    return pairs.select(
        [
            "variantid1",
            "variantid2",
        ]
        + [
            pl.when(
                pl.col(f"category_level_{level}_1")
                == pl.col(f"category_level_{level}_2")
            )
            .then(pl.col(f"category_level_{level}_1"))
            .otherwise(pl.lit("NO MATCH"))
            .alias(f"category_level_{level}")
            for level in levels
        ]
    )
