import polars as pl
from ozon_matching.andrey_solution.utils import compose


def preprocess(pairs: pl.DataFrame) -> pl.DataFrame:
    return compose(
        cast_variant_id,
        cast_target,
        remove_index,
    )(pairs)


def cast_variant_id(pairs: pl.DataFrame) -> pl.DataFrame:
    return pairs.with_columns(
        [pl.col("variantid1").cast(pl.UInt32), pl.col("variantid2").cast(pl.UInt32)]
    )


def cast_target(pairs: pl.DataFrame) -> pl.DataFrame:
    if "target" in pairs.columns:
        pairs = pairs.with_columns(pl.col("target").cast(pl.UInt8))
    return pairs


def remove_index(pairs: pl.DataFrame) -> pl.DataFrame:
    return pairs.select(pl.exclude("__index_level_0__"))
