import polars as pl


def generate_features(pairs: pl.DataFrame) -> pl.DataFrame:
    return id_diff(pairs)


def id_diff(pairs: pl.DataFrame) -> pl.DataFrame:
    return pairs.select(
        [
            pl.col("variantid1"),
            pl.col("variantid2"),
            (pl.col("variantid2") - pl.col("variantid2")).alias("variant_id_diff"),
        ]
    )
