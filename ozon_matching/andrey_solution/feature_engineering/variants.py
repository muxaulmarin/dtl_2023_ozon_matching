import polars as pl


def id_diff(pairs: pl.DataFrame) -> pl.DataFrame:
    return pairs.select(
        [
            pl.col("variantid1"),
            pl.col("variantid2"),
            (pl.col("variantid2") - pl.col("variantid2")).alias("variant_id_diff"),
        ]
    )
