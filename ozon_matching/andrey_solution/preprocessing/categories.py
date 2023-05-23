import polars as pl


def add_cat3_grouped(
    pairs: pl.DataFrame, products: pl.DataFrame, min_cnt: int = 1_000
) -> pl.DataFrame:
    return (
        pairs.join(
            other=products.select(
                [pl.col("variantid").alias("variantid1"), "category_level_3"]
            ),
            how="left",
            on="variantid1",
        )
        .with_columns(pl.count().over("category_level_3").alias("cat3_count"))
        .select(
            [
                "target",
                "variantid1",
                "variantid2",
                pl.when(pl.col("cat3_count") > min_cnt)
                .then(pl.col("category_level_3"))
                .otherwise(pl.lit("rest"))
                .alias("cat3_grouped"),
            ]
        )
    )
