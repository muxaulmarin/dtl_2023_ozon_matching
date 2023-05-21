import polars as pl


def read_products(path: str) -> pl.DataFrame:
    df = pl.read_parquet(path)
    df = df.with_columns(pl.col("variantid").cast(pl.UInt32))
    df = (
        df.with_columns(pl.col("categories").apply(eval))
        .with_columns(
            [
                pl.col("categories").struct.field(str(i)).alias(f"category_level_{i}")
                for i in range(1, 5)
            ]
        )
        .select(pl.exclude("categories"))
    )
    df = df.with_columns(pl.col("main_pic_embeddings_resnet_v1").arr.first())
    df = df.with_columns(
        [
            pl.col("characteristic_attributes_mapping")
            .apply(
                lambda attrs: list(eval(attrs).keys())
                if isinstance(attrs, str)
                else attrs
            )
            .alias("characteristics"),
            pl.col("characteristic_attributes_mapping")
            .apply(
                lambda attrs: list(eval(attrs).values())
                if isinstance(attrs, str)
                else attrs
            )
            .alias("attributes"),
        ]
    ).select(pl.exclude("characteristic_attributes_mapping"))
    return df


def read_pairs(path: str) -> pl.DataFrame:
    df = pl.read_parquet(path)
    df = df.with_columns(
        [pl.col("variantid1").cast(pl.UInt32), pl.col("variantid2").cast(pl.UInt32)]
    )
    if "target" in df.columns:
        df = df.with_columns(pl.col("target").cast(pl.UInt8))
    return df.select(pl.exclude("__index_level_0__"))
