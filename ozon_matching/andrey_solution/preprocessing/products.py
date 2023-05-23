from functools import reduce

import polars as pl
from ozon_matching.andrey_solution.utils import compose, normalize


def preprocess(products: pl.DataFrame) -> pl.DataFrame:
    return compose(
        cast_variant_id,
        normalize_names,
        extract_categories,
        squeeze_main_pic_embeddings,
        normalize_characteristic_attributes,
    )(products)


def normalize_names(products: pl.DataFrame) -> pl.DataFrame:
    return products.with_columns(
        [
            pl.col("name")
            .apply(lambda name: name.strip().lower().split())
            .alias("name_tokens"),
            pl.col("name").apply(normalize).alias("name_norm"),
        ]
    ).with_columns(
        [
            pl.col("name_tokens").arr.join(" ").alias("name"),
            pl.col("name_norm")
            .apply(lambda name: name.split("_"))
            .alias("name_norm_tokens"),
        ]
    )


def cast_variant_id(products: pl.DataFrame) -> pl.DataFrame:
    return products.with_columns(pl.col("variantid").cast(pl.UInt32))


def extract_categories(products: pl.DataFrame) -> pl.DataFrame:
    return (
        products.with_columns(pl.col("categories").str.json_extract())
        .with_columns(
            [
                pl.col("categories").struct.field(str(i)).alias(f"category_level_{i}")
                for i in range(1, 5)
            ]
        )
        .select(pl.exclude("categories"))
    )


def squeeze_main_pic_embeddings(products: pl.DataFrame) -> pl.DataFrame:
    return products.with_columns(pl.col("main_pic_embeddings_resnet_v1").arr.first())


def normalize_characteristic_attributes(products: pl.DataFrame) -> pl.DataFrame:
    return (
        products.join(
            other=products.select(
                [
                    "variantid",
                    pl.col("characteristic_attributes_mapping")
                    .apply(
                        lambda char_attrs_map: list(
                            map(normalize, eval(char_attrs_map).keys())
                        )
                        if char_attrs_map is not None
                        else char_attrs_map
                    )
                    .alias("characteristics"),
                ]
            ),
            how="left",
            on="variantid",
        )
        .join(
            other=products.select(
                [
                    "variantid",
                    pl.col("characteristic_attributes_mapping")
                    .apply(
                        lambda char_attrs_map: list(
                            map(
                                lambda attr_list: list(map(normalize, attr_list)),
                                eval(char_attrs_map).values(),
                            )
                        )
                        if char_attrs_map is not None
                        else char_attrs_map
                    )
                    .alias("attributes"),
                ]
            ),
            how="left",
            on="variantid",
        )
        .with_columns(
            [
                pl.col("attributes")
                .apply(
                    lambda attrs: list(
                        map(lambda attr_list: "+".join(sorted(attr_list)), attrs)
                    )
                    if attrs is not None
                    else attrs
                )
                .alias("attributes_joined"),
                pl.struct(["characteristics", "attributes"])
                .apply(
                    lambda char_attrs: reduce(
                        lambda char_attrs, char_attr_list: char_attrs
                        + list(
                            map(
                                lambda attr: f"{char_attr_list[0]}:{attr}",
                                char_attr_list[1],
                            )
                        ),
                        zip(char_attrs["characteristics"], char_attrs["attributes"]),
                        [],
                    )
                    if char_attrs["characteristics"] is not None
                    and char_attrs["attributes"] is not None
                    else char_attrs
                )
                .alias("characteristics_attributes"),
            ]
        )
        .select(pl.exclude("characteristic_attributes_mapping"))
    )


if __name__ == "__main__":
    test_products = pl.read_parquet("data/raw/test_data.parquet")
    import time

    start = time.perf_counter()
    print(normalize_names(test_products))
    stop = time.perf_counter()
    print("took", stop - start, "sec")
