import os

import numpy as np
import polars as pl
from ozon_matching.kopatych_solution.colors import ColorModel
from ozon_matching.kopatych_solution.utils import (
    log_cli,
    read_model,
    read_parquet,
    write_parquet,
)
from ozon_matching.kopatych_solution.workflows.v11.characteristics import (
    CharacteristicsModel,
)
from ozon_matching.kopatych_solution.workflows.v11.nlp import FilterToken
from typer import Option, Typer

cli = Typer()


@cli.command()
def dummy():
    pass


@cli.command()
@log_cli
def create_brand_features(data_dir: str = Option(...), fold: str = Option(...)):
    pairs = read_parquet(
        os.path.join(data_dir, fold, "pairs.parquet"),
        columns=["variantid1", "variantid2"],
    )
    characteristics_model: CharacteristicsModel = read_model(
        os.path.join(data_dir, "characteristics_model.jbl")
    )

    brand_feature = []
    for v1, v2 in pairs.iter_rows():
        c1 = characteristics_model.characteristics.get(v1, {})
        c2 = characteristics_model.characteristics.get(v2, {})
        if "бренд" in c1 and "бренд" in c2:
            b1 = c1["бренд"]
            b2 = c2["бренд"]
            if b1 == b2 and b1 != "нет бренда":
                f = 1
            elif b1 != b2 and b1 != "нет бренда" and b2 != "нет бренда":
                f = 2
            elif (b1 == "нет бренда" and b2 != "нет бренда") or (
                b1 != "нет бренда" and b2 == "нет бренда"
            ):
                f = 3
            else:
                f = 4
            brand_feature.append([v1, v2, f])

    brand_feature = pl.DataFrame(
        brand_feature,
        orient="row",
        schema={
            "variantid1": pl.Int64,
            "variantid2": pl.Int64,
            "brand_cat_feature": pl.Int8,
        },
    )
    brand_feature = pairs.join(
        brand_feature, on=["variantid1", "variantid2"], how="left"
    )
    brand_feature = brand_feature.fill_null(-1)
    write_parquet(brand_feature, os.path.join(data_dir, fold, "brand_features.parquet"))


@cli.command()
@log_cli
def create_compatible_devices_feature(
    data_dir: str = Option(...), fold: str = Option(...)
):
    pairs = read_parquet(
        os.path.join(data_dir, fold, "pairs.parquet"),
        columns=["variantid1", "variantid2"],
    )
    characteristics_model: CharacteristicsModel = read_model(
        os.path.join(data_dir, "characteristics_model.jbl")
    )
    brands = set()
    for _, char in characteristics_model.characteristics.items():
        if "бренд" in char:
            brands.add(char["бренд"].lower())

    token_filter = FilterToken(brands)

    compatible_devices_feature = []
    attr = "список совместимых устройств"
    for v1, v2 in pairs.iter_rows():
        c1 = characteristics_model.characteristics.get(v1, {})
        c2 = characteristics_model.characteristics.get(v2, {})
        if attr in c1 and attr in c2:
            cd1 = c1[attr]
            cd2 = c2[attr]
            cd1 = token_filter.get_compatible_devices(cd1)
            cd2 = token_filter.get_compatible_devices(cd2)
            compatible_devices_feature.append(
                [v1, v2, len(cd1), len(cd2), len(cd1.intersection(cd2))]
            )

    compatible_devices_feature = pl.DataFrame(
        compatible_devices_feature,
        orient="row",
        schema={
            "variantid1": pl.Int64,
            "variantid2": pl.Int64,
            "len_compatible_devices_1": pl.Int16,
            "len_compatible_devices_2": pl.Int16,
            "len_intersection_compatible_devices": pl.Int16,
        },
    )
    compatible_devices_feature = compatible_devices_feature.with_columns(
        [
            (
                pl.col("len_intersection_compatible_devices")
                / pl.col("len_compatible_devices_1")
            ).alias("compatible_devices_1_ratio"),
            (
                pl.col("len_intersection_compatible_devices")
                / pl.col("len_compatible_devices_2")
            ).alias("compatible_devices_2_ratio"),
        ]
    )
    compatible_devices_feature = pairs.join(
        compatible_devices_feature, on=["variantid1", "variantid2"], how="left"
    )
    compatible_devices_feature = compatible_devices_feature.fill_null(-1)
    write_parquet(
        compatible_devices_feature,
        os.path.join(data_dir, fold, "compatible_devices_feature.parquet"),
    )


@cli.command()
@log_cli
def create_color_feature(data_dir: str = Option(...), fold: str = Option(...)):
    pairs = read_parquet(
        os.path.join(data_dir, fold, "pairs.parquet"),
        columns=["variantid1", "variantid2"],
    )
    data = read_parquet(
        os.path.join(data_dir, "common_data.parquet"),
        columns=["variantid", "color_parsed"],
    )

    pairs = pairs.join(
        data.rename({"variantid": "variantid1", "color_parsed": "color_parsed1"}),
        on=["variantid1"],
    ).join(
        data.rename({"variantid": "variantid2", "color_parsed": "color_parsed2"}),
        on=["variantid2"],
    )

    cm = ColorModel()

    color_feature = []
    for row in pairs.iter_rows():
        c1, c2 = row[-2], row[-1]
        if c1 is None or c2 is None:
            continue
        m = np.zeros((len(c1), len(c2)))
        for i in range(len(c1)):
            for j in range(len(c2)):
                m[i, j] = cm.distance(c1[i], c2[j])
        color_feature.append([row[0], row[1], np.mean(m), np.max(m), np.min(m)])

    color_feature = pl.DataFrame(
        color_feature,
        orient="row",
        schema={
            "variantid1": pl.Int64,
            "variantid2": pl.Int64,
            "color_feature_mean_distance": pl.Float64,
            "color_feature_max_distance": pl.Float64,
            "color_feature_min_distance": pl.Float64,
        },
    )
    color_feature = pairs.join(
        color_feature, on=["variantid1", "variantid2"], how="left"
    )
    color_feature = color_feature.fill_null(-1)
    write_parquet(color_feature, os.path.join(data_dir, fold, "color_feature.parquet"))


if __name__ == "__main__":
    cli()
