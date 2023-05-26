from itertools import product
import polars as pl
from ozon_matching.andrey_solution.utils import normalize


def convert_to_vw(pairs: pl.DataFrame):
    vw_template = "{target}|n {name} |c {categories} |a {attributes}"
    for row in pairs.iter_rows(named=True):
        if "target" in row:
            target = ("1" if row["target"] == 1 else "-1") + " "
        else:
            target = " "
        yield vw_template.format(
            target=target,
            name=" ".join(
                map(
                    lambda tokens: "_".join(sorted(tokens)), 
                    product(row["name_norm_tokens_1"], row["name_norm_tokens_2"]),
                )
            ),
            categories=" ".join(
                "___".join(map(lambda cat: normalize(cat), sorted([cat_1, cat_2]))) for cat_1, cat_2 in zip(
                    [row["category_level_2_1"], row["category_level_3_1"], row["category_level_4_1"]],
                    [row["category_level_2_2"], row["category_level_3_2"], row["category_level_4_2"]]
                )
            ),
            attributes=" ".join(
                map(
                    lambda attrs: "___".join(map(lambda a: a.replace(":", "_"), sorted(attrs))), 
                    product(row["characteristics_attributes_1"], row["characteristics_attributes_2"]),
                )
            ) 
            if row["characteristics_attributes_1"] is not None and row["characteristics_attributes_2"] is not None
            else ""
        )
