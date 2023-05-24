import os
from functools import reduce
from typing import Callable, Sequence

import polars as pl


def compose(
    *funcs: Callable[[pl.DataFrame], pl.DataFrame]
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    return reduce(lambda f, g: lambda df: g(f(df)), funcs)


def normalize(text: str) -> str:
    text = text.lower()
    chars = []
    for char in text:
        if char.isalnum():
            chars.append(char)
        else:
            chars.append(" ")
    tokens = "".join(chars).split()
    return "_".join(tokens)


def longest_common_prefix(s1: Sequence, s2: Sequence) -> int:
    return len(os.path.commonprefix([s1, s2]))


def longest_common_subsequence(s1: Sequence, s2: Sequence) -> int:
    n1, n2 = len(s1), len(s2)
    lcs = [[0] * (n2 + 1) for _ in range(n1 + 1)]
    for i in range(n1):
        for j in range(n2):
            if s1[i] == s2[j]:
                lcs[i + 1][j + 1] = lcs[i][j] + 1
    return max(max(row) for row in lcs)


def map_products(pairs: pl.DataFrame, products: pl.DataFrame) -> pl.DataFrame:
    return pairs.join(
        other=products.rename(
            {
                col: (col + "_1" if col != "variantid" else "variantid1")
                for col in products.columns
            }
        ),
        how="left",
        on="variantid1",
    ).join(
        other=products.rename(
            {
                col: (col + "_2" if col != "variantid" else "variantid2")
                for col in products.columns
            }
        ),
        how="left",
        on="variantid2",
    )


def merge_tables(
    *tables: pl.DataFrame, on: Sequence[str] = ("variantid1", "variantid2")
) -> pl.DataFrame:
    merged = tables[0]
    for table in tables[1:]:
        merged = merged.join(table, how="outer", on=on)
    return merged
