import os
from typing import Sequence

import polars as pl
from ozon_matching.andrey_solution.feature_engineering.utils import max_min


def has_full_match(pairs: pl.DataFrame, col_name: str) -> pl.DataFrame:
    return pairs.select(
        [
            "variantid1",
            "variantid2",
            (pl.col(f"{col_name}_1") == pl.col(f"{col_name}_2")).alias(
                f"has_full_match_{col_name}"
            ),
        ]
    )


def longest_common(pairs: pl.DataFrame, col_name: str) -> pl.DataFrame:
    return (
        pairs.select(
            [
                "variantid1",
                "variantid2",
                pl.struct([f"{col_name}_1", f"{col_name}_2"])
                .apply(
                    lambda cols: _longest_common_prefix(
                        cols[f"{col_name}_1"], cols[f"{col_name}_2"]
                    )
                )
                .alias(f"n_{col_name}_lcp"),
                pl.struct([f"{col_name}_1", f"{col_name}_2"])
                .apply(
                    lambda cols: _longest_common_subsequence(
                        cols[f"{col_name}_1"], cols[f"{col_name}_2"]
                    )
                )
                .alias(f"n_{col_name}_lcs"),
            ]
        )
        .join(
            other=max_min(pairs, col_name),
            how="left",
            on=["variantid1", "variantid2"],
        )
        .select(
            [
                "variantid1",
                "variantid2",
                pl.col(f"n_{col_name}_lcp").cast(pl.Float32),
                pl.col(f"n_{col_name}_lcs").cast(pl.Float32),
                (pl.col(f"n_{col_name}_lcs") - pl.col(f"n_{col_name}_lcp")).alias(
                    f"n_{col_name}_lcs_diff_lcp"
                ),
                (pl.col(f"n_{col_name}_lcp") / pl.col(f"n_{col_name}_lcs"))
                .fill_nan(0.0)
                .cast(pl.Float32)
                .alias(f"n_{col_name}_lcp_to_lcs"),
                (pl.col(f"n_{col_name}_lcp") / pl.col(f"n_{col_name}_max"))
                .cast(pl.Float32)
                .alias(f"n_{col_name}_lcp_to_max"),
                (pl.col(f"n_{col_name}_lcs") / pl.col(f"n_{col_name}_max"))
                .cast(pl.Float32)
                .alias(f"n_{col_name}_lcs_to_max"),
                (pl.col(f"n_{col_name}_lcp") / pl.col(f"n_{col_name}_min"))
                .cast(pl.Float32)
                .alias(f"n_{col_name}_lcp_to_min"),
                (pl.col(f"n_{col_name}_lcs") / pl.col(f"n_{col_name}_min"))
                .cast(pl.Float32)
                .alias(f"n_{col_name}_lcs_to_min"),
            ]
        )
    )


def _longest_common_prefix(s1: Sequence, s2: Sequence) -> int:
    return len(os.path.commonprefix([s1, s2]))


def _longest_common_subsequence(s1: Sequence, s2: Sequence) -> int:
    n1, n2 = len(s1), len(s2)
    lcs = [[0] * (n2 + 1) for _ in range(n1 + 1)]
    for i in range(n1):
        for j in range(n2):
            if s1[i] == s2[j]:
                lcs[i + 1][j + 1] = lcs[i][j] + 1
    return max(max(row) for row in lcs)


if __name__ == "__main__":
    from ozon_matching.andrey_solution.preprocessing import (
        preprocess_pairs,
        preprocess_products,
    )

    products = preprocess_products(
        (pl.read_parquet("data/raw/test_data.parquet"))
    ).with_columns(pl.col("name").apply(list))
    pairs = preprocess_pairs(pl.read_parquet("data/raw/test_pairs_wo_target.parquet"))
    dataset = pairs.join(
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

    import time

    start = time.perf_counter()
    print(
        longest_common(
            dataset,
            col_name="name",
        )
    )
    stop = time.perf_counter()
    print("took", stop - start, "sec")
