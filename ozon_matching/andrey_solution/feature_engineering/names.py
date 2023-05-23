import polars as pl
from ozon_matching.andrey_solution.feature_engineering.utils import list_match, max_min
from ozon_matching.andrey_solution.utils import (
    longest_common_prefix,
    longest_common_subsequence,
    merge_tables,
)


def generate_features(pairs: pl.DataFrame) -> pl.DataFrame:
    pairs = pairs.with_columns(
        [
            pl.col("name_1").apply(list).alias("name_list_1"),
            pl.col("name_2").apply(list).alias("name_list_2"),
            pl.col("name_norm_1").apply(list).alias("name_norm_list_1"),
            pl.col("name_norm_2").apply(list).alias("name_norm_list_2"),
        ]
    )
    return merge_tables(
        has_full_match(pairs, col_name="name"),
        has_full_match(pairs, col_name="name_norm"),
        has_full_match(pairs, col_name="name_tokens"),
        has_full_match(pairs, col_name="name_norm_tokens"),
        max_min(pairs, col_name="name_list"),
        max_min(pairs, col_name="name_norm_list"),
        max_min(pairs, col_name="name_tokens"),
        max_min(pairs, col_name="name_norm_tokens"),
        longest_common(pairs, col_name="name_list"),
        longest_common(pairs, col_name="name_norm_list"),
        longest_common(pairs, col_name="name_tokens"),
        longest_common(pairs, col_name="name_norm_tokens"),
        list_match(pairs, col_name="name_list"),
        list_match(pairs, col_name="name_norm_list"),
        list_match(pairs, col_name="name_tokens"),
        list_match(pairs, col_name="name_norm_tokens"),
    )


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
                    lambda cols: longest_common_prefix(
                        cols[f"{col_name}_1"], cols[f"{col_name}_2"]
                    )
                )
                .alias(f"n_{col_name}_lcp"),
                pl.struct([f"{col_name}_1", f"{col_name}_2"])
                .apply(
                    lambda cols: longest_common_subsequence(
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


if __name__ == "__main__":
    from ozon_matching.andrey_solution.preprocessing import preprocess_pairs

    products = preprocess_pairs(
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
