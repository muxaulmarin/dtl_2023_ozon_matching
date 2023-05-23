import numpy as np
import polars as pl
from scipy.spatial.distance import cdist


def pairwise_distance(
    pairs: pl.DataFrame,
    col_name: str,
    metric: str = "cosine",
    batch_size: int = 10_000,
) -> pl.DataFrame:
    col_name_1, col_name_2 = f"{col_name}_1", f"{col_name}_2"
    dims = pl.concat(
        [
            pairs[col_name_1].arr.lengths().unique(),
            pairs[col_name_2].arr.lengths().unique(),
        ]
    ).unique()
    if len(dims) != 1:
        raise ValueError((col_name_1, col_name_1), dims)
    dim = dims[0]

    dist = np.zeros(len(pairs), dtype=np.float32)
    for i, batch in enumerate(
        pairs.select([col_name_1, col_name_2]).iter_slices(batch_size)
    ):
        start = i * batch_size
        stop = start + len(batch)

        vectors1 = np.zeros((len(batch), dim), dtype=np.float32)
        vectors2 = np.zeros((len(batch), dim), dtype=np.float32)
        for j, (v1, v2) in enumerate(batch.iter_rows()):
            vectors1[j] = np.asarray(v1, dtype=np.float32)
            vectors2[j] = np.asarray(v2, dtype=np.float32)

        dist[start:stop] = cdist(vectors1, vectors2, metric=metric).diagonal()

    return pairs.select(
        ["variantid1", "variantid2", pl.lit(dist).alias(f"{col_name}_{metric}_dist")]
    )


def fillness(pairs: pl.DataFrame, col_name: str) -> pl.DataFrame:
    return pairs.select(
        [
            "variantid1",
            "variantid2",
            pl.when(
                pl.col(f"{col_name}_1").is_not_null()
                & pl.col(f"{col_name}_2").is_not_null()
            )
            .then(pl.lit("BOTH"))
            .otherwise(
                pl.when(
                    pl.col(f"{col_name}_1").is_null()
                    & pl.col(f"{col_name}_2").is_null()
                )
                .then(pl.lit("NONE"))
                .otherwise(pl.lit("ONLY ONE"))
            )
            .alias(f"{col_name}_fillness"),
        ]
    )


def max_min(pairs: pl.DataFrame, col_name: str) -> pl.DataFrame:
    return pairs.select(
        [
            "variantid1",
            "variantid2",
            pl.min(
                pl.col(f"{col_name}_1").arr.lengths(),
                pl.col(f"{col_name}_2").arr.lengths(),
            ).alias(f"n_{col_name}_min"),
            pl.max(
                pl.col(f"{col_name}_1").arr.lengths(),
                pl.col(f"{col_name}_2").arr.lengths(),
            ).alias(f"n_{col_name}_max"),
        ]
    ).select(
        [
            "variantid1",
            "variantid2",
            f"n_{col_name}_min",
            f"n_{col_name}_max",
            (pl.col(f"n_{col_name}_max") - pl.col(f"n_{col_name}_min")).alias(
                f"n_{col_name}_diff"
            ),
            (pl.col(f"n_{col_name}_min") / pl.col(f"n_{col_name}_max"))
            .fill_nan(1.0)
            .cast(pl.Float32)
            .alias(f"n_{col_name}_min_to_max"),
        ]
    )


def list_match(pairs: pl.DataFrame, col_name: str) -> pl.DataFrame:
    return (
        pairs.select(
            [
                "variantid1",
                "variantid2",
                pl.struct([f"{col_name}_1", f"{col_name}_2"])
                .apply(
                    lambda row: len(
                        set(row[f"{col_name}_1"]) & set(row[f"{col_name}_2"])
                    )
                    if row[f"{col_name}_1"] is not None
                    and row[f"{col_name}_2"] is not None
                    else None
                )
                .alias(f"n_{col_name}_matched"),
                pl.struct([f"{col_name}_1", f"{col_name}_2"])
                .apply(
                    lambda row: len(
                        set(row[f"{col_name}_1"]) | set(row[f"{col_name}_2"])
                    )
                    if row[f"{col_name}_1"] is not None
                    and row[f"{col_name}_2"] is not None
                    else None
                )
                .alias(f"n_{col_name}_union"),
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
                f"n_{col_name}_matched",
                f"n_{col_name}_union",
                (pl.col(f"n_{col_name}_union") - pl.col(f"n_{col_name}_matched")).alias(
                    f"n_{col_name}_union_diff_matched"
                ),
                (pl.col(f"n_{col_name}_matched") / pl.col(f"n_{col_name}_union"))
                .fill_nan(1.0)
                .cast(pl.Float32)
                .alias(f"n_{col_name}_matched_to_union"),
                (pl.col(f"n_{col_name}_matched") / pl.col(f"n_{col_name}_max"))
                .fill_nan(1.0)
                .cast(pl.Float32)
                .alias(f"n_{col_name}_matched_to_max"),
                (pl.col(f"n_{col_name}_union") / pl.col(f"n_{col_name}_max"))
                .fill_nan(1.0)
                .cast(pl.Float32)
                .alias(f"n_{col_name}_union_to_max"),
                (pl.col(f"n_{col_name}_matched") / pl.col(f"n_{col_name}_min"))
                .fill_nan(1.0)
                .cast(pl.Float32)
                .alias(f"n_{col_name}_matched_to_min"),
                (pl.col(f"n_{col_name}_union") / pl.col(f"n_{col_name}_min"))
                .fill_nan(1.0)
                .cast(pl.Float32)
                .alias(f"n_{col_name}_union_to_min"),
            ]
        )
    )


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


if __name__ == "__main__":
    from ozon_matching.andrey_solution.preprocessing import (
        preprocess_pairs,
        preprocess_products,
    )

    products = preprocess_products((pl.read_parquet("data/raw/test_data.parquet")))
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
    print(list_match(dataset, col_name="characteristics_attributes"))
    stop = time.perf_counter()
    print("took", stop - start, "sec")
