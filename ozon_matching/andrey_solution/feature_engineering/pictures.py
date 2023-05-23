import numpy as np
import polars as pl
from scipy.spatial.distance import cdist


def distance_from_main_to_others(
    pairs: pl.DataFrame,
    col_main: str,
    col_others: str,
    metric: str = "cosine",
) -> pl.DataFrame:
    dist = np.zeros((len(pairs), 4), dtype=np.float32)
    for i, (main, others) in enumerate(
        pairs.select([col_main, col_others]).iter_rows()
    ):
        if main is None or others is None:
            dist[i] = np.array([np.nan, np.nan, np.nan, np.nan])
            continue
        main_emb = np.array(main, dtype=np.float32).reshape((1, -1))
        others_emb = np.array(others, dtype=np.float32)
        dists = cdist(main_emb, others_emb, metric=metric)
        dist[i] = np.array(
            [np.mean(dists), np.median(dists), np.min(dists), np.max(dists)],
            dtype=np.float32,
        )
    return pairs.select(
        [
            "variantid1",
            "variantid2",
            pl.lit(dist[:, 0]).alias(f"{col_main}_{col_others}_dist_mean"),
            pl.lit(dist[:, 1]).alias(f"{col_main}_{col_others}_dist_median"),
            pl.lit(dist[:, 2]).alias(f"{col_main}_{col_others}_dist_min"),
            pl.lit(dist[:, 3]).alias(f"{col_main}_{col_others}_dist_max"),
        ]
    )


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
    print(
        distance_from_main_to_others(
            dataset,
            col_main="main_pic_embeddings_resnet_v1_1",
            col_others="pic_embeddings_resnet_v1_2",
        )
    )
    stop = time.perf_counter()
    print("took", stop - start, "sec")
