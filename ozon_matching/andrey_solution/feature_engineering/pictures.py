import numpy as np
import polars as pl
from ozon_matching.andrey_solution.feature_engineering.utils import (
    fillness,
    max_min,
    pairwise_distance,
)
from ozon_matching.andrey_solution.utils import merge_tables
from scipy.spatial.distance import cdist


def generate_features(pairs: pl.DataFrame) -> pl.DataFrame:
    return merge_tables(
        max_min(pairs, col_name="pic_embeddings_resnet_v1"),
        fillness(pairs, col_name="pic_embeddings_resnet_v1"),
        pairwise_distance(
            pairs, col_name="main_pic_embeddings_resnet_v1", metric="cosine"
        ),
        distance_from_main_to_others(pairs, metric="cosine"),
    )


def distance_from_main_to_others(
    pairs: pl.DataFrame,
    metric: str = "cosine",
) -> pl.DataFrame:
    dist = np.zeros((len(pairs), 4), dtype=np.float32)
    for i, (main1, main2, others1, others2) in enumerate(
        pairs.select(
            [
                "main_pic_embeddings_resnet_v1_1",
                "main_pic_embeddings_resnet_v1_2",
                "pic_embeddings_resnet_v1_1",
                "pic_embeddings_resnet_v1_2",
            ]
        ).iter_rows()
    ):
        if others1 is None and others2 is None:
            dist[i] = np.array([np.nan, np.nan, np.nan, np.nan])
            continue

        main1_emb = np.array(main1, dtype=np.float32).reshape((1, -1))
        main2_emb = np.array(main2, dtype=np.float32).reshape((1, -1))
        others1_emb = (
            np.array(others1, dtype=np.float32) if others1 is not None else None
        )
        others2_emb = (
            np.array(others2, dtype=np.float32) if others2 is not None else None
        )

        dists = np.concatenate(
            [
                cdist(main1_emb, others2_emb, metric=metric)[0]
                if others2_emb is not None
                else np.array([]),
                cdist(main2_emb, others1_emb, metric=metric)[0]
                if others1_emb is not None
                else np.array([]),
            ]
        )
        dist[i] = np.array(
            [np.mean(dists), np.median(dists), np.min(dists), np.max(dists)],
            dtype=np.float32,
        )
    return pairs.select(
        [
            "variantid1",
            "variantid2",
            pl.lit(dist[:, 0]).alias("pic_embeddings_main_to_others_dist_mean"),
            pl.lit(dist[:, 1]).alias("pic_embeddings_main_to_others_dist_median"),
            pl.lit(dist[:, 2]).alias("pic_embeddings_main_to_others_dist_min"),
            pl.lit(dist[:, 3]).alias("pic_embeddings_main_to_others_dist_max"),
        ]
    )


if __name__ == "__main__":

    products = pl.read_parquet("data/preprocessed/test_data.parquet")
    pairs = pl.read_parquet("data/preprocessed/test_pairs_wo_target.parquet")
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
        )
    )
    stop = time.perf_counter()
    print("took", stop - start, "sec")
