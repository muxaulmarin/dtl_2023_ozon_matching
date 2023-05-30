import itertools as it

import numpy as np
import pandas as pd
import polars as pl
from scipy import sparse as sp
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm


def fit_tfidf(
    products_train: pl.DataFrame,
    products_test: pl.DataFrame,
    col_name: str,
) -> tuple[sp.csr_matrix, dict[int, int], dict[str, int]]:
    uniq_values = set(products_train[col_name].explode().unique().drop_nulls()) | set(
        products_test[col_name].explode().unique().drop_nulls()
    )
    value_to_index = dict(zip(uniq_values, range(len(uniq_values))))

    uniq_variants = set(products_train["variantid"].unique()) | set(
        products_test["variantid"].unique()
    )
    variant_id_to_index = dict(zip(uniq_variants, range(len(uniq_variants))))
    bow = sp.dok_matrix((len(uniq_variants), len(uniq_values)), dtype=np.uint8)
    for variant_id, col_values in tqdm(
        it.chain(
            products_train.select(["variantid", col_name]).iter_rows(),
            products_test.select(["variantid", col_name]).iter_rows(),
        ),
        total=len(products_train) + len(products_test),
    ):
        if col_values is None:
            continue
        for col_value in col_values:
            bow[variant_id_to_index[variant_id], value_to_index[col_value]] = 1

    tfidf = TfidfTransformer().fit_transform(bow.tocsr()).astype(np.float32)
    return tfidf, variant_id_to_index, value_to_index


def calc_similarity(
    pairs: pl.DataFrame,
    col_name: str,
    tfidf_matrix: sp.spmatrix,
    variant_id_to_index: dict[int, int],
    metric: str = "cosine",
    batch_size: int = 10,
) -> pl.DataFrame:
    variant_id_to_index_map = pl.from_pandas(
        pd.Series(variant_id_to_index)
        .reset_index(drop=False)
        .rename(columns={"index": "variantid", 0: "tfidf_index"})
    ).with_columns(
        [
            pl.col("variantid").cast(pl.UInt32),
            pl.col("tfidf_index").cast(pl.UInt32),
        ]
    )
    pairs = (
        pairs.select(["variantid1", "variantid2"])
        .join(
            other=variant_id_to_index_map.rename(
                {"variantid": "variantid1", "tfidf_index": "tfidf_index_1"}
            ),
            how="left",
            on="variantid1",
        )
        .join(
            other=variant_id_to_index_map.rename(
                {"variantid": "variantid2", "tfidf_index": "tfidf_index_2"}
            ),
            how="left",
            on="variantid2",
        )
    )
    dist = np.zeros(len(pairs), dtype=np.float32)
    for i, (tfidf_index_1, tfidf_index_2) in enumerate(
        pairs.select(["tfidf_index_1", "tfidf_index_2"]).iter_rows()
    ):
        dist[i] = cdist(
            tfidf_matrix[tfidf_index_1].todense(),
            tfidf_matrix[tfidf_index_2].todense(),
            metric=metric,
        ).diagonal()

    return pairs.select(
        [
            "variantid1",
            "variantid2",
            pl.lit(dist).alias(f"{col_name}_tfidf_{metric}_dist"),
        ]
    )
