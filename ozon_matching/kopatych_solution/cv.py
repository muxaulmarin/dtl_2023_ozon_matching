from typing import Iterator, Tuple

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold


def random_kfold(
    pairs: pl.DataFrame, k=5
) -> Iterator[Tuple[pl.DataFrame, pl.DataFrame]]:
    pairs = pairs.with_columns([(pl.lit(np.arange(pairs.shape[0])) % k).alias("fold")])
    for fold in range(k):
        yield (
            pairs.filter(pl.col("fold") != fold),
            pairs.filter(pl.col("fold") == fold),
        )


def stratified_k_fold(
    data: pl.DataFrame, stratify_col: str, k=5, random_state=13
) -> Iterator[Tuple[int, pl.DataFrame, pl.DataFrame]]:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    cv = skf.split(data, data[stratify_col].to_numpy())
    n = 1
    for train_indecies, test_indecies in cv:
        yield (
            n,
            data.join(
                pl.DataFrame(train_indecies, schema={"index": pl.UInt32}), on=["index"]
            ),
            data.join(
                pl.DataFrame(test_indecies, schema={"index": pl.UInt32}), on=["index"]
            ),
        )
        n += 1
