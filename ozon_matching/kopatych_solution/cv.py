from typing import Iterator, Tuple

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold
from typer import Option, Typer
from ozon_matching.kopatych_solution.utils import (
    extract_category_levels,
    get_and_create_dir,
    log_cli,
    read_json,
    read_model,
    read_parquet,
    write_json,
    write_model,
    write_parquet,
)
import os

cli = Typer()

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

@cli.command()
def dummy():
    pass


@cli.command()
@log_cli
def split_data_for_cv_adversarial_holdout(
    data_dir: str = Option(...),
    n_folds: int = Option(...),
    adversarial_data_path: str = Option(...)
):

    pairs = read_parquet(
        os.path.join(data_dir, f"cv_{n_folds + 1}", "train", "pairs.parquet")
    )
    pairs = pairs.join(
        read_parquet(adversarial_data_path, columns=['variantid1', 'variantid2']),
        on=['variantid1', 'variantid2'],
        how='anti'
    )
    pairs = pairs.with_columns([pl.col("target").cast(pl.Int8)])
    pairs = pairs.with_row_count(name="index")


    for n, train_pairs, test_pairs in stratified_k_fold(
        data=pairs, stratify_col="target", k=int(n_folds)
    ):

        write_parquet(
            train_pairs.select(pl.col(["variantid1", "variantid2", "target"])),
            os.path.join(data_dir, f"cv_{n}", "train", "pairs.parquet"),
        )
        write_parquet(
            test_pairs.select(pl.col(["variantid1", "variantid2"])),
            os.path.join(data_dir, f"cv_{n}", "test", "pairs.parquet"),
        )


if __name__ == "__main__":
    cli()
