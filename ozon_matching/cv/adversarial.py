import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from loguru import logger
from ozon_matching.kopatych_solution.utils import (
    extract_category_levels,
    log_cli,
    read_parquet,
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
from typer import Option, Typer

cli = Typer()


@cli.command()
def dummy():
    pass


@cli.command()
@log_cli
def adversarial_check(
    train_path: str = Option(...),
    test_path: str = Option(...),
    train_data_path: str = Option(...),
    test_data_path: str = Option(...),
):

    data = pl.concat(
        [
            read_parquet(train_data_path, columns=["variantid", "categories"]),
            read_parquet(test_data_path, columns=["variantid", "categories"]),
        ]
    )
    data = data.unique(subset=["variantid"])
    data = extract_category_levels(data, [3, 4])
    data = data.select(pl.col(["variantid", "category_level_3", "category_level_4"]))
    data = data.join(
        (
            data.select(pl.col("category_level_3"))
            .unique()
            .with_row_count(name="category_level_3_id")
        ),
        on=["category_level_3"],
    )
    data = data.join(
        (
            data.select(pl.col("category_level_4"))
            .unique()
            .with_row_count(name="category_level_4_id")
        ),
        on=["category_level_4"],
    )
    data = data.drop(["category_level_3", "category_level_4"])

    train = read_parquet(train_path)
    train = train.select(pl.col(["variantid1", "variantid2"])).with_columns(
        [pl.lit(1).cast(pl.Int8).alias("is_train")]
    )

    test = read_parquet(test_path)
    test = test.select(pl.col(["variantid1", "variantid2"])).with_columns(
        [pl.lit(0).cast(pl.Int8).alias("is_train")]
    )

    train = train.join(test, on=["variantid1", "variantid2"], how="anti")

    pairs = pl.concat([train, test])

    pairs = (
        pairs.join(
            data.rename(
                {
                    "variantid": "variantid1",
                    "category_level_3_id": "category_level_3_id_1",
                    "category_level_4_id": "category_level_4_id_1",
                }
            ),
            on=["variantid1"],
        )
        .join(
            data.rename(
                {
                    "variantid": "variantid2",
                    "category_level_3_id": "category_level_3_id_2",
                    "category_level_4_id": "category_level_4_id_2",
                }
            ),
            on=["variantid2"],
        )
        .to_pandas()
    )

    cv1 = StratifiedKFold(n_splits=3, random_state=13, shuffle=True)
    cv2 = StratifiedKFold(n_splits=5, random_state=13, shuffle=True)

    X = pairs[
        [
            "category_level_3_id_1",
            "category_level_4_id_1",
            "category_level_3_id_2",
            "category_level_4_id_2",
        ]
    ].values
    y = pairs["is_train"].values

    scores = []
    for cv_index, holdout_index in tqdm(cv1.split(X, y)):
        X_cv, X_holdout = X[cv_index], X[holdout_index]
        y_cv, y_holdout = y[cv_index], y[holdout_index]
        holdout_score = []
        for train_index, valid_index in tqdm(cv2.split(X_cv, y_cv)):
            X_train, X_valid = X_cv[train_index], X_cv[valid_index]
            y_train, y_valid = y_cv[train_index], y_cv[valid_index]

            model = LGBMClassifier(n_estimators=5000, silent=True)
            model.fit(
                X=X_train,
                y=y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric=["auc"],
                categorical_feature=[0, 1, 2, 3],
                early_stopping_rounds=50,
                verbose=-1,
            )
            holdout_score.append(
                roc_auc_score(y_holdout, model.predict_proba(X_holdout)[:, 1])
            )
        scores.append(np.mean(holdout_score))

    logger.info(f"Adversarial data - {test_path}")
    logger.info(f"Adversarial check. ROCAUC={np.mean(scores)}")


if __name__ == "__main__":
    cli()
