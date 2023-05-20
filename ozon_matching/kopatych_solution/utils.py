import os
from typing import List

import joblib
import polars as pl
from loguru import logger


def extract_category_levels(
    data: pl.DataFrame, levels: List[int], category_col="categories"
):
    if not set(levels).intersection([1, 2, 3, 4]):
        raise ValueError("")
    return data.with_columns(
        [
            pl.col(category_col)
            .str.json_path_match(r"$." + str(level))
            .alias(f"category_level_{level}")
            for level in levels
        ]
    )


def write_model(path, model):
    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        logger.info(f"Create dir - {folder}")
        os.mkdir(folder)
    logger.info(f"Save model to {folder}")
    with open(path, "wb") as f:
        joblib.dump(model, f)


def load_model(path, model):
    with open(path, "rb") as f:
        model = joblib.load(f)
    return model
