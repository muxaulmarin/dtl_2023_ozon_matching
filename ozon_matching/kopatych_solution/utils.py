import json
import os
from functools import wraps
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
    _ = get_and_create_dir(os.path.dirname(path))
    logger.info(f"Save model to {path}")
    with open(path, "wb") as f:
        joblib.dump(model, f)


def read_model(path):
    logger.info(f"Read Model from {path}")
    with open(path, "rb") as f:
        model = joblib.load(f)
    return model


def read_json(path):
    logger.info(f"Read JSON from {path}")
    with open(path) as f:
        return json.load(f)


def write_json(data, path):
    logger.info(f"Write JSON to {path}")
    _ = get_and_create_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f)


def read_parquet(path: str, columns: List[str] = None) -> pl.DataFrame:
    logger.info(f"Read Parquet from {path}")
    data = pl.read_parquet(path, columns=columns)
    logger.info(f"N Rows - {data.shape[0]}, N Cols - {data.shape[1]}")
    return data


def write_parquet(data: pl.DataFrame, path: str):
    logger.info(
        f"Write Parquet to {path}, N Rows - {data.shape[0]}, N Cols - {data.shape[1]}"
    )
    _ = get_and_create_dir(os.path.dirname(path))
    data.write_parquet(path)


def get_and_create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def start_logger(cli: str):
    return "\n" * 2 + (" " * 50) + ("-" * 10) + f"Start {cli}" + ("-" * 10) + "\n" * 2


def end_logger(cli: str):
    return "\n" * 2 + (" " * 50) + ("-" * 10) + f"End {cli}" + ("-" * 10) + "\n" * 2


def log_cli(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(start_logger(func.__name__))
        result = func(*args, **kwargs)
        logger.info(end_logger(func.__name__))
        return result

    return wrapper
