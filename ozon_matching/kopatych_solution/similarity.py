from typing import Dict

import numpy as np
import polars as pl
from loguru import logger
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm
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

class SimilarityEngine:
    def __init__(self, index_col: str, vector_col: str):
        logger.info(f"Init SimilarityEngine for {vector_col}")
        self.index_col = index_col
        self.vector_col = vector_col

    def fit(self, data: pl.DataFrame):
        logger.info(f"Fit SimilarityEngine for {self.vector_col}")
        if self.index_col not in data.columns or self.vector_col not in data.columns:
            raise ValueError("")

        data = data.select(pl.col([self.index_col, self.vector_col]))

        mapping = {}
        vectors = []

        for i, row in tqdm(enumerate(data.iter_rows()), total=data.shape[0]):
            mapping[row[0]] = i
            vectors.append(np.array(row[1]).reshape(1, -1))

        norm_vectors = normalize(np.vstack(vectors))

        self.mapping: Dict[int, int] = mapping
        self.vectors: np.ndarray = norm_vectors

    def _get_similarity(self, index_a: int, index_b: int):
        try:
            vector_a = self.vectors[self.mapping[index_a]].reshape(1, -1)
            vector_b = self.vectors[self.mapping[index_b]].reshape(1, -1)
            return np.dot(vector_a, vector_b.T)
        except KeyError:
            logger.info(f"Error for {index_a} and {index_b}")
            return 0

    def predict(self, data: pl.DataFrame):
        batch = []
        rows = data.select(pl.col(["variantid1", "variantid2"])).iter_rows()
        for pair in tqdm(rows, total=data.shape[0]):
            batch.append(self._get_similarity(*pair))
        feature = pl.DataFrame(
            batch, schema={f"cosine_similarity_{self.vector_col}": pl.Float32}
        )
        feature = feature.with_columns(
            [
                data["variantid1"],
                data["variantid2"],
            ]
        )
        return feature


@cli.command()
def dummy():
    pass

@cli.command()
@log_cli
def fit_similarity_engine(data_dir: str = Option(...), vector_col: str = Option(...)):

    data = read_parquet(
        os.path.join(data_dir, "union_data.parquet"),
        columns=["variantid", vector_col],
    )

    similarity_engine = SimilarityEngine(index_col="variantid", vector_col=vector_col)
    similarity_engine.fit(data)
    write_model(
        os.path.join(data_dir, f"similarity_engine_{vector_col}.jbl"),
        similarity_engine,
    )


@cli.command()
@log_cli
def create_similarity_features(
    data_dir: str = Option(...),
    fold_type: str = Option(...),
    n_folds: int = Option(...),
):
    pairs = read_parquet(os.path.join(data_dir, fold_type, "pairs.parquet"))

    pic_similarity_engine: SimilarityEngine = read_model(
        os.path.join(
            os.path.dirname(data_dir),
            f"cv_{n_folds + 1}",
            "similarity_engine_main_pic_embeddings_resnet_v1.jbl",
        )
    )
    name_similarity_engine: SimilarityEngine = read_model(
        os.path.join(
            os.path.dirname(data_dir),
            f"cv_{n_folds + 1}",
            "similarity_engine_name_bert_64.jbl",
        )
    )
    for engine in [pic_similarity_engine, name_similarity_engine]:
        feature = engine.predict(pairs)
        write_parquet(
            feature,
            os.path.join(
                data_dir, fold_type, f"similarity_features_{engine.vector_col}.parquet"
            ),
        )


if __name__ == "__main__":
    cli()
