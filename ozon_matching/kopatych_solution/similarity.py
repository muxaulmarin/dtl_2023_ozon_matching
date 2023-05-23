from typing import Dict

import numpy as np
import polars as pl
from loguru import logger
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm


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
