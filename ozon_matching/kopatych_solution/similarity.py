from typing import Dict

import numpy as np
import polars as pl
from loguru import logger
from sklearn.preprocessing import normalize


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

        for i, row in enumerate(data.iter_rows()):
            mapping[row[0]] = i
            vectors.append(np.array(row[1]).reshape(1, -1))

        norm_vectors = normalize(np.vstack(vectors))

        self.mapping: Dict[int, int] = mapping
        self.vectors: np.ndarray = norm_vectors

    def get_similarity(self, index_a: int, index_b: int):
        try:
            vector_a = self.vectors[self.mapping[index_a]].reshape(1, -1)
            vector_b = self.vectors[self.mapping[index_b]].reshape(1, -1)
            return np.dot(vector_a, vector_b.T)
        except KeyError:
            logger.info(f"Error for {index_a} and {index_b}")
            return 0
