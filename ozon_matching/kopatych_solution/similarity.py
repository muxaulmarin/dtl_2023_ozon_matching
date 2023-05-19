from typing import Dict, List

import numpy as np
import polars as pl
from sklearn.preprocessing import normalize


class SimilarityEngine:
    def __init__(self, index_col: str, vector_col: str):
        self.index_col = index_col
        self.vector_col = vector_col

    def fit(self, data: pl.DataFrame):
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

    def _get_indexes(self, indexes: List[int]) -> List[int]:
        return [self.mapping[index] for index in indexes]

    def _get_vectors(self, indexes):
        return self.vectors[indexes]

    def get_similarity(self, indexes_a: List[int], indexes_b: List[int]):
        try:
            return np.dot(
                self._get_vectors(self._get_indexes(indexes_a)),
                self._get_vectors(self._get_indexes(indexes_b)).T,
            )
        except KeyError:
            return 0
