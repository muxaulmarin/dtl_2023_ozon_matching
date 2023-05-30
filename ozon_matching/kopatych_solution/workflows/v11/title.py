import os
import string
from typing import List, Union

import nltk
import polars as pl
from Levenshtein import distance as levenshtein_distance
from loguru import logger
from nltk.corpus import stopwords
from ozon_matching.kopatych_solution.utils import (
    log_cli,
    read_model,
    read_parquet,
    write_model,
    write_parquet,
)
from ozon_matching.kopatych_solution.workflows.v11.nlp import (
    longest_common_prefix,
    longest_common_subsequence,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm
from typer import Option, Typer

cli = Typer()


class TitleModel:
    def __init__(self):
        logger.info("Init TitleModel")
        nltk.download("stopwords")
        nltk.download("punkt")
        self.titles = {}
        self.indexes = {}
        self.remove_punctuation_map = dict(
            (ord(char), None) for char in string.punctuation
        )
        self.stopwords = stopwords.words("russian")
        self.punctuation = set(string.punctuation)
        self.numbers = set("1234567890")

    def preprocess(self, text):
        return nltk.word_tokenize(text.lower().translate(self.remove_punctuation_map))

    def extract_numbers(self, text: str):
        numbers = set()
        number = ""
        for char in text:
            if char in self.numbers:
                number += char
            else:
                if len(number):
                    numbers.add(number)
                    number = ""
        return numbers

    def replace_punctuation(self, input_string):
        for punctuation in self.punctuation:
            input_string = input_string.replace(punctuation, " ")
        return " ".join(input_string.split())

    def _update_titles(self, data):
        for variantid, title in tqdm(
            zip(
                data["variantid"].to_list(),
                data["name"].to_list(),
            ),
            total=data.shape[0],
        ):
            if title is not None:
                self.titles[variantid] = self.replace_punctuation(title.lower().strip())

        for n, variantid in enumerate(self.titles):
            self.indexes[variantid] = n

    def fit(self, data: pl.DataFrame):
        self._update_titles(data)
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(
            tokenizer=self.preprocess, stop_words=self.stopwords
        )
        self.matrix = self.vectorizer.fit_transform(list(self.titles.values()))
        self.matrix = normalize(self.matrix)

    def predict(self, data: pl.DataFrame) -> pl.DataFrame:
        rows = data.select(pl.col(["variantid1", "variantid2"])).iter_rows()
        features = []
        for pair in tqdm(rows, total=data.shape[0]):
            vector = self._predict(*pair)
            features.append(vector)
        features = pl.DataFrame(
            features,
            orient="row",
            schema={
                "variantid1": pl.Int64,
                "variantid2": pl.Int64,
                "tfidf_sim": pl.Float32,
                "title_levenshtein_distance": pl.Float32,
                "title_lcp": pl.Float32,
                "title_lcs": pl.Float32,
                "title_len_1": pl.Int16,
                "title_len_2": pl.Int16,
                "title_delta_max": pl.Float64,
                "title_delta_min": pl.Float64,
                "title_match": pl.Int8,
                "title_len_num_1": pl.Int16,
                "title_len_num_2": pl.Int16,
                "title_num_intersect_max": pl.Float32,
                "title_num_intersect_min": pl.Float32,
                "title_num_iou": pl.Float32,
                "title_intersect_max": pl.Float32,
                "title_intersect_min": pl.Float32,
                "title_iou": pl.Float32,
            },
        )
        return features

    def _predict(self, variantid1: int, variantid2: int) -> List[Union[int, float]]:
        title1 = self.titles[variantid1]
        title2 = self.titles[variantid2]
        sim = (
            self.matrix.getrow(self.indexes[variantid1])
            .dot(self.matrix.getrow(self.indexes[variantid2]).T)
            .A[0, 0]
        )
        numbers1 = self.extract_numbers(title1)
        numbers2 = self.extract_numbers(title2)

        title1_set = set(title1.split(" "))
        title2_set = set(title2.split(" "))

        return [
            variantid1,
            variantid2,
            sim,
            levenshtein_distance(title1, title2),
            len(longest_common_prefix([title1, title2]).strip()),
            longest_common_subsequence(title1, title2),
            len(title1),
            len(title2),
            abs(len(title1) - len(title2)) / max(len(title1), len(title2), 1),
            abs(len(title1) - len(title2)) / min(len(title1), len(title2), 1),
            int(title1 == title2),
            len(numbers1),
            len(numbers2),
            len(numbers1.intersection((numbers2)))
            / max(len(numbers1), len(numbers2), 1),
            len(numbers1.intersection((numbers2)))
            / max(min(len(numbers1), len(numbers2)), 1),
            len(numbers1.intersection((numbers2)))
            / max(len(numbers1.union(numbers2)), 1),
            len(title1_set.intersection((title2_set)))
            / max(len(title1_set), len(title2_set), 1),
            len(title1_set.intersection((title2_set)))
            / max(min(len(title1_set), len(title2_set)), 1),
            len(title1_set.intersection((title2_set)))
            / max(len(title1_set.union(title2_set)), 1),
        ]


@cli.command()
@log_cli
def fit_titles_model(data_dir: str = Option(...)):
    data = read_parquet(
        os.path.join(data_dir, "common_data.parquet"),
        columns=["variantid", "name"],
    )
    model = TitleModel()
    model.fit(data)

    write_model(
        os.path.join(data_dir, "titles_model.jbl"),
        model,
    )


@cli.command()
@log_cli
def create_titles_features(data_dir: str = Option(...), fold: str = Option(...)):
    pairs = read_parquet(os.path.join(data_dir, fold, "pairs.parquet"))

    model: TitleModel = read_model(os.path.join(data_dir, "titles_model.jbl"))
    feature = model.predict(pairs)
    write_parquet(feature, os.path.join(data_dir, fold, "titles_features.parquet"))


if __name__ == "__main__":
    cli()
