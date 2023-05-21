from functools import reduce
from typing import Callable

import polars as pl


def compose(
    *funcs: Callable[[pl.DataFrame], pl.DataFrame]
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    return reduce(lambda f, g: lambda df: g(f(df)), funcs)


def normalize(text: str) -> str:
    text = text.lower()
    chars = []
    for char in text:
        if char.isalnum():
            chars.append(char)
        else:
            chars.append(" ")
    tokens = "".join(chars).split()
    return "_".join(tokens)
