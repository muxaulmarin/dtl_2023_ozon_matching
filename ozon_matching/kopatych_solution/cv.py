import polars as pl
from typing import Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold

def random_kfold(pairs: pl.DataFrame, k=5) -> Tuple[pl.DataFrame]:
    pairs = pairs.with_columns(
        [
            (pl.lit(np.arange(pairs.shape[0])) % k).alias('fold')
        ]
    )
    for fold in range(k):
        yield (pairs.filter(pl.col('fold') != fold), pairs.filter(pl.col('fold') == fold))
        
        
        
def stratified_k_fold(data: pl.DataFrame, strats: str, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=13)
    cv = skf.split(data, data[strats].to_numpy())
    
    for train_indecies, test_indecies in cv:
        yield (
            data.join(pl.DataFrame(train_indecies, schema={'index': pl.UInt32}),on=['index']), 
            data.join(pl.DataFrame(test_indecies, schema={'index': pl.UInt32}),on=['index'])
        )