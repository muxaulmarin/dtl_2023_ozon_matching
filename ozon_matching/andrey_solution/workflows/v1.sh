# preprocessing
python -m ozon_matching.andrey_solution preprocess \
    --data-type pairs \
    --input-path data/raw/test_pairs_wo_target.parquet \
    --input-path data/raw/train_pairs.parquet
python -m ozon_matching.andrey_solution preprocess \
    --data-type products \
    --input-path data/raw/test_data.parquet \
    --input-path data/raw/train_data.parquet

# feature engineering
python -m ozon_matching.andrey_solution generate-features \
    --feature-type categories \
    --feature-type characteristics \
    --feature-type colors \
    --feature-type names \
    --feature-type pictures \
    --feature-type variants\
    --pairs-path data/preprocessed/test_pairs_wo_target.parquet \
    --products-path data/preprocessed/test_data.parquet \
    --pairs-path data/preprocessed/train_pairs.parquet \
    --products-path data/preprocessed/train_data.parquet

# join features
python -m ozon_matching.andrey_solution join-features \
    --features-path data/features/ \
    --pairs-path data/preprocessed/test_pairs_wo_target.parquet \
    --pairs-path data/preprocessed/train_pairs.parquet
