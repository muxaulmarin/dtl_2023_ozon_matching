# preprocessing
python -m ozon_matching.andrey_solution preprocess \
    --pairs-path data/raw/test_pairs_wo_target.parquet \
    --products-path data/raw/test_data.parquet \
    --pairs-path data/raw/train_pairs.parquet \
    --products-path data/raw/train_data.parquet

# chains generation
python -m ozon_matching.andrey_solution extract-chains \
    --pairs-path data/raw/train_pairs.parquet \
    --products-path data/preprocessed/train_data.parquet

# feature engineering
python -m ozon_matching.andrey_solution generate-features \
    --feature-type categories \
    --feature-type characteristics \
    --feature-type colors \
    --feature-type names \
    --feature-type pictures \
    --feature-type variants \
    --pairs-path data/preprocessed/test_pairs_wo_target.parquet \
    --products-path data/preprocessed/test_data.parquet \
    --pairs-path data/preprocessed/train_pairs.parquet \
    --products-path data/preprocessed/train_data.parquet

# feature engineering for chains
python -m ozon_matching.andrey_solution generate-features \
    --feature-type categories \
    --feature-type characteristics \
    --feature-type colors \
    --feature-type names \
    --feature-type pictures \
    --feature-type variants \
    --pairs-path data/preprocessed/train_chains_pairs.parquet \
    --products-path data/preprocessed/train_data.parquet \
    --output-file train_chains.parquet

# join features
python -m ozon_matching.andrey_solution join-features \
    --features-path data/features/ \
    --pairs-path data/preprocessed/test_pairs_wo_target.parquet \
    --pairs-path data/preprocessed/train_pairs.parquet

# join features to chains
python -m ozon_matching.andrey_solution join-features \
    --features-path data/features/ \
    --pairs-path data/preprocessed/train_chains_pairs.parquet \
    --output-file chains.parquet

# fit catboost
python -m ozon_matching.andrey_solution fit-catboost \
    --train-path data/dataset/train.parquet \
    --experiment-path experiments/v3 \
    --folds-path data/cv_pivot.parquet \
    --chains-path data/dataset/chains.parquet

# prepare submission
python -m ozon_matching.andrey_solution prepare-submission \
    --test-path data/dataset/test.parquet \
    --experiment-path experiments/v3
