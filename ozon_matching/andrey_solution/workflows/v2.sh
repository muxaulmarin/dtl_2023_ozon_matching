# preprocessing
python -m ozon_matching.andrey_solution preprocess \
    --pairs-path data/raw/test_pairs_wo_target.parquet \
    --products-path data/raw/test_data.parquet \
    --pairs-path data/raw/train_pairs.parquet \
    --products-path data/raw/train_data.parquet

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

# fit catboost
python -m ozon_matching.andrey_solution fit-catboost \
    --train-path data/dataset/train.parquet \
    --experiment-path experiments/v2 \
    --folds-path data/cv_pivot.parquet

# prepare submission
python -m ozon_matching.andrey_solution prepare-submission \
    --test-path data/dataset/test.parquet \
    --experiment-path experiments/v2
