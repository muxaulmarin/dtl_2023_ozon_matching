# preprocessing
python -m ozon_matching.andrey_solution preprocess \
    --pairs-path data/raw/train_pairs.parquet \
    --products-path data/raw/train_data.parquet

# feature engineering
## stats
python -m ozon_matching.andrey_solution generate-features \
    --feature-type categories \
    --feature-type characteristics \
    --feature-type colors \
    --feature-type names \
    --feature-type pictures \
    --feature-type variants \
    --pairs-path data/preprocessed/train_pairs.parquet \
    --products-path data/preprocessed/train_data.parquet
## tfidf
python -m ozon_matching.andrey_solution create-tfidf-matrix \
    --col-name characteristics_attributes \
    --col-name characteristics \
    --col-name name_norm_tokens \
    --col-name name_tokens \
    --col-name color_parsed \
    --test-products-path data/preprocessed/test_data.parquet \
    --train-products-path data/preprocessed/train_data.parquet
python -m ozon_matching.andrey_solution create-tfidf-similarity-features \
    --col-name characteristics_attributes \
    --col-name characteristics \
    --col-name name_norm_tokens \
    --col-name name_tokens \
    --col-name color_parsed \
    --pairs-path data/preprocessed/train_pairs.parquet \
    --output-file train.parquet
## Misha features
bash ozon_matching/kopatych_solution/workflows/v11/dag.sh
## Sergey features
python ozon_matching/sergey_solution/multibert.py && python ozon_matching/sergey_solution/multibert_chstic.py && python ozon_matching/sergey_solution/multibert_colors.py


# splits generation
## to be done

# preparing dataset
python -m ozon_matching.andrey_solution join-features \
    --features-path data/features/ \
    --pairs-path data/preprocessed/train_pairs.parquet \
    --output-file train.parquet

# fit model
python -m ozon_matching.andrey_solution fit-catboost \
    --train-path data/dataset/train.parquet \
    --experiment-path experiments/final \
    --folds-path data/cv_pivot.parquet
