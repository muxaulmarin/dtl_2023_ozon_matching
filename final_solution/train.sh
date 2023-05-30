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
wget https://storage.yandexcloud.net/lcr/models_weights.tar?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=YCAJE3aFMqzbzkXdc3Q23murc%2F20230530%2Fru-central1%2Fs3%2Faws4_request&X-Amz-Date=20230530T193615Z&X-Amz-Expires=864000&X-Amz-Signature=0449107A1F498B238C9CDA2DEB42225B54F3567D8F61B2AF6221EF3C77DE9B4A&X-Amz-SignedHeaders=host
tar -xf models_weights.tarmodels_weights.tar
mkdir data/preprocessed/nn
python multibert.py && python multibert_chstic.py && python multibert_colors.py


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
