# preprocessing
python -m ozon_matching.andrey_solution preprocess \
    --pairs-path data/raw/test_pairs_wo_target.parquet \
    --products-path data/raw/test_data.parquet

# feature engineering
## stats
python -m ozon_matching.andrey_solution generate-features \
    --feature-type categories \
    --feature-type characteristics \
    --feature-type colors \
    --feature-type names \
    --feature-type pictures \
    --feature-type variants \
    --pairs-path data/preprocessed/test_pairs_wo_target.parquet \
    --products-path data/preprocessed/test_data.parquet
## tfidf
python -m ozon_matching.andrey_solution create-tfidf-similarity-features \
    --col-name characteristics_attributes \
    --col-name characteristics \
    --col-name name_norm_tokens \
    --col-name name_tokens \
    --col-name color_parsed \
    --pairs-path data/preprocessed/test_pairs_wo_target.parquet \
    --output-file test.parquet
## Misha features
bash ozon_matching/kopatych_solution/workflows/v11/dag.sh
## Sergey features
### to be done

# preparing dataset
python -m ozon_matching.andrey_solution join-features \
    --features-path data/features/ \
    --pairs-path data/preprocessed/test_pairs_wo_target.parquet \
    --output-file test.parquet

# preparing submission
python -m ozon_matching.andrey_solution prepare-submission \
    --test-path data/dataset/test.parquet \
    --experiment-path experiments/final
