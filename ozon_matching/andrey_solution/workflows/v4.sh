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


# prepare vw dataset for train
python -m ozon_matching.andrey_solution prepare-vw-dataset \
    --pairs-path data/preprocessed/train_pairs.parquet \
    --products-path data/preprocessed/train_data.parquet \
    --output-file train.vw
# prepare vw dataset for chains
python -m ozon_matching.andrey_solution prepare-vw-dataset \
    --pairs-path data/preprocessed/train_chains_pairs.parquet \
    --products-path data/preprocessed/train_data.parquet \
    --output-file chains.vw
# merge train with chains
cat data/vw-dataset/train.vw data/vw-dataset/chains.vw > data/vw-dataset/train_with_chains.vw
# prepare vw dataset for test
python -m ozon_matching.andrey_solution prepare-vw-dataset \
    --pairs-path data/preprocessed/test_pairs_wo_target.parquet \
    --products-path data/preprocessed/test_data.parquet \
    --output-file test.vw

# fit
vw data/vw-dataset/train_with_chains.vw \
    --final_regressor data/vw-dataset/model.bin \
    --loss_function=logistic \
    --link=logistic \
    --bit_precision 28 \
    --passes 100 \
    --interactions ca \
    --holdout_off \
    --cache -k

# inference on train
vw data/vw-dataset/train_with_chains.vw \
    --initial_regressor data/vw-dataset/model.bin \
    --testonly \
    --predictions data/vw-dataset/train_with_chains_pred.txt
# calc metrics
python -m ozon_matching.andrey_solution calc-metric-train-vw \
    --pairs-path data/preprocessed/train_pairs.parquet \
    --pairs-path data/preprocessed/train_chains_pairs.parquet \
    --predictions-path data/vw-dataset/train_with_chains_pred.txt

# inference on test
vw data/vw-dataset/test.vw \
    --initial_regressor data/vw-dataset/model.bin \
    --testonly \
    --predictions data/vw-dataset/test_pred.txt
# prepare submission
python -m ozon_matching.andrey_solution prepare-submission-vw \
    --pairs-path data/preprocessed/test_pairs_wo_target.parquet \
    --predictions-path data/vw-dataset/test_pred.txt
