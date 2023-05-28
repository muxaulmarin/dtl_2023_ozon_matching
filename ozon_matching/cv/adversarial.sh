train_path=data/train_pairs.parquet
test_path=notebooks/adversarial_v4.parquet
train_data_path=data/train_data.parquet
test_data_path=data/test_data.parquet

python ozon_matching/cv/adversarial.py adversarial-check --train-path $train_path --test-path $test_path --train-data-path $train_data_path --test-data-path $test_data_path

train_path=data/train_pairs.parquet
test_path=data/test_pairs_wo_target.parquet
train_data_path=data/train_data.parquet
test_data_path=data/test_data.parquet
python ozon_matching/cv/adversarial.py adversarial-check --train-path $train_path --test-path $test_path --train-data-path $train_data_path --test-data-path $test_data_path
