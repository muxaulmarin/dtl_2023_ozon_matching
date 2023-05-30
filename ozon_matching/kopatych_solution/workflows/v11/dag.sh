experiment_folder=experiments
experiment=v11
commod_folder=common
workflow=ozon_matching/kopatych_solution/workflows/$experiment
raw_data_dir=data/raw

# --- Создаем нужную структура папок ---
mkdir $experiment_folder
mkdir $experiment_folder/$experiment
mkdir $experiment_folder/$experiment/$commod_folder
mkdir $experiment_folder/$experiment/$commod_folder/train
mkdir $experiment_folder/$experiment/$commod_folder/test

# --- Копируем сырые данные в нужную директорию ---
cp $raw_data_dir/train_data.parquet $experiment_folder/$experiment/$commod_folder/train/data.parquet
cp $raw_data_dir/train_pairs.parquet $experiment_folder/$experiment/$commod_folder/train/pairs.parquet
cp $raw_data_dir/test_data.parquet $experiment_folder/$experiment/$commod_folder/test/data.parquet
cp $raw_data_dir/test_pairs_wo_target.parquet $experiment_folder/$experiment/$commod_folder/test/pairs.parquet

python $workflow/data.py prepare-data --data-dir $experiment_folder/$experiment/$commod_folder
python $workflow/title.py fit-titles-model --data-dir $experiment_folder/$experiment/$commod_folder
python $workflow/similarity.py fit-similarity-engine --data-dir $experiment_folder/$experiment/$commod_folder --vector-col main_pic_embeddings_resnet_v1
python $workflow/similarity.py fit-similarity-engine --data-dir $experiment_folder/$experiment/$commod_folder --vector-col name_bert_64
python $workflow/characteristics.py fit-characteristics-model --data-dir $experiment_folder/$experiment/$commod_folder

for fold in train test
do
    python $workflow/title.py create-titles-features --data-dir $experiment_folder/$experiment/$commod_folder --fold $fold
    python $workflow/similarity.py create-similarity-features --data-dir $experiment_folder/$experiment/$commod_folder --fold $fold
    python $workflow/characteristics.py create-characteristics-features --data-dir $experiment_folder/$experiment/$commod_folder --fold $fold
    python $workflow/features.py create-brand-features --data-dir $experiment_folder/$experiment/$commod_folder --fold $fold
    python $workflow/features.py create-compatible-devices-feature --data-dir $experiment_folder/$experiment/$commod_folder --fold $fold
    python $workflow/features.py create-color-feature --data-dir $experiment_folder/$experiment/$commod_folder --fold $fold
    python $workflow/features.py create-watch-feature --data-dir $experiment_folder/$experiment/$commod_folder --fold $fold
    python $workflow/features.py create-size-feature --data-dir $experiment_folder/$experiment/$commod_folder --fold $fold
done


mkdir data/features
for feature_name in brands characteristics-extra colors-extra compatible-devices titles watch size
do
    mkdir data/features/$feature_name
done

for fold in train test
do
    cp $experiment_folder/$experiment/$commod_folder/$fold/brand_features.parquet data/features/brands/$fold.parquet
    cp $experiment_folder/$experiment/$commod_folder/$fold/characteristics_features.parquet data/features/characteristics-extra/$fold.parquet
    cp $experiment_folder/$experiment/$commod_folder/$fold/color_feature.parquet data/features/colors-extra/$fold.parquet
    cp $experiment_folder/$experiment/$commod_folder/$fold/compatible_devices_feature.parquet data/features/compatible-devices/$fold.parquet
    cp $experiment_folder/$experiment/$commod_folder/$fold/titles_features.parquet data/features/titles/$fold.parquet
    cp $experiment_folder/$experiment/$commod_folder/$fold/watch_feature.parquet data/features/watch/$fold.parquet
    cp $experiment_folder/$experiment/$commod_folder/$fold/size_feature.parquet data/features/size/$fold.parquet
done

# rm -rf $experiment_folder
