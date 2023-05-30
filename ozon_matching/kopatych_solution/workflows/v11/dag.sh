experiment_folder=experiments
experiment=v11
n_folds=5
commod_folder=common
workflow=ozon_matching/kopatych_solution/workflows/$experiment

# # --- Создаем нужную структура папок ---
# mkdir $experiment_folder
# mkdir $experiment_folder/$experiment
# mkdir $experiment_folder/$experiment/$commod_folder

# for fold in $(seq 1 $n_folds)
# do
#     mkdir $experiment_folder/$experiment/cv_$fold
#     mkdir $experiment_folder/$experiment/cv_$fold/train
#     mkdir $experiment_folder/$experiment/cv_$fold/test
# done

# mkdir $experiment_folder/$experiment/$commod_folder
# mkdir $experiment_folder/$experiment/$commod_folder/train
# mkdir $experiment_folder/$experiment/$commod_folder/test

# # --- Копируем сырые данные в нужную директорию ---
# cp data/train_data.parquet $experiment_folder/$experiment/$commod_folder/train/data.parquet
# cp data/train_pairs.parquet $experiment_folder/$experiment/$commod_folder/train/pairs.parquet
# cp data/test_data.parquet $experiment_folder/$experiment/$commod_folder/test/data.parquet
# cp data/test_pairs_wo_target.parquet $experiment_folder/$experiment/$commod_folder/test/pairs.parquet

# python $workflow/data.py prepare-data --data-dir $experiment_folder/$experiment/$commod_folder
# python $workflow/title.py fit-titles-model --data-dir $experiment_folder/$experiment/$commod_folder
# python $workflow/similarity.py fit-similarity-engine --data-dir $experiment_folder/$experiment/$commod_folder --vector-col main_pic_embeddings_resnet_v1
# python $workflow/similarity.py fit-similarity-engine --data-dir $experiment_folder/$experiment/$commod_folder --vector-col name_bert_64
# python $workflow/characteristics.py fit-characteristics-model --data-dir $experiment_folder/$experiment/$commod_folder

# for fold in train test
# do
    # python $workflow/title.py create-titles-features --data-dir $experiment_folder/$experiment/$commod_folder --fold $fold
    # python $workflow/similarity.py create-similarity-features --data-dir $experiment_folder/$experiment/$commod_folder --fold $fold
    # python $workflow/characteristics.py create-characteristics-features --data-dir $experiment_folder/$experiment/$commod_folder --fold $fold
# done

#python $workflow/cv.py split-data-for-cv --data-dir $experiment_folder/$experiment --n-folds $n_folds

# for fold in $(seq 1 $n_folds)
# do
#     python $workflow/classifier.py fit-model --data-dir $experiment_folder/$experiment/cv_$fold
#     for fold_type in train test
#     do
#         python $workflow/classifier.py predict --data-dir $experiment_folder/$experiment/cv_$fold/$fold_type
#         python $workflow/classifier.py evaluate --data-dir $experiment_folder/$experiment/cv_$fold --fold-type $fold_type
#     done
# done

# python $workflow/classifier.py cv-scores --data-dir $experiment_folder/$experiment --n-folds $n_folds

python $workflow/data.py prepare-data-submit --data-dir $experiment_folder/$experiment/$commod_folder/test

python $workflow/classifier.py oof-predict --data-dir $experiment_folder/$experiment/$commod_folder --n-folds $n_folds
python $workflow/data.py oof-predict --data-dir $experiment_folder/$experiment --n-folds $n_folds