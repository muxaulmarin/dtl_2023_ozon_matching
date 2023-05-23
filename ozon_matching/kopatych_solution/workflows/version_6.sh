experiment_folder=experiments
experiment=v6
n_folds=5
oof=$(($n_folds + 1))
workflow=ozon_matching/kopatych_solution/workflow.py

#--- Создаем нужную структура папок ---
mkdir $experiment_folder
mkdir $experiment_folder/$experiment

for fold in $(seq 1 $oof)
do
    mkdir $experiment_folder/$experiment/cv_$fold
    mkdir $experiment_folder/$experiment/cv_$fold/train
    mkdir $experiment_folder/$experiment/cv_$fold/test
done

#--- Копируем сырые данные в нужную директорию ---
cp data/train_data.parquet $experiment_folder/$experiment/cv_$oof/train/data.parquet
cp data/train_pairs.parquet $experiment_folder/$experiment/cv_$oof/train/pairs.parquet
cp data/test_data.parquet $experiment_folder/$experiment/cv_$oof/test/data.parquet
cp data/test_pairs_wo_target.parquet $experiment_folder/$experiment/cv_$oof/test/pairs.parquet

# --- Добавляем категории в исходные файлы
python $workflow prepare-data --data-dir $experiment_folder/$experiment/cv_$oof

# --- Обучаем общие для пайплайна модели, которые никак не привязаны к фолду ---

python $workflow fit-characteristics-model --data-dir $experiment_folder/$experiment/cv_$oof
python $workflow fit-similarity-engine --data-dir $experiment_folder/$experiment/cv_$oof --vector-col main_pic_embeddings_resnet_v1
python $workflow fit-similarity-engine --data-dir $experiment_folder/$experiment/cv_$oof --vector-col name_bert_64

# --- Разбираем обучающую выборку на N фолдов
python $workflow split-data-for-cv-$experiment --data-dir $experiment_folder/$experiment --n-folds $n_folds

for fold in $(seq 1 $n_folds)
do
    for fold_type in train test
    do
        python $workflow create-similarity-features --data-dir $experiment_folder/$experiment/cv_$fold --fold-type $fold_type --n-folds $n_folds
        python $workflow create-characteristics-features --data-dir $experiment_folder/$experiment/cv_$fold --fold-type $fold_type --n-folds $n_folds
        python $workflow create-dataset --data-dir $experiment_folder/$experiment/cv_$fold --fold-type $fold_type
    done

    python $workflow fit-model --data-dir $experiment_folder/$experiment/cv_$fold --params-version $experiment
    python $workflow predict --data-dir $experiment_folder/$experiment/cv_$fold --params-version $experiment
    for fold_type in train test
    do
        python $workflow evaluate --data-dir $experiment_folder/$experiment/cv_$fold --fold-type $fold_type --n-folds $n_folds
    done
done

python $workflow cv-scores --data-dir $experiment_folder/$experiment --n-folds $n_folds

python $workflow create-similarity-features --data-dir $experiment_folder/$experiment/cv_$oof --fold-type test --n-folds $n_folds
python $workflow create-characteristics-features --data-dir $experiment_folder/$experiment/cv_$oof --fold-type test --n-folds $n_folds
python $workflow create-dataset --data-dir $experiment_folder/$experiment/cv_$oof --fold-type test

python $workflow oof-predict --data-dir $experiment_folder/$experiment --n-folds $n_folds --experiment-version $experiment
