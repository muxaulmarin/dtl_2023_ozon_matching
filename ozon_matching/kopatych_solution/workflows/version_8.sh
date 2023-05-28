experiment_folder=experiments
experiment=v9
n_folds=5
oof=$(($n_folds + 1))
workflow=ozon_matching/kopatych_solution
adversarial_data_path=notebooks/adversarial_v4.parquet
sf_oof_path=data/oof.parquet
sf_submit_path=data/sf_submit.parquet

# --- Создаем нужную структура папок ---
mkdir $experiment_folder
mkdir $experiment_folder/$experiment

for fold in $(seq 1 $oof)
do
    mkdir $experiment_folder/$experiment/cv_$fold
    mkdir $experiment_folder/$experiment/cv_$fold/train
    mkdir $experiment_folder/$experiment/cv_$fold/test
    mkdir $experiment_folder/$experiment/cv_$fold/adversarial
    cp $adversarial_data_path $experiment_folder/$experiment/cv_$fold/adversarial/pairs.parquet
done

# --- Копируем сырые данные в нужную директорию ---
cp data/train_data.parquet $experiment_folder/$experiment/cv_$oof/train/data.parquet
cp data/train_pairs.parquet $experiment_folder/$experiment/cv_$oof/train/pairs.parquet
cp data/test_data.parquet $experiment_folder/$experiment/cv_$oof/test/data.parquet
cp data/test_pairs_wo_target.parquet $experiment_folder/$experiment/cv_$oof/test/pairs.parquet

python $workflow/cv.py split-data-for-cv-adversarial-holdout --data-dir $experiment_folder/$experiment --n-folds $n_folds --adversarial-data-path $adversarial_data_path
python $workflow/data.py prepare-data --data-dir $experiment_folder/$experiment --n-folds $n_folds
python $workflow/characteristics.py fit-characteristics-model --data-dir $experiment_folder/$experiment/cv_$oof
python $workflow/similarity.py fit-similarity-engine --data-dir $experiment_folder/$experiment/cv_$oof --vector-col main_pic_embeddings_resnet_v1
python $workflow/similarity.py fit-similarity-engine --data-dir $experiment_folder/$experiment/cv_$oof --vector-col name_bert_64

for fold in $(seq 1 $n_folds)
do

    for fold_type in train test
    do
        python $workflow/similarity.py create-similarity-features --data-dir $experiment_folder/$experiment/cv_$fold --fold-type $fold_type --n-folds $n_folds
        python $workflow/characteristics.py create-characteristics-features --data-dir $experiment_folder/$experiment/cv_$fold --fold-type $fold_type --n-folds $n_folds
        python $workflow/classifier.py create-dataset --data-dir $experiment_folder/$experiment/cv_$fold --fold-type $fold_type --sf-oof-path $sf_oof_path --n-folds $n_folds
    done

    python $workflow/classifier.py fit-model --data-dir $experiment_folder/$experiment/cv_$fold --params-version $experiment

    for fold_type in train test
    do
        python $workflow/classifier.py predict --data-dir $experiment_folder/$experiment/cv_$fold --fold-type $fold_type --params-version $experiment
        python $workflow/classifier.py evaluate --data-dir $experiment_folder/$experiment/cv_$fold --fold-type $fold_type --n-folds $n_folds
    done
done
python $workflow/classifier.py cv-scores --data-dir $experiment_folder/$experiment --n-folds $n_folds

python $workflow/similarity.py create-similarity-features --data-dir $experiment_folder/$experiment/cv_$oof --fold-type test --n-folds $n_folds
python $workflow/characteristics.py create-characteristics-features --data-dir $experiment_folder/$experiment/cv_$oof --fold-type test --n-folds $n_folds
python $workflow/classifier.py create-dataset --data-dir $experiment_folder/$experiment/cv_$oof --fold-type test --sf-oof-path $sf_submit_path --n-folds $n_folds

python $workflow/classifier.py oof-predict --data-dir $experiment_folder/$experiment --n-folds $n_folds --experiment-version $experiment
