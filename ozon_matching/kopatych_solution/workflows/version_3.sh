# mkdir data/data/train
# mkdir data/data/test

# cp data/raw_data/hackathon_files_for_participants_ozon/train_data.parquet data/data/train/data.parquet
# cp data/raw_data/hackathon_files_for_participants_ozon/train_pairs.parquet data/data/train/pairs.parquet
# cp data/raw_data/hackathon_files_for_participants_ozon/test_data.parquet data/data/test/data.parquet
# cp data/raw_data/hackathon_files_for_participants_ozon/test_pairs_wo_target.parquet data/data/test/pairs.parquet

n_folds=5
workflow=ozon_matching/kopatych_solution/workflow.py
experiment=v3

python $workflow create-characteristics-dict --data-dir data --experiment $experiment
python $workflow split-data-for-cv --data-dir data --n-folds $n_folds --experiment $experiment

for fold in $(seq 1 $n_folds)
do
    python $workflow fit-similarity-engine --data-dir data/$experiment/cv_$fold --vector-col main_pic_embeddings_resnet_v1
    python $workflow fit-similarity-engine --data-dir data/$experiment/cv_$fold --vector-col name_bert_64
    for fold_type in train test
    do
        python $workflow create-similarity-features --data-dir data/$experiment/cv_$fold --fold-type $fold_type
        python $workflow create-characteristics-features --data-dir data/$experiment/cv_$fold --fold-type $fold_type
        python $workflow create-dataset --data-dir data/$experiment/cv_$fold --fold-type $fold_type
    done
    python $workflow fit-model --data-dir data/$experiment/cv_$fold --params-version $experiment
    python $workflow predict --data-dir data/$experiment/cv_$fold --params-version $experiment
    for fold_type in train test
    do
        python $workflow evaluate --data-dir data/$experiment/cv_$fold --fold-type $fold_type
    done
done

python $workflow cv-scores --data-dir data --n-folds $n_folds --experiment $experiment
