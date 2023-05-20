n_folds=2
workflow="ozon_matching/kopatych_solution/workflow.py"

# python $workflow split-data-for-cv --data-dir data --n-folds $n_folds

for fold in $(seq 1 $n_folds)
do
    # python $workflow fit-similarity-engine --data-dir data/cv_$fold --vector-col main_pic_embeddings_resnet_v1
    # python $workflow fit-similarity-engine --data-dir data/cv_$fold --vector-col name_bert_64
    # for fold_type in train test
    # do
    #     python $workflow create-similarity-features --data-dir data/cv_$fold --fold-type $fold_type
    #     python $workflow create-characteristics-features --data-dir data/cv_$fold --fold-type $fold_type
    #     python $workflow create-dataset --data-dir data/cv_$fold --fold-type $fold_type
    # done
    # python $workflow fit-model --data-dir data/cv_$fold --params-version v1
    # python $workflow predict --data-dir data/cv_$fold --params-version v1
    for fold_type in train test
    do
        python $workflow evaluate --data-dir data/cv_$fold --fold-type $fold_type
    done
done
