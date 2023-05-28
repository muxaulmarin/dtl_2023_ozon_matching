catboost_params = {
    "model_params": {
        "iterations": 1_000,
        "early_stopping_rounds": 10,
        "random_seed": 777,
    },
    "pool_params": {
        "cat_features": [
            "category_level_2",
            "category_level_3",
            "category_level_4",
            "pic_embeddings_resnet_v1_fillness",
            "color_parsed_fillness",
        ]
    },
}
