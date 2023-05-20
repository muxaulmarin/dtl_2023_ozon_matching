lgbm_params = {
    "v1": {
        "boosting_type": "gbdt",
        "objective": "binary",
        "n_estimators": 300,
        "max_depth": -1,
        "num_leaves": 37,
        "min_data_in_leaf": 25,
        "learning_rate": 0.05,
        "lambda_l1": 1e-3,
        "lambda_l2": 1e-3,
        "max_bin": 255,
        "use_missing": True,
        "zero_as_missing": False,
        "random_state": 13,
    }
}
