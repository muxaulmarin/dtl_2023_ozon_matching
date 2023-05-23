import os

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from ozon_matching.kopatych_solution.characteristics import CharacteristicsModel
from ozon_matching.kopatych_solution.config.model_cfg import lgbm_params
from ozon_matching.kopatych_solution.cv import stratified_k_fold
from ozon_matching.kopatych_solution.features import (
    _create_characteristics_dict,
    _create_characteristics_dict_v5,
    _create_characteristics_keys,
    create_characteristic_features_v5,
)
from ozon_matching.kopatych_solution.similarity import SimilarityEngine
from ozon_matching.kopatych_solution.utils import (
    extract_category_levels,
    get_and_create_dir,
    log_cli,
    read_json,
    read_model,
    read_parquet,
    write_json,
    write_model,
    write_parquet,
)
from sklearn.metrics import roc_auc_score
from typer import Option, Typer

cli = Typer()


@cli.command()
@log_cli
def prepare_data(data_dir: str = Option(...)):
    categories = pl.concat(
        [
            read_parquet(
                os.path.join(data_dir, "train", "data.parquet"),
                columns=["variantid", "categories"],
            ),
            read_parquet(
                os.path.join(data_dir, "test", "data.parquet"),
                columns=["variantid", "categories"],
            ),
        ]
    )
    categories = categories.unique(subset=["variantid"])
    categories = extract_category_levels(categories, [3, 4], "categories")

    categories = categories.join(
        categories.select(pl.col("category_level_3"))
        .unique()
        .with_row_count(name="category_level_3_id"),
        on=["category_level_3"],
    )
    categories = categories.join(
        categories.select(pl.col("category_level_4"))
        .unique()
        .with_row_count(name="category_level_4_id"),
        on=["category_level_4"],
    )
    categories = categories.join(
        (
            categories["category_level_3_id"]
            .value_counts()
            .filter(pl.col("counts") > 1000)
            .select(pl.col("category_level_3_id"))
            .with_row_count(name="category_level_3_id_grouped", offset=1)
        ),
        on=["category_level_3_id"],
        how="left",
    )
    categories = categories.with_columns(
        [pl.col("category_level_3_id_grouped").fill_null(0)]
    )
    categories = categories.drop(["categories"])

    write_parquet(
        read_parquet(os.path.join(data_dir, "train", "data.parquet"))
        .join(categories, on=["variantid"])
        .drop("categories"),
        os.path.join(data_dir, "train", "data.parquet"),
    )

    write_parquet(
        read_parquet(os.path.join(data_dir, "test", "data.parquet"))
        .join(categories, on=["variantid"])
        .drop("categories"),
        os.path.join(data_dir, "test", "data.parquet"),
    )

    write_parquet(
        read_parquet(os.path.join(data_dir, "train", "pairs.parquet")).select(
            pl.col(["variantid1", "variantid2", "target"])
        ),
        os.path.join(data_dir, "train", "pairs.parquet"),
    )

    write_parquet(
        read_parquet(os.path.join(data_dir, "test", "pairs.parquet")).select(
            pl.col(["variantid1", "variantid2"])
        ),
        os.path.join(data_dir, "test", "pairs.parquet"),
    )


@cli.command()
@log_cli
def split_data_for_cv(
    data_dir: str = Option(...),
    n_folds: int = Option(...),
):

    pairs = read_parquet(
        os.path.join(data_dir, f"cv_{n_folds + 1}", "train", "pairs.parquet")
    )
    pairs = pairs.with_columns([pl.col("target").cast(pl.Int8)])
    pairs = pairs.with_row_count(name="index")

    data = read_parquet(
        os.path.join(data_dir, f"cv_{n_folds + 1}", "train", "data.parquet")
    )

    pairs = pairs.join(
        data.select(
            [
                pl.col("variantid").alias("variantid1"),
                pl.col("category_level_3_id_grouped"),
            ]
        ),
        on=["variantid1"],
    )
    cv_target = (
        pairs.select(pl.col(["target", "category_level_3_id_grouped"]))
        .unique()
        .sort(["category_level_3_id_grouped", "target"])
        .with_row_count(name="cv_target")
    )
    pairs = pairs.join(cv_target, on=["target", "category_level_3_id_grouped"]).drop(
        ["category_level_3_id_grouped"]
    )
    for n, train_pairs, test_pairs in stratified_k_fold(
        data=pairs, stratify_col="cv_target", k=int(n_folds)
    ):

        write_parquet(
            train_pairs.select(pl.col(["variantid1", "variantid2", "target"])),
            os.path.join(data_dir, f"cv_{n}", "train", "pairs.parquet"),
        )
        write_parquet(
            test_pairs.select(pl.col(["variantid1", "variantid2"])),
            os.path.join(data_dir, f"cv_{n}", "test", "pairs.parquet"),
        )

        train_data = data.filter(
            pl.col("variantid").is_in(train_pairs["variantid1"])
            | pl.col("variantid").is_in(train_pairs["variantid2"])
        )
        write_parquet(
            train_data, os.path.join(data_dir, f"cv_{n}", "train", "data.parquet")
        )

        test_data = data.filter(
            pl.col("variantid").is_in(test_pairs["variantid1"])
            | pl.col("variantid").is_in(test_pairs["variantid2"])
        )
        write_parquet(
            test_data, os.path.join(data_dir, f"cv_{n}", "test", "data.parquet")
        )


@cli.command()
@log_cli
def split_data_for_cv_v6(
    data_dir: str = Option(...),
    n_folds: int = Option(...),
):

    pairs = read_parquet(
        os.path.join(data_dir, f"cv_{n_folds + 1}", "train", "pairs.parquet")
    )
    pairs = pairs.with_columns([pl.col("target").cast(pl.Int8)])
    pairs = pairs.with_row_count(name="index")

    data = read_parquet(
        os.path.join(data_dir, f"cv_{n_folds + 1}", "train", "data.parquet")
    )

    pairs = pairs.join(
        data.select(
            [
                pl.col("variantid").alias("variantid1"),
                pl.col("category_level_3_id_grouped"),
            ]
        ),
        on=["variantid1"],
    )
    cv_target = (
        pairs.select(pl.col(["target", "category_level_3_id_grouped"]))
        .unique()
        .sort(["category_level_3_id_grouped", "target"])
        .with_row_count(name="cv_target")
    )
    pairs = pairs.join(cv_target, on=["target", "category_level_3_id_grouped"]).drop(
        ["category_level_3_id_grouped"]
    )
    for n, train_pairs, test_pairs in stratified_k_fold(
        data=pairs, stratify_col="target", k=int(n_folds)
    ):

        write_parquet(
            train_pairs.select(pl.col(["variantid1", "variantid2", "target"])),
            os.path.join(data_dir, f"cv_{n}", "train", "pairs.parquet"),
        )
        write_parquet(
            test_pairs.select(pl.col(["variantid1", "variantid2"])),
            os.path.join(data_dir, f"cv_{n}", "test", "pairs.parquet"),
        )

        train_data = data.filter(
            pl.col("variantid").is_in(train_pairs["variantid1"])
            | pl.col("variantid").is_in(train_pairs["variantid2"])
        )
        write_parquet(
            train_data, os.path.join(data_dir, f"cv_{n}", "train", "data.parquet")
        )

        test_data = data.filter(
            pl.col("variantid").is_in(test_pairs["variantid1"])
            | pl.col("variantid").is_in(test_pairs["variantid2"])
        )
        write_parquet(
            test_data, os.path.join(data_dir, f"cv_{n}", "test", "data.parquet")
        )


# @cli.command()
# @log_cli
# def create_characteristics_dict(
#     data_dir: str = Option(...), experiment: str = Option(...)
# ):
#     train_data = read_parquet(
#         os.path.join(data_dir, "data", "train", "data.parquet"),
#         columns=["variantid", "characteristic_attributes_mapping", "categories"],
#     )
#     test_data = read_parquet(
#         os.path.join(data_dir, "data", "test", "data.parquet"),
#         columns=["variantid", "characteristic_attributes_mapping", "categories"],
#     )
#     data = pl.concat([train_data, test_data]).unique(subset=["variantid"])
#     characteristics_dict = _create_characteristics_dict(data)
#     write_json(
#         characteristics_dict,
#         os.path.join(data_dir, experiment, "characteristics_dict.json"),
#     )

#     train_pairs = read_parquet(
#         os.path.join(data_dir, "data", "train", "pairs.parquet"),
#         columns=["variantid1", "variantid2"],
#     )
#     test_pairs = read_parquet(
#         os.path.join(data_dir, "data", "test", "pairs.parquet"),
#         columns=["variantid1", "variantid2"],
#     )
#     pairs = pl.concat([train_pairs, test_pairs]).unique()
#     characteristics_index = {
#         characteristic: i
#         for i, characteristic in enumerate(
#             _create_characteristics_keys(pairs, characteristics_dict)
#         )
#     }
#     write_json(
#         characteristics_index,
#         os.path.join(data_dir, experiment, "characteristics_index.json"),
#     )

#     data = extract_category_levels(data, [3], "categories")
#     categories = (
#         data.select(pl.col("category_level_3"))
#         .unique()
#         .with_row_count(name="category_level_3_id")
#     )
#     write_parquet(categories, os.path.join(data_dir, experiment, "categories.parquet"))


# @cli.command()
# @log_cli
# def create_characteristics_dict_v5(data_dir: str = Option(...)):
#     train_data = read_parquet(
#         os.path.join(data_dir, "train", "data.parquet"),
#         columns=["variantid", "characteristic_attributes_mapping", "categories"],
#     )
#     test_data = read_parquet(
#         os.path.join(data_dir, "test", "data.parquet"),
#         columns=["variantid", "characteristic_attributes_mapping", "categories"],
#     )
#     data = pl.concat([train_data, test_data]).unique(subset=["variantid"])

#     characteristics_dict = _create_characteristics_dict_v5(data)
#     write_json(
#         characteristics_dict,
#         os.path.join(data_dir, "characteristics_dict.json"),
#     )

#     train_pairs = read_parquet(
#         os.path.join(data_dir, "train", "pairs.parquet"),
#         columns=["variantid1", "variantid2"],
#     )
#     test_pairs = read_parquet(
#         os.path.join(data_dir, "test", "pairs.parquet"),
#         columns=["variantid1", "variantid2"],
#     )
#     pairs = pl.concat([train_pairs, test_pairs]).unique()
#     characteristics_index = {
#         characteristic: i
#         for i, characteristic in enumerate(
#             _create_characteristics_keys(pairs, characteristics_dict)
#         )
#     }
#     write_json(
#         characteristics_index,
#         os.path.join(data_dir, "characteristics_index.json"),
#     )

#     data = extract_category_levels(data, [3, 4], "categories")
#     categories_level_3 = (
#         data.select(pl.col(["category_level_3"]))
#         .unique()
#         .with_row_count(name="category_level_3_id")
#     )
#     categories_level_4 = (
#         data.select(pl.col(["category_level_4"]))
#         .unique()
#         .with_row_count(name="category_level_4_id")
#     )
#     write_parquet(
#         categories_level_3, os.path.join(data_dir, "categories_level_3.parquet")
#     )
#     write_parquet(
#         categories_level_4, os.path.join(data_dir, "categories_level_4.parquet")
#     )


@cli.command()
@log_cli
def fit_characteristics_model(data_dir: str = Option(...)):
    data = pl.concat(
        [
            read_parquet(
                os.path.join(data_dir, "train", "data.parquet"),
                columns=["variantid", "characteristic_attributes_mapping"],
            ),
            read_parquet(
                os.path.join(data_dir, "test", "data.parquet"),
                columns=["variantid", "characteristic_attributes_mapping"],
            ),
        ]
    )
    pairs = pl.concat(
        [
            read_parquet(
                os.path.join(data_dir, "train", "pairs.parquet"),
                columns=["variantid1", "variantid2"],
            ),
            read_parquet(
                os.path.join(data_dir, "test", "pairs.parquet"),
                columns=["variantid1", "variantid2"],
            ),
        ]
    )

    model = CharacteristicsModel()
    model.fit(data, pairs)

    write_model(
        os.path.join(data_dir, "characteristics_model.jbl"),
        model,
    )


@cli.command()
@log_cli
def fit_similarity_engine(data_dir: str = Option(...), vector_col: str = Option(...)):

    train_data = read_parquet(
        os.path.join(data_dir, "train", "data.parquet"),
        columns=["variantid", vector_col],
    )
    test_data = read_parquet(
        os.path.join(data_dir, "test", "data.parquet"),
        columns=["variantid", vector_col],
    )

    data = pl.concat([train_data, test_data])

    similarity_engine = SimilarityEngine(index_col="variantid", vector_col=vector_col)
    similarity_engine.fit(data)
    write_model(
        os.path.join(data_dir, f"similarity_engine_{vector_col}.jbl"),
        similarity_engine,
    )


@cli.command()
@log_cli
def create_similarity_features(
    data_dir: str = Option(...),
    fold_type: str = Option(...),
    n_folds: int = Option(...),
):
    pairs = read_parquet(os.path.join(data_dir, fold_type, "pairs.parquet"))

    pic_similarity_engine: SimilarityEngine = read_model(
        os.path.join(
            os.path.dirname(data_dir),
            f"cv_{n_folds + 1}",
            "similarity_engine_main_pic_embeddings_resnet_v1.jbl",
        )
    )
    name_similarity_engine: SimilarityEngine = read_model(
        os.path.join(
            os.path.dirname(data_dir),
            f"cv_{n_folds + 1}",
            "similarity_engine_name_bert_64.jbl",
        )
    )
    for engine in [pic_similarity_engine, name_similarity_engine]:
        feature = engine.predict(pairs)
        write_parquet(
            feature,
            os.path.join(
                data_dir, fold_type, f"similarity_features_{engine.vector_col}.parquet"
            ),
        )


@cli.command()
@log_cli
def create_characteristics_features(
    data_dir: str = Option(...),
    fold_type: str = Option(...),
    n_folds: int = Option(...),
):
    pairs = read_parquet(os.path.join(data_dir, fold_type, "pairs.parquet"))

    characteristics_model: CharacteristicsModel = read_model(
        os.path.join(
            os.path.dirname(data_dir), f"cv_{n_folds + 1}", "characteristics_model.jbl"
        )
    )
    feature = characteristics_model.predict(pairs)
    write_parquet(
        feature, os.path.join(data_dir, fold_type, "characteristics_features.parquet")
    )


# @cli.command()
# @log_cli
# def create_characteristics_features_v5(
#     data_dir: str = Option(...), fold_type: str = Option(...)
# ):
#     pairs = read_parquet(os.path.join(data_dir, fold_type, "pairs.parquet"))
#     characteristics_dict = read_json(
#         os.path.join(os.path.dirname(data_dir), "characteristics_dict.json")
#     )
#     feature = create_characteristic_features_v5(pairs, characteristics_dict)

#     fodler = get_and_create_dir(os.path.join(data_dir, fold_type, "features"))
#     write_parquet(feature, os.path.join(fodler, "characteristics_features_v5.parquet"))


@cli.command()
@log_cli
def create_dataset(data_dir: str = Option(...), fold_type: str = Option(...)):

    pairs = read_parquet(os.path.join(data_dir, fold_type, "pairs.parquet"))

    pic_sim_features = read_parquet(
        os.path.join(
            data_dir,
            fold_type,
            "similarity_features_main_pic_embeddings_resnet_v1.parquet",
        )
    )
    name_sim_features = read_parquet(
        os.path.join(data_dir, fold_type, "similarity_features_name_bert_64.parquet")
    )
    characteristics_features = read_parquet(
        os.path.join(data_dir, fold_type, "characteristics_features.parquet")
    )

    categories = read_parquet(
        os.path.join(data_dir, fold_type, "data.parquet"),
        columns=[
            "category_level_3_id",
            "category_level_4_id",
            "category_level_3_id_grouped",
            "variantid",
        ],
    )
    categories = categories.rename({"variantid": "variantid1"})

    dataset = (
        pairs.join(pic_sim_features, on=["variantid1", "variantid2"])
        .join(name_sim_features, on=["variantid1", "variantid2"])
        .join(characteristics_features, on=["variantid1", "variantid2"])
        .join(categories, on=["variantid1"])
    )
    write_parquet(dataset, os.path.join(data_dir, fold_type, "dataset.parquet"))


# @cli.command()
# @log_cli
# def create_dataset(data_dir: str = Option(...), fold_type: str = Option(...)):

#     pairs = read_parquet(os.path.join(data_dir, fold_type, "pairs.parquet"))
#     pairs = pairs.drop(["index", "cv_target"])
#     categories = read_parquet(
#         os.path.join(os.path.dirname(data_dir), "categories.parquet")
#     )
#     pairs = pairs.join(categories, on=["category_level_3"]).drop("category_level_3")

#     pic_sim_features = read_parquet(
#         os.path.join(
#             data_dir,
#             fold_type,
#             "features",
#             "similarity_features_main_pic_embeddings_resnet_v1.parquet",
#         )
#     )
#     name_sim_features = read_parquet(
#         os.path.join(
#             data_dir, fold_type, "features", "similarity_features_name_bert_64.parquet"
#         )
#     )
#     characteristics_features = read_parquet(
#         os.path.join(
#             data_dir, fold_type, "features", "characteristics_features.parquet"
#         )
#     )

#     dataset = (
#         pairs.join(pic_sim_features, on=["variantid1", "variantid2"])
#         .join(name_sim_features, on=["variantid1", "variantid2"])
#         .join(characteristics_features, on=["variantid1", "variantid2"])
#     )
#     write_parquet(
#         dataset, os.path.join(data_dir, fold_type, "features", "dataset.parquet")
#     )

#     dataset_pandas = dataset.to_pandas()

#     lgbm_dataset = lightgbm.Dataset(
#         data=dataset_pandas.drop(columns=["target", "variantid1", "variantid2"]),
#         label=dataset_pandas["target"].values,
#         reference=None
#         if fold_type == "train"
#         else read_model(os.path.join(data_dir, "models", "train_lgbm_dataset.jbl")),
#         categorical_feature=["category_level_3_id"],
#     )
#     write_model(
#         os.path.join(data_dir, "models", f"{fold_type}_lgbm_dataset.jbl"), lgbm_dataset
#     )


@cli.command()
@log_cli
def fit_model(data_dir: str = Option(...), params_version: str = Option(...)):
    dataset = read_parquet(os.path.join(data_dir, "train", "dataset.parquet"))
    model = LGBMClassifier(**lgbm_params[params_version])
    model.fit(
        X=dataset.drop(["variantid1", "variantid2", "target"]).to_pandas(),
        y=dataset["target"].to_numpy(),
        categorical_feature=[
            "category_level_3_id",
            "category_level_4_id",
            "category_level_3_id_grouped",
        ],
    )
    write_model(os.path.join(data_dir, f"lgbm_{params_version}.jbl"), model)


@cli.command()
@log_cli
def predict(data_dir: str = Option(...), params_version: str = Option(...)):
    model = read_model(os.path.join(data_dir, f"lgbm_{params_version}.jbl"))
    for fold_type in ["train", "test"]:
        data = read_parquet(os.path.join(data_dir, fold_type, "dataset.parquet"))
        data = data.to_pandas()
        data["scores"] = model.predict_proba(data[model.feature_name_])[:, 1]
        data[["variantid1", "variantid2", "scores"]].to_csv(
            os.path.join(data_dir, fold_type, "prediction.csv"), index=False
        )


@cli.command()
@log_cli
def evaluate(
    data_dir: str = Option(...),
    fold_type: str = Option(...),
    n_folds: int = Option(...),
):
    target = read_parquet(
        os.path.join(
            os.path.dirname(data_dir), f"cv_{n_folds + 1}", "train", "pairs.parquet"
        ),
        columns=["variantid1", "variantid2", "target"],
    )
    predictions = pl.read_csv(os.path.join(data_dir, fold_type, "prediction.csv"))
    df = (
        predictions.select(pl.col(["variantid1", "variantid2", "scores"]))
        .join(target, on=["variantid1", "variantid2"])
        .to_pandas()
    )

    # categories = pl.read_parquet(os.path.join(os.path.dirname(data_dir), f'cv_{n_folds + 1}', 'train', "data.parquet"), columns=['variantid', 'category_level_3_id_grouped'])
    # categories = categories.rename({'variantid': 'variantid1'})

    # target = target.join(predictions.drop('scores'), on=['variantid1', 'variantid2'])

    # predictions = predictions.join(
    #     target, on=['variantid1', 'variantid2']
    # ).join(
    #     categories, on=['variantid1']
    # ).drop('target')

    # predictions = predictions.to_pandas()
    # target = target.to_pandas()

    # score = pr_auc_macro(
    #     target, predictions, prec_level=0.75, cat_column="category_level_3_id_grouped"
    # )
    roc_auc = roc_auc_score(df["target"].values, df["scores"].values)
    write_json(
        {"competition_metric": 0.5, "rocauc": roc_auc},
        os.path.join(data_dir, fold_type, "score.json"),
    )


@cli.command()
@log_cli
def cv_scores(
    data_dir: str = Option(...),
    n_folds: int = Option(...),
):
    metrics = {}
    for fold_type in ["train", "test"]:
        scores = {"competition_metric": [], "rocauc": []}
        for n in range(1, n_folds + 1):
            path = os.path.join(data_dir, f"cv_{n}", fold_type, "score.json")
            data = read_json(path)
            scores["competition_metric"].append(data["competition_metric"])
            scores["rocauc"].append(data["rocauc"])
        metrics[fold_type] = {
            "competition_metric": scores["competition_metric"],
            "avg_competition_metric": np.mean(scores["competition_metric"]),
            "std_competition_metric": np.std(scores["competition_metric"]),
            "rocauc": scores["rocauc"],
            "avg_rocauc": np.mean(scores["rocauc"]),
            "std_rocauc": np.std(scores["rocauc"]),
        }
    write_json(metrics, os.path.join(data_dir, "cv_result.json"))


@cli.command()
@log_cli
def oof_predict(
    data_dir: str = Option(...),
    n_folds: int = Option(...),
    experiment_version: str = Option(...),
):
    dataset = read_parquet(
        os.path.join(data_dir, f"cv_{n_folds + 1}", "test", "dataset.parquet")
    ).to_pandas()
    predict_cols = []
    for fold in range(1, n_folds + 1):
        model: LGBMClassifier = read_model(
            os.path.join(data_dir, f"cv_{fold}", f"lgbm_{experiment_version}.jbl")
        )
        predict_col = f"predict_fold_{fold}"
        dataset[predict_col] = model.predict_proba(dataset[model.feature_name_])[:, 1]
        predict_cols.append(predict_col)
    dataset[["variantid1", "variantid2"] + predict_cols].to_csv(
        os.path.join(data_dir, "submit_folds.csv"), index=False
    )
    dataset["scores"] = dataset[predict_cols].mean(axis=1)
    dataset[["variantid1", "variantid2", "scores"]].to_csv(
        os.path.join(data_dir, f"submit_{experiment_version}.csv"), index=False
    )


if __name__ == "__main__":
    cli()
