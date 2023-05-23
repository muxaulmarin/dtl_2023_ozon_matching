from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

import catboost as cb
import numpy as np
import pandas as pd
import polars as pl
from catboost.utils import get_gpu_device_count
from ozon_matching.andrey_solution.modeling.validation import calc_metrics, kfold_split


class CatBoostCV:
    def __init__(
        self,
        model_params: dict[str, Any],
        pool_params: dict[str, Any],
        splitter: Callable[[pl.DataFrame], tuple[np.ndarray, np.ndarray]] = kfold_split,
    ) -> None:
        model_params.setdefault(
            "task_type", "GPU" if get_gpu_device_count() > 0 else "CPU"
        )
        model_params["allow_writing_files"] = False

        self.model_params = model_params
        self.pool_params = pool_params
        self.splitter = splitter

    def fit(
        self, train: pl.DataFrame, verbose: Optional[int] = None
    ) -> list[dict[str, float]]:
        self.models_ = []
        self.metrics_ = []

        for fold, (train_idx, val_idx) in enumerate(self.splitter(train)):
            train_fold, val_fold = train[train_idx], train[val_idx]

            train_fold_pool, val_fold_pool = self.to_pool(train_fold), self.to_pool(
                val_fold
            )
            fold_model = cb.CatBoostClassifier(**self.model_params)
            fold_model.fit(
                train_fold_pool,
                eval_set=val_fold_pool,
                verbose=verbose or self.model_params.get("early_stopping_rounds"),
            )
            self.models_.append(fold_model)

            fold_metrics = calc_metrics(
                y_true=val_fold["target"].to_pandas(),
                y_pred=fold_model.predict_proba(val_fold_pool)[:, 1],
                categories=val_fold["cat3_grouped"],
            )
            print()
            print(f"{'-' * 20} {fold = } {'-' * 20}")
            for name, value in fold_metrics.items():
                print(f"{name} = {value:.4f}")
            print(f"{'-' * 20} {fold = } {'-' * 20}")
            print()
            self.metrics_.append(fold_metrics)

        return self.metrics_

    def predict(self, test: pl.DataFrame, fold: Optional[int] = None) -> pd.DataFrame:
        if fold is not None:
            models = [self.models_[fold]]
        else:
            models = self.models_

        test_pool = self.to_pool(test)
        pred = pd.DataFrame()
        pred.loc[:, ["variantid1", "variantid2"]] = test.select(
            ["variantid1", "variantid2"]
        ).to_pandas()
        pred.loc[:, "target"] = np.mean(
            [model.predict_proba(test_pool)[:, 1] for model in models], axis=0
        )

        return pred

    def predict_oof(self, train: pl.DataFrame) -> pd.DataFrame:
        return pd.concat(
            [
                self.predict(train[val_idx], fold=fold).assign(fold=fold)
                for fold, (_, val_idx) in enumerate(self.splitter(train))
            ]
        ).reset_index(drop=True)

    @property
    def feature_importances(self) -> pd.Series:
        fi = 0
        for model in self.models_:
            fi += pd.Series(
                model.feature_importances_, index=model.feature_names_
            ) / len(self.models_)
        return fi.sort_values(ascending=False)

    def to_pool(self, dataset: pl.DataFrame) -> cb.Pool:
        return cb.Pool(
            data=dataset.select(
                pl.exclude(["variantid1", "variantid2", "target", "cat3_grouped"])
            ).to_pandas(),
            label=dataset["target"].to_pandas()
            if "target" in dataset.columns
            else None,
            **self.pool_params,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for fold, model in enumerate(self.models_):
            model.save_model(str(path / f"fold-{fold}.cbm"))
        with open(path / "metrics.json", "w") as f:
            json.dump(self.metrics_, f)

    @classmethod
    def from_snapshot(cls, path: str | Path, **kwargs) -> CatBoostCV:
        model = cls(**kwargs)

        path = Path(path)
        folds = [p for p in path.iterdir() if p.name.startswith("fold")]

        models = []
        for fold in sorted(folds):
            fold_model = cb.CatBoostClassifier()
            fold_model.load_model(str(fold))
            models.append(fold_model)

        with open(path / "metrics.json", "r") as f:
            metrics = json.load(f)

        model.models_ = models
        model.metrics_ = metrics
        return model
