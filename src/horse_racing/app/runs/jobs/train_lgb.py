from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Any

import lightgbm as lgb
import optuna
import polars as pl
from sklearn.metrics import roc_auc_score

from horse_racing.app.runs.jobs.utils.train import split_train_data, preprocess, collect_data, upload_data, upload_model
from horse_racing.core.gcp.storage import StorageClient
from horse_racing.core.logging import logger
from horse_racing.usecase.race_result import ResultColumn


@dataclass
class TrainConfig:
    train_first_date: str = ""
    train_last_date: str = ""
    valid_last_date: str = ""
    data_version: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    _feature_columns: str = ",".join(
        [
            ResultColumn.HORSE_NUMBER,
            ResultColumn.FRAME,
            ResultColumn.AGE,
            ResultColumn.GENDER,
            ResultColumn.TOTAL_WEIGHT,
            ResultColumn.ODDS,
            ResultColumn.HORSE_WEIGHT_DIFF_DEV,
            # categorical
            f"{ResultColumn.HORSE_ID}_cat",
            f"{ResultColumn.JOCKEY_ID}_cat",
            f"{ResultColumn.TRAINER_ID}_cat",
            # race info
            ResultColumn.RACE_NUMBER,
            ResultColumn.START_AT,
            ResultColumn.DISTANCE,
            ResultColumn.ROTATE,
            ResultColumn.FIELD_TYPE,
            ResultColumn.WEATHER,
            ResultColumn.FIELD_CONDITION,
        ]
    )

    # constants
    model: str = "lightgbm"
    model_version: str = datetime.now().strftime("%Y%m%d_%H%M%S")

    @property
    def feature_columns(self) -> list[str]:
        return self._feature_columns.split(",")


def train(
    params: dict[str, Any],
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
    feature_columns: list[str],
) -> tuple[lgb.Booster, dict[str, float]]:
    train_feature_df = train_df.select(feature_columns)
    train_label = (train_df[ResultColumn.RANK] == "1").cast(int).to_numpy()
    valid_feature_df = valid_df.select(feature_columns)
    valid_label = (valid_df[ResultColumn.RANK] == "1").cast(int).to_numpy()

    ds_train = lgb.Dataset(train_feature_df.to_pandas(), label=train_label)
    ds_valid = lgb.Dataset(valid_feature_df.to_pandas(), label=valid_label)

    model = lgb.train(
        params,
        ds_train,
        num_boost_round=300,
        valid_names=["train", "valid"],
        valid_sets=[ds_train, ds_valid],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=20,
                # verbose=True,
            ),
        ],
    )

    # logging feature importance
    importance_type = "gain"
    importance_values = model.feature_importance(importance_type=importance_type)
    importance_dict = {name: importance for name, importance in zip(feature_columns, importance_values)}
    logger.info(f"feature importance ({importance_type}): {importance_dict}")

    y_pred = model.predict(valid_feature_df.to_pandas())
    y_true = ds_valid.label
    auc = roc_auc_score(y_true=y_true, y_score=y_pred)
    return model, {"auc": auc}


def tune_hyper_params(
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
    feature_columns: list[str],
    metric_key: str,
    n_trials: int,
    direction: Literal["maximize", "minimize"],
) -> dict[str, Any]:
    const_params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
    }
    param_settings = {
        # "learning_rate": ("suggest_loguniform", {"low": 1e-3, "high": 1e-1}),
        "num_leaves": ("suggest_int", {"low": 20, "high": 150}),
        "min_data_in_leaf": ("suggest_int", {"low": 10, "high": 100}),
        "feature_fraction": ("suggest_uniform", {"low": 0.5, "high": 1.0}),
        # "bagging_fraction": ("suggest_uniform", {"low": 0.5, "high": 1.0}),
        # "bagging_freq": ("suggest_int", {"low": 1, "high": 7}),
    }

    def objective(trial: optuna.Trial) -> float:
        suggest_fn_dict = {
            "suggest_int": trial.suggest_int,
            "suggest_loguniform": trial.suggest_loguniform,
            "suggest_uniform": trial.suggest_uniform,
        }

        params = dict(**const_params)
        for k, (fn_name, kw) in param_settings.items():
            params[k] = suggest_fn_dict[fn_name](k, **kw)

        _, metric = train(params=params, train_df=train_df, valid_df=valid_df, feature_columns=feature_columns)
        return metric[metric_key]

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)
    return dict(**study.best_params)


def main() -> None:
    parser = ArgumentParser()

    # required
    parser.add_argument("--train-first-date", type=str)
    parser.add_argument("--train-last-date", type=str)
    parser.add_argument("--valid-last-date", type=str)

    # optional
    parser.add_argument("--data-version", type=str)

    args, _ = parser.parse_known_args(namespace=TrainConfig())
    logger.info(f"{args=}")

    storage_client = StorageClient()

    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        raw_df = collect_data(
            storage_client=storage_client,
            version=args.data_version,
            tmp_dir=tmp_dir,
            first_date=args.train_first_date,
            last_date=args.valid_last_date,
        )
        logger.info(f"raw_df: {raw_df.shape}, {raw_df.columns}")

    # preprocess
    data = preprocess(raw_df=raw_df)
    upload_data(data=data, storage_client=storage_client, version=args.data_version)
    processed_df = data["feature"]

    # split train and valid
    train_df, valid_df = split_train_data(
        data_df=processed_df,
        train_last_date=args.train_last_date,
        train_first_date=args.train_first_date,
    )

    # hyper param tuning
    best_params = tune_hyper_params(
        train_df=train_df,
        valid_df=valid_df,
        feature_columns=args.feature_columns,
        metric_key="auc",
        direction="maximize",
        n_trials=100,
    )

    # train
    model, metric = train(
        params=best_params,
        train_df=train_df,
        valid_df=valid_df,
        feature_columns=args.feature_columns,
    )

    # save model
    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        model_path = tmp_dir / "model.txt"
        model.save_model(str(model_path))
        upload_model(
            model_path=model_path,
            best_params=best_params,
            metric=metric,
            tmp_dir=tmp_dir,
            storage_client=storage_client,
            model_name=args.model,
            version=args.model_version,
        )


if __name__ == "__main__":
    main()
