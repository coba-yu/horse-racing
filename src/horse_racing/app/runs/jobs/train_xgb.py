import math
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Any

import numpy as np
import optuna
import polars as pl
import xgboost as xgb
from sklearn.metrics import roc_auc_score, mean_squared_error

from horse_racing.app.runs.jobs.utils.train import (
    Target,
    split_train_data,
    preprocess,
    collect_data,
    upload_data,
    upload_model,
    TrainConfig,
)
from horse_racing.core.gcp.storage import StorageClient
from horse_racing.core.logging import logger
from horse_racing.usecase.race_result import ResultColumn


@dataclass
class TrainResult:
    model: xgb.Booster
    metric: dict[str, float]
    importance: dict[str, dict[str, float]]


def _get_target_rank_win(df: pl.DataFrame) -> np.ndarray:
    return (df[ResultColumn.RANK] == "1").cast(int).to_numpy()


def _get_target_rank_show(df: pl.DataFrame) -> np.ndarray:
    return (
        pl.when(pl.col(ResultColumn.RANK).is_in(["1", "2", "3"])).then(pl.col(ResultColumn.RANK).cast(int)).otherwise(0)
    ).to_numpy()


def _get_target_odds(df: pl.DataFrame) -> np.ndarray:
    return (df[ResultColumn.ODDS]).cast(pl.Float64).to_numpy()


def train(
    params: dict[str, Any],
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
    feature_columns: list[str],
    target: str = Target.RANK_WIN,
    num_boost_round: int = 300,
    early_stopping_rounds: int = 20,
) -> TrainResult:
    train_feature_df = train_df.select(feature_columns)
    valid_feature_df = valid_df.select(feature_columns)

    if target == Target.RANK_WIN:
        params["objective"] = "binary:logistic"
        train_label = _get_target_rank_win(train_df)
        valid_label = _get_target_rank_win(valid_df)
    elif target == Target.RANK_SHOW:
        params["objective"] = "multi:softprob"
        params["num_class"] = 4
        train_label = _get_target_rank_show(train_df)
        valid_label = _get_target_rank_show(valid_df)
    elif target == Target.ODDS:
        params["objective"] = "reg:squarederror"
        train_label = _get_target_odds(train_df)
        valid_label = _get_target_odds(valid_df)
    else:
        raise ValueError(f"Invalid target: {target}")

    ds_train = xgb.DMatrix(train_feature_df.to_pandas(), label=train_label, enable_categorical=True)
    ds_valid = xgb.DMatrix(valid_feature_df.to_pandas(), label=valid_label, enable_categorical=True)

    model = xgb.train(
        params,
        ds_train,
        num_boost_round=num_boost_round,
        evals=[(ds_train, "train"), (ds_valid, "valid")],
        early_stopping_rounds=early_stopping_rounds,
    )

    y_pred = model.predict(ds_valid)
    y_true = ds_valid.get_label()
    if target == Target.RANK_WIN:
        auc = roc_auc_score(y_true=y_true, y_score=y_pred)
        metric = {"valid_auc": auc}
    elif target == Target.ODDS:
        mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
        rmse = math.sqrt(mse)
        metric = {"valid_rmse": rmse}
    else:
        raise ValueError(f"Invalid target: {target}")

    importance_types = ("weight", "gain", "cover")
    importance_dict = {it: model.get_score(importance_type=it) for it in importance_types}

    return TrainResult(
        model=model,
        metric=metric,
        importance=importance_dict,
    )


def tune_hyper_params(
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
    feature_columns: list[str],
    n_trials: int,
    direction: Literal["maximize", "minimize"],
    booster_type: Literal["gbtree", "dart", "gblinear"] = "gbtree",
    target: str = Target.RANK_WIN,
) -> dict[str, Any]:
    const_params = {
        "booster": booster_type,
        "tree_method": "hist",
    }
    if target == Target.RANK_WIN:
        const_params["eval_metric"] = "auc"
    elif target == Target.RANK_SHOW:
        const_params["eval_metric"] = "mlogloss"
    elif target == Target.ODDS:
        const_params["eval_metric"] = "rmse"
    else:
        raise ValueError(f"Invalid target: {target}")

    param_settings = {
        "eta": ("suggest_float", {"low": 1e-2, "high": 0.2, "log": True}),
        "max_depth": ("suggest_int", {"low": 3, "high": 8}),
        # overfitを防ぐために小さめに設定
        "subsample": ("suggest_float", {"low": 0.5, "high": 0.8, "step": 0.1}),
        # overfitを防ぐために小さめに設定
        "colsample_bytree": ("suggest_float", {"low": 0.5, "high": 0.8, "step": 0.1}),
        "gamma": ("suggest_int", {"low": 0, "high": 5}),
        "reg_alpha": ("suggest_float", {"low": 1e-8, "high": 10.0, "log": True}),
        "reg_lambda": ("suggest_float", {"low": 1.0, "high": 100.0, "log": True}),
    }

    def objective(trial: optuna.Trial) -> float:
        suggest_fn_dict = {
            "suggest_int": trial.suggest_int,
            "suggest_float": trial.suggest_float,
        }

        params = dict(**const_params)
        for k, (fn_name, kw) in param_settings.items():
            params[k] = suggest_fn_dict[fn_name](k, **kw)

        train_result = train(
            params=params,
            train_df=train_df,
            valid_df=valid_df,
            feature_columns=feature_columns,
            target=target,
        )
        metric_key = f'valid_{const_params["eval_metric"]}'
        return train_result.metric[metric_key]

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)
    return dict(**const_params, **study.best_params)


def main() -> None:
    parser = ArgumentParser()

    # required
    parser.add_argument("--train-first-date", type=str)
    parser.add_argument("--train-last-date", type=str)
    parser.add_argument("--valid-last-date", type=str)

    # optional
    parser.add_argument("--data-version", type=str)

    args, _ = parser.parse_known_args(
        namespace=TrainConfig(model="xgboost"),
    )
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
    data = preprocess(raw_df=raw_df, feature_columns=args.feature_columns)
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
        direction="maximize",
        n_trials=100,
    )

    # train
    train_result = train(
        params=best_params,
        train_df=train_df,
        valid_df=valid_df,
        feature_columns=args.feature_columns,
    )

    # save model
    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        model_path = tmp_dir / "model.txt"
        train_result.model.save_model(str(model_path))
        upload_model(
            model_path=model_path,
            best_params=best_params,
            metric=train_result.metric,
            feature_columns=args.feature_columns,
            importance=train_result.importance,
            tmp_dir=tmp_dir,
            storage_client=storage_client,
            model_name=args.model,
            version=args.model_version,
        )


if __name__ == "__main__":
    main()
