import json
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Any

import optuna
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from horse_racing.core.gcp.storage import StorageClient
from horse_racing.core.logging import logger
from horse_racing.infrastructure.netkeiba.race_result import RaceResultNetkeibaRepository
from horse_racing.usecase.race_result import (
    RaceResultUsecase,
    ResultColumn,
    GENDER_AGE_COLUMN,
    HORSE_WEIGHT_AND_DIFF_COLUMN,
)


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


def collect_data(
    storage_client: StorageClient,
    version: str,
    first_date: str,
    last_date: str,
    tmp_dir: Path,
) -> pl.DataFrame:
    result_repository = RaceResultNetkeibaRepository(
        storage_client=storage_client,
        root_dir=tmp_dir,
    )
    result_usecase = RaceResultUsecase(race_result_repository=result_repository, root_dir=tmp_dir)
    return result_usecase.get(version=version, first_date=first_date, last_date=last_date)


def _label_encode(df: pl.DataFrame, column: str, label_dict: dict[str, int]) -> pl.DataFrame:
    return df.with_columns(
        pl.col(column)
        .str.extract(rf'({"|".join(list(label_dict))})')
        .replace_strict(label_dict, default=-1)
        .cast(pl.Int8)
    )


def _remove_debut_race(df: pl.DataFrame) -> pl.DataFrame:
    df = df.filter(~pl.col("race_name").str.contains("新馬"))
    return df


def preprocess(
    raw_df: pl.DataFrame,
    weight_diff_avg_df: pl.DataFrame | None = None,
    mode: Literal["train", "predict"] = "train",
) -> dict[str, pl.DataFrame]:
    df = _remove_debut_race(raw_df)
    df = df.drop("race_name")

    # filter race
    select_exprs = [
        pl.col(ResultColumn.HORSE_NUMBER).cast(pl.Int32),
        pl.col(ResultColumn.FRAME).cast(pl.Int32),
        pl.col(GENDER_AGE_COLUMN).alias(ResultColumn.GENDER),
        pl.col(GENDER_AGE_COLUMN).str.extract(r"(\d+)").cast(pl.Int32).alias(ResultColumn.AGE),
        pl.col(ResultColumn.TOTAL_WEIGHT).cast(pl.Float64).alias(ResultColumn.TOTAL_WEIGHT),
        # race
        pl.col(ResultColumn.RACE_NUMBER).str.extract(r"(\d+)").cast(pl.Int32).alias(ResultColumn.RACE_NUMBER),
        pl.col(ResultColumn.START_AT).str.extract(r"^(\d+)").cast(pl.Int32).alias(ResultColumn.START_AT),
        pl.col(ResultColumn.DISTANCE).str.extract(r"(\d+)").cast(pl.Int32).alias(ResultColumn.DISTANCE),
        pl.col(ResultColumn.DISTANCE).alias(ResultColumn.ROTATE),
        pl.col(ResultColumn.DISTANCE).alias(ResultColumn.FIELD_TYPE),
        pl.col(ResultColumn.WEATHER),
        pl.col(ResultColumn.FIELD_CONDITION),
        # fresh
        pl.col(ResultColumn.POPULAR).cast(pl.Int32).alias(ResultColumn.POPULAR),
        pl.col(ResultColumn.ODDS).cast(pl.Float32).alias(ResultColumn.ODDS),
        (
            pl.col(HORSE_WEIGHT_AND_DIFF_COLUMN)
            .str.extract(r"\(([-\+\d]+)\)")
            .cast(pl.Int32)
            .alias(ResultColumn.HORSE_WEIGHT_DIFF)
        ),
        # not feature
        pl.col(ResultColumn.RACE_ID),
        pl.col(ResultColumn.RACE_DATE),
        pl.col(ResultColumn.HORSE_NAME),
        pl.col(ResultColumn.HORSE_ID),
        pl.col(ResultColumn.JOCKEY_ID),
        pl.col(ResultColumn.TRAINER_ID),
    ]
    if mode == "train":
        df = df.filter(~pl.col(ResultColumn.RANK).is_in({"中止", "除外", "取消"}))
        select_exprs.append(pl.col(ResultColumn.RANK).cast(pl.Int32))
    df = df.select(select_exprs)

    # label encoding
    gender_label_dict = {"牝": 0, "牡": 1, "セ": 2}
    df = _label_encode(df=df, column=ResultColumn.GENDER, label_dict=gender_label_dict)

    df = _label_encode(df=df, column=ResultColumn.ROTATE, label_dict={"左": 0, "右": 1})
    df = _label_encode(df=df, column=ResultColumn.FIELD_TYPE, label_dict={"芝": 0, "ダ": 1, "障": 2})
    df = _label_encode(
        df=df,
        column=ResultColumn.WEATHER,
        label_dict={"晴": 0, "曇": 1, "小雨": 2, "雨": 3, "小雪": 4, "雪": 5},
    )
    df = _label_encode(
        df=df,
        column=ResultColumn.FIELD_CONDITION,
        label_dict={"良": 0, "稍": 1, "重": 2, "不": 3, "未": 4},
    )

    # category type
    df = df.with_columns(
        [
            pl.col(c).cast(pl.Categorical).alias(f"{c}_cat")
            for c in (ResultColumn.HORSE_ID, ResultColumn.JOCKEY_ID, ResultColumn.TRAINER_ID)
        ]
    )

    # weight dev
    if mode == "train":
        weight_diff_avg_df = df.group_by("horse_id").agg(
            pl.mean(ResultColumn.HORSE_WEIGHT_DIFF).alias("weight_diff_avg")
        )
    elif weight_diff_avg_df is None:
        raise ValueError("mode is not train, but weight_diff_avg_df is None")

    df = df.join(weight_diff_avg_df, on="horse_id", how="left")
    df = df.with_columns(
        (pl.col(ResultColumn.HORSE_WEIGHT_DIFF) - pl.col("weight_diff_avg")).alias(ResultColumn.HORSE_WEIGHT_DIFF_DEV)
    )
    return {
        "feature": df,
        "weight_diff_avg": weight_diff_avg_df,
    }


def split_train_data(
    data_df: pl.DataFrame,
    train_first_date: str,
    train_last_date: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    # split train and valid
    train_df = data_df.filter((pl.col("race_date") >= train_first_date) & (pl.col("race_date") <= train_last_date))
    valid_df = data_df.filter(pl.col("race_date") > train_last_date)
    return train_df, valid_df


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
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "feature_pre_filter": False,
    }

    def objective(trial: optuna.Trial) -> float:
        params = {
            **const_params,
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1e-1),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
            "feature_fraction": trial.suggest_uniform("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        }

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
    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        race_result_repository = RaceResultNetkeibaRepository(
            storage_client=storage_client,
            root_dir=tmp_dir,
        )
        for k, df in data.items():
            path = tmp_dir / f"{k}.parquet"
            df.write_parquet(path)
            race_result_repository.upload_data_to_storage(path=path, version=args.data_version)
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
    bucket = storage_client.get_bucket("yukob-horse-racing-models")
    with TemporaryDirectory() as tmp_dir_str:
        model_path = Path(tmp_dir_str) / "model.txt"
        model.save_model(model_path)
        bucket.blob(f"model_version={args.model_version}/model.txt").upload_from_filename(str(model_path))

        param_path = Path(tmp_dir_str) / "params.json"
        with open(param_path, "w") as fp:
            json.dump(best_params, fp)
        bucket.blob(f"model_version={args.model_version}/params.json").upload_from_filename(str(param_path))

        metric_path = Path(tmp_dir_str) / "metrics.json"
        with open(metric_path, "w") as fp:
            json.dump(metric, fp)
        bucket.blob(f"model_version={args.model_version}/metrics.json").upload_from_filename(str(metric_path))


if __name__ == "__main__":
    main()
