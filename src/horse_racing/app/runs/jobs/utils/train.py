import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Any

import polars as pl

from horse_racing.core.gcp.storage import StorageClient
from horse_racing.infrastructure.netkeiba.race_result import RaceResultNetkeibaRepository
from horse_racing.usecase.race_result import (
    RaceResultUsecase,
    ResultColumn,
    GENDER_AGE_COLUMN,
    HORSE_WEIGHT_AND_DIFF_COLUMN,
)


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


def upload_data(data: dict[str, pl.DataFrame], storage_client: StorageClient, version: str) -> None:
    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        race_result_repository = RaceResultNetkeibaRepository(
            storage_client=storage_client,
            root_dir=tmp_dir,
        )
        for k, df in data.items():
            path = tmp_dir / f"{k}.parquet"
            df.write_parquet(path)
            race_result_repository.upload_data_to_storage(path=path, version=version)


def split_train_data(
    data_df: pl.DataFrame,
    train_first_date: str,
    train_last_date: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    # split train and valid
    train_df = data_df.filter((pl.col("race_date") >= train_first_date) & (pl.col("race_date") <= train_last_date))
    valid_df = data_df.filter(pl.col("race_date") > train_last_date)
    return train_df, valid_df


def upload_model(
    model_path: Path,
    best_params: dict[str, Any],
    metric: dict[str, float],
    tmp_dir: Path,
    storage_client: StorageClient,
    model_name: str,
    version: str,
) -> None:
    bucket = storage_client.get_bucket("yukob-horse-racing-models")
    prefix = f"{model_name}/model_version={version}"

    # upload model
    bucket.blob(f"{prefix}/model.txt").upload_from_filename(str(model_path))

    # upload params
    param_path = tmp_dir / "params.json"
    with open(param_path, "w") as fp:
        json.dump(best_params, fp)
    bucket.blob(f"{prefix}/params.json").upload_from_filename(str(param_path))

    # upload metric
    metric_path = tmp_dir / "metrics.json"
    with open(metric_path, "w") as fp:
        json.dump(metric, fp)
    bucket.blob(f"{prefix}/metrics.json").upload_from_filename(str(metric_path))
