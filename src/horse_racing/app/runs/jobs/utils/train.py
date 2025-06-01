import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Any

import polars as pl

from horse_racing.core.gcp.storage import StorageClient
from horse_racing.core.logging import logger
from horse_racing.infrastructure.netkeiba.race_result import RaceResultNetkeibaRepository
from horse_racing.usecase.race_result import (
    RaceResultUsecase,
    ResultColumn,
    GENDER_AGE_COLUMN,
    HORSE_WEIGHT_AND_DIFF_COLUMN,
)

FIELD_TYPES = (
    0,  # "芝"
    1,  # "ダ"
    2,  # "障害"
)
DISTANCES = (
    1000,
    1150,
    1200,
    1300,
    1400,
    1500,
    1600,
    1700,
    1800,
    1900,
    2000,
    2100,
    2200,
    2400,
    2600,
)
RACE_PLACES = (
    "福島",
    "新潟",
    "中山",
    "小倉",
    "東京",
    "阪神",
    "函館",
    "中京",
    "札幌",
    "京都",
)


@dataclass
class TrainConfig:
    model: str

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
            ResultColumn.ODDS,  # TODO: use predicted odds
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


def _calculate_grouped_stats(
    base_df: pl.DataFrame,
    group_column: str,
    group_value: int | str,
    rank_condition: pl.Expr,
    prefix: str,
) -> pl.DataFrame:
    count_df = (
        base_df.filter(pl.col(group_column) == group_value)
        .group_by(ResultColumn.JOCKEY_ID)
        .agg(
            pl.col(ResultColumn.RANK)
            .filter(rank_condition)
            .count()
            .alias(f"{prefix}_count_{group_column}_{group_value}")
        )
    )

    rate_df = (
        base_df.filter(pl.col(group_column) == group_value)
        .group_by(ResultColumn.JOCKEY_ID)
        .agg(
            (pl.col(ResultColumn.RANK).filter(rank_condition).count() / pl.col(ResultColumn.RANK).count()).alias(
                f"{prefix}_rate_{group_column}_{group_value}"
            )
        )
    )

    return count_df.join(rate_df, on=ResultColumn.JOCKEY_ID, how="left")


def _calculate_stats(
    base_df: pl.DataFrame, race_count_df: pl.DataFrame, rank_condition: pl.Expr, ticket_type: str
) -> pl.DataFrame:
    """Helper function to calculate win/show statistics"""
    # Overall stats
    count_df = base_df.group_by(ResultColumn.JOCKEY_ID).agg(
        pl.col(ResultColumn.RANK).filter(rank_condition).count().alias(f"{ResultColumn.JOCKEY_ID}_{ticket_type}_count")
    )

    rate_df = race_count_df.join(count_df, on=ResultColumn.JOCKEY_ID, how="left").select(
        pl.col(ResultColumn.JOCKEY_ID),
        (pl.col(f"{ResultColumn.JOCKEY_ID}_{ticket_type}_count") / pl.col("race_count")).alias(
            f"{ResultColumn.JOCKEY_ID}_{ticket_type}_rate"
        ),
    )

    result_df = count_df.join(rate_df, on=ResultColumn.JOCKEY_ID, how="left")

    # Field type stats
    for field_type in FIELD_TYPES:
        prefix = f"{ResultColumn.JOCKEY_ID}_{ticket_type}"
        actual_field_types = base_df[ResultColumn.FIELD_TYPE].unique().to_list()

        if field_type in actual_field_types:
            field_type_stat_df = _calculate_grouped_stats(
                base_df,
                group_column=ResultColumn.FIELD_TYPE,
                group_value=field_type,
                rank_condition=rank_condition,
                prefix=prefix,
            )
            result_df = result_df.join(field_type_stat_df, on=ResultColumn.JOCKEY_ID, how="left")
        else:
            logger.info("%s not exists in %s and fill with null", field_type, actual_field_types)
            result_df[f"{prefix}_{field_type}_count"] = None

    # Distance stats
    for distance in DISTANCES:
        prefix = f"{ResultColumn.JOCKEY_ID}_{ticket_type}"
        actual_distances = base_df[ResultColumn.DISTANCE].unique().to_list()

        if distance in actual_distances:
            distance_stat_df = _calculate_grouped_stats(
                base_df,
                group_column=ResultColumn.DISTANCE,
                group_value=distance,
                rank_condition=rank_condition,
                prefix=prefix,
            )
            result_df = result_df.join(distance_stat_df, on=ResultColumn.JOCKEY_ID, how="left")
        else:
            logger.info("%s not exists in %s and fill with null", distance, actual_distances)
            result_df[f"{prefix}_{distance}_count"] = None

    # race_place stats
    for race_place in RACE_PLACES:
        prefix = f"{ResultColumn.JOCKEY_ID}_{ticket_type}"
        actual_race_places = base_df[ResultColumn.RACE_PLACE].unique().to_list()

        if race_place in actual_race_places:
            race_place_stat_df = _calculate_grouped_stats(
                base_df,
                group_column=ResultColumn.RACE_PLACE,
                group_value=race_place,
                rank_condition=rank_condition,
                prefix=prefix,
            )
            result_df = result_df.join(race_place_stat_df, on=ResultColumn.JOCKEY_ID, how="left")
        else:
            logger.info("%s not exists in %s and fill with null", race_place, actual_race_places)
            result_df[f"{prefix}_{race_place}_count"] = None

    return result_df


def _agg_jockey(df: pl.DataFrame) -> pl.DataFrame:
    # Base dataframe with minimal columns
    jockey_base_df = df.select(
        [
            pl.col(ResultColumn.RANK).cast(pl.Int32),
            pl.col(ResultColumn.JOCKEY_ID).cast(pl.String),
            pl.col(ResultColumn.FIELD_TYPE),
            pl.col(ResultColumn.DISTANCE),
            pl.col(ResultColumn.RACE_PLACE),
        ]
    )

    # Calculate race counts once
    jockey_race_count_df = jockey_base_df.group_by(ResultColumn.JOCKEY_ID).agg(
        pl.col(ResultColumn.RANK).count().alias("race_count")
    )

    # Define conditions once
    win_rank_condition = pl.col(ResultColumn.RANK) == 1
    show_rank_condition = (pl.col(ResultColumn.RANK) >= 1) & (pl.col(ResultColumn.RANK) <= 3)

    # Calculate win stats
    logger.info("Calculating jockey win count and rate...")
    win_stats = _calculate_stats(
        base_df=jockey_base_df,
        race_count_df=jockey_race_count_df,
        rank_condition=win_rank_condition,
        ticket_type="win",
    )
    logger.info("win count and rate features calculated:\n%s", win_stats)

    # Calculate show stats
    logger.info("Calculating jockey show count and rate...")
    show_stats = _calculate_stats(
        base_df=jockey_base_df,
        race_count_df=jockey_race_count_df,
        rank_condition=show_rank_condition,
        ticket_type="show",
    )
    logger.info("show count and rate features calculated:\n%s", show_stats)

    # Combine all stats
    logger.info("Combining win and show stats...")
    return win_stats.join(show_stats, on=ResultColumn.JOCKEY_ID, how="left")


def preprocess(
    raw_df: pl.DataFrame,
    feature_columns: list[str],
    weight_diff_avg_df: pl.DataFrame | None = None,
    jockey_df: pl.DataFrame | None = None,
    mode: Literal["train", "predict"] = "train",
) -> dict[str, pl.DataFrame]:
    df = _remove_debut_race(raw_df)
    df = df.drop("race_name")

    # filter race
    select_exprs = [
        pl.col(ResultColumn.HORSE_NUMBER).cast(pl.Int32),
        pl.col(ResultColumn.FRAME).cast(pl.Int32),
        pl.col(GENDER_AGE_COLUMN).alias(ResultColumn.GENDER),  # 後ほどextractされる
        pl.col(GENDER_AGE_COLUMN).str.extract(r"(\d+)").cast(pl.Int32).alias(ResultColumn.AGE),
        pl.col(ResultColumn.TOTAL_WEIGHT).cast(pl.Float64).alias(ResultColumn.TOTAL_WEIGHT),
        # race
        pl.col(ResultColumn.RACE_NUMBER).str.extract(r"(\d+)").cast(pl.Int32).alias(ResultColumn.RACE_NUMBER),
        pl.col(ResultColumn.RACE_PLACE).cast(pl.String),
        pl.col(ResultColumn.RACE_CLASS).cast(pl.String),
        pl.col(ResultColumn.START_AT).str.extract(r"^(\d+)").cast(pl.Int32).alias(ResultColumn.START_AT),
        pl.col(ResultColumn.DISTANCE).str.extract(r"(\d+)").cast(pl.Int32).alias(ResultColumn.DISTANCE),
        pl.col(ResultColumn.DISTANCE).alias(ResultColumn.ROTATE),
        pl.col(ResultColumn.DISTANCE).alias(ResultColumn.FIELD_TYPE),
        pl.col(ResultColumn.WEATHER),
        pl.col(ResultColumn.FIELD_CONDITION),
        # fresh
        pl.col(ResultColumn.POPULAR).cast(pl.Int32).alias(ResultColumn.POPULAR),
        pl.col(ResultColumn.ODDS).cast(pl.Float32).alias(ResultColumn.ODDS),  # TODO: use predicted odds
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
    if ResultColumn.HORSE_WEIGHT_DIFF_DEV in feature_columns:
        select_exprs.append(
            pl.col(HORSE_WEIGHT_AND_DIFF_COLUMN)
            .str.extract(r"\(([-\+\d]+)\)")
            .cast(pl.Int32)
            .alias(ResultColumn.HORSE_WEIGHT_DIFF)
        )
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
            for c in (
                ResultColumn.HORSE_ID,
                ResultColumn.JOCKEY_ID,
                ResultColumn.TRAINER_ID,
                ResultColumn.RACE_PLACE,
                ResultColumn.RACE_CLASS,
            )
        ]
    )

    # jockey target encoding
    if jockey_df is None:
        if mode == "train":
            logger.info("Preprocessing jockey features...")
            jockey_df = _agg_jockey(df)
        else:
            raise ValueError("mode is not train, but jockey_df is not provided")
    df = df.join(jockey_df, on=ResultColumn.JOCKEY_ID, how="left")

    # weight dev
    logger.info("Preprocessing horse weight difference average...")
    if ResultColumn.HORSE_WEIGHT_DIFF_DEV in feature_columns and weight_diff_avg_df is None:
        if mode == "train":
            weight_diff_avg_df = df.group_by("horse_id").agg(
                pl.mean(ResultColumn.HORSE_WEIGHT_DIFF).alias("weight_diff_avg")
            )
        else:
            raise ValueError("mode is not train, but weight_diff_avg_df is None")

    if weight_diff_avg_df is not None:
        df = df.join(weight_diff_avg_df, on="horse_id", how="left")
        df = df.with_columns(
            (pl.col(ResultColumn.HORSE_WEIGHT_DIFF) - pl.col("weight_diff_avg")).alias(
                ResultColumn.HORSE_WEIGHT_DIFF_DEV
            )
        )

    return {
        "feature": df,
        "weight_diff_avg": weight_diff_avg_df,
        "jockey": jockey_df,
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
    feature_columns: list[str],
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

    feature_columns_path = tmp_dir / "feature_columns.json"
    with open(feature_columns_path, "w") as fp:
        json.dump(feature_columns, fp)

    # upload metric
    metric_path = tmp_dir / "metrics.json"
    with open(metric_path, "w") as fp:
        json.dump(metric, fp)
    bucket.blob(f"{prefix}/metrics.json").upload_from_filename(str(metric_path))
