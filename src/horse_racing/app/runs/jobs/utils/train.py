import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Any

import polars as pl

from horse_racing.app.runs.jobs.utils.preprocess import NUM_CORNERS, select_base_columns
from horse_racing.core.datetime import get_current_yyyymmdd_hhmmss
from horse_racing.core.gcp.storage import StorageClient
from horse_racing.core.logging import logger
from horse_racing.infrastructure.netkeiba.race_result import RaceResultNetkeibaRepository
from horse_racing.usecase.race_result import (
    RaceResultUsecase,
    ResultColumn,
)

FIELD_TYPES = (
    0,  # "芝"
    1,  # "ダ"
    2,  # "障害"
)
DISTANCE_CLASSES = (
    "sprint",
    "mile",
    "middle",
    "middle_to_long",
    "long",
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
    data_version: str = get_current_yyyymmdd_hhmmss()
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
    model_version: str = get_current_yyyymmdd_hhmmss()

    @property
    def feature_columns(self) -> list[str]:
        return self._feature_columns.split(",")


class Target:
    RANK_WIN: str = "rank_win"
    RANK_SHOW: str = "rank_show"
    ODDS: str = "odds"


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


def _remove_debut_race(df: pl.DataFrame) -> pl.DataFrame:
    df = df.filter(~pl.col("race_name").str.contains("新馬"))
    return df


def get_win_rank_condition() -> pl.Expr:
    return pl.col(ResultColumn.RANK) == 1


def get_win_label_expr() -> pl.Expr:
    return pl.when(get_win_rank_condition()).then(1).otherwise(0).alias(ResultColumn.WIN_LABEL)


def get_show_rank_condition() -> pl.Expr:
    return (pl.col(ResultColumn.RANK) >= 1) & (pl.col(ResultColumn.RANK) <= 3)


def get_show_label_expr() -> pl.Expr:
    return pl.when(get_show_rank_condition()).then(1).otherwise(0).alias(ResultColumn.SHOW_LABEL)


def get_inverse_rank_expr() -> pl.Expr:
    return (pl.lit(1.0, dtype=pl.Float64) / pl.col(ResultColumn.RANK)).cast(pl.Float64).alias(ResultColumn.INVERSE_RANK)


def get_inverse_rank_log2_expr() -> pl.Expr:
    return (
        (pl.lit(1.0, dtype=pl.Float64) / (pl.col(ResultColumn.RANK) + pl.lit(1, dtype=pl.Int32)).log(base=2))
        .cast(pl.Float64)
        .alias(ResultColumn.INVERSE_RANK_LOG2)
    )


def get_rank_clipped_expr(
    raw_column: str = ResultColumn.RANK,
    max_rank: int = 6,
) -> pl.Expr:
    return (
        pl.when(pl.col(raw_column) > max_rank)
        .then(max_rank)
        .otherwise(pl.col(raw_column))
        .alias(f"{raw_column}_clipped")
    )


def _label_encode(df: pl.DataFrame, column: str, label_dict: dict[str, int]) -> pl.DataFrame:
    return df.with_columns(
        pl.col(column)
        .cast(pl.Utf8)
        .str.extract(rf'({"|".join(list(label_dict))})')
        .replace_strict(label_dict, default=-1)
        .cast(pl.Int8)
    )


def _format_last_column_name(raw_column: str, prefix: str = "horse_id_", n_shift: int = 0) -> str:
    return f"{prefix}last{n_shift + 1}_{raw_column}"


def _shift_horse_result_expr(raw_column: str, prefix: str = "horse_id_", n_shift: int = 0) -> pl.Expr:
    alias_name = _format_last_column_name(raw_column=raw_column, prefix=prefix, n_shift=n_shift)
    return (
        pl.col(raw_column).sort_by(ResultColumn.RACE_DATE).shift(n_shift).over(ResultColumn.HORSE_ID).alias(alias_name)
    )


def preprocess_horse(
    df: pl.DataFrame,
    feature_columns: list[str],
    num_corners: int = NUM_CORNERS,
) -> pl.DataFrame:
    # [previous]
    # - horse_id_last1_rank
    # - horse_id_last1_distance
    # - horse_id_last1_corner_{i}_rank
    # - horse_id_last1_goal_time
    # - horse_id_last1_goal_speed
    # - horse_id_last1_last_3f_time
    # - horse_id_last1_corner_rank_relative_avg
    logger.info("Calculating previous result...")
    previous_mean_feature_columns = [
        ResultColumn.GOAL_TIME,
        ResultColumn.GOAL_SPEED,
        ResultColumn.LAST_3F_TIME,
        f"{ResultColumn.CORNER_RANK}_relative_avg",
    ]
    num_mean_races = 3
    corner_rank_columns = [f"corner_{i+1}_rank" for i in range(num_corners)]
    mean_columns = [
        *previous_mean_feature_columns,
        *corner_rank_columns,
    ]
    last_race_exprs = [
        pl.col(ResultColumn.HORSE_ID),
        pl.col(ResultColumn.RACE_ID),
        # horse_id_last{i + 1}_{c}
        _shift_horse_result_expr(raw_column=ResultColumn.RANK),
        _shift_horse_result_expr(raw_column=ResultColumn.DISTANCE),
        *(_shift_horse_result_expr(raw_column=c, n_shift=i) for c in mean_columns for i in range(num_mean_races)),
    ]
    race_previous_columns = [
        _format_last_column_name(raw_column=c, n_shift=i) for c in mean_columns for i in range(num_mean_races)
    ]

    max_num_races = 5
    if ResultColumn.HORSE_WEIGHT_DIFF_DEV in feature_columns:
        # weight diff
        last_race_exprs.extend(
            [
                _shift_horse_result_expr(raw_column=ResultColumn.HORSE_WEIGHT_DIFF, n_shift=i)
                for i in range(max_num_races)
            ]
        )
    last_race_df = df.select(last_race_exprs)

    # [mean features of last 3 races]
    # - horse_id_last3_goal_time_avg
    # - horse_id_last3_goal_speed_avg
    # - horse_id_last3_last_3f_time_avg
    # - horse_id_last3_corner_1_rank_avg
    # - horse_id_last3_corner_2_rank_avg
    # - horse_id_last3_corner_3_rank_avg
    # - horse_id_last3_corner_4_rank_avg
    # - horse_id_last3_corner_1_rank_clipped_avg
    # - horse_id_last3_corner_2_rank_clipped_avg
    # - horse_id_last3_corner_3_rank_clipped_avg
    # - horse_id_last3_corner_4_rank_clipped_avg
    last_prefix = f"{ResultColumn.HORSE_ID}_last"
    num_mean = 3
    last3_mean_column_dict = {c: f"{ResultColumn.HORSE_ID}_last{num_mean}_{c}_avg" for c in mean_columns}
    last3_mean_exprs = [
        pl.mean_horizontal([pl.col(f"{last_prefix}{i + 1}_{src_c}") for i in range(num_mean)]).alias(alias_c)
        for src_c, alias_c in last3_mean_column_dict.items()
    ]
    last_race_df = last_race_df.with_columns(last3_mean_exprs)
    last3_mean_columns = [c for _, c in last3_mean_column_dict.items()]

    # [mean features of last 5 races]
    # - horse_id_last5_horse_weight_diff_avg
    if ResultColumn.HORSE_WEIGHT_DIFF_DEV in feature_columns:
        weight_diff_avg_column = f"{ResultColumn.HORSE_ID}_last{max_num_races}_{ResultColumn.HORSE_WEIGHT_DIFF}_avg"
        last_race_df = last_race_df.with_columns(
            (
                sum([pl.col(f"{last_prefix}{i + 1}_{ResultColumn.HORSE_WEIGHT_DIFF}") for i in range(max_num_races)])
                / max_num_races
            ).alias(weight_diff_avg_column)
        )
        weight_diff_avg_previous_columns = [weight_diff_avg_column]
    else:
        weight_diff_avg_previous_columns = []

    # [previous.race_date]
    race_date_df = df.select(
        pl.col(ResultColumn.HORSE_ID).cast(pl.Utf8),
        pl.col(ResultColumn.RACE_ID).cast(pl.Utf8),
        # Get the date of the last race
        _shift_horse_result_expr(raw_column=ResultColumn.RACE_DATE),
    )
    race_date_previous_columns = [_format_last_column_name(raw_column=ResultColumn.RACE_DATE)]

    # Combine all stats
    logger.info("Combining win, show and previous stats...")
    horse_race_df = df.select(ResultColumn.HORSE_ID, ResultColumn.RACE_ID).unique()
    return {
        "data": (
            horse_race_df.join(last_race_df, on=[ResultColumn.HORSE_ID, ResultColumn.RACE_ID], how="left").join(
                race_date_df, on=[ResultColumn.HORSE_ID, ResultColumn.RACE_ID], how="left"
            )
        ),
        "previous_feature_columns": [
            *race_previous_columns,
            *last3_mean_columns,
            *weight_diff_avg_previous_columns,
            *race_date_previous_columns,
        ],
    }


def preprocess(
    raw_df: pl.DataFrame,
    feature_columns: list[str],
    horse_df: pl.DataFrame | None = None,
    jockey_df: pl.DataFrame | None = None,
    mode: Literal["train", "predict"] = "train",
) -> dict[str, pl.DataFrame]:
    """Preprocess data.

    1. Filter race
    2. Select base columns for preprocess
    3. Prepare label
    4. Calcurate goal speed
    5. Process corner rank
    6. Process categorical features
    7. Prepare horse features
    """

    # 1. Filter race
    if mode == "train":
        df = _remove_debut_race(raw_df)
    else:
        df = raw_df
    df = df.drop("race_name")

    # 2. Select base columns for preprocess
    df = select_base_columns(original_df=df, feature_columns=feature_columns, mode=mode)

    if mode == "train":
        # 3. Prepare label
        df = df.with_columns(
            get_win_label_expr(),
            get_show_label_expr(),
            get_rank_clipped_expr(),
            get_inverse_rank_expr(),  # target for ranker
            get_inverse_rank_log2_expr(),  # target for ranker
        )

    logger.info("After selecting basic features, shape: %s", df.shape)

    # 4. Process categorical features
    df = df.with_columns(
        [
            pl.col(c).cast(pl.Categorical).alias(f"{c}_cat")
            for c in (
                ResultColumn.HORSE_ID,
                ResultColumn.JOCKEY_ID,
                ResultColumn.TRAINER_ID,
                ResultColumn.RACE_CLASS,
                ResultColumn.RACE_PLACE,
            )
        ]
    )
    logger.info("After preprocessing category type, shape: %s", df.shape)

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

    # 5. Prepare horse features
    if horse_df is None:
        if mode == "train":
            # [win]
            # - horse_id_win_count
            # - horse_id_win_rate
            # - horse_id_win_count_distance_class_{distance_class}
            # - horse_id_win_rate_distance_class_{distance_class}
            # [show]
            # - horse_id_show_count
            # - horse_id_show_rate
            # - horse_id_show_count_distance_class_{distance_class}
            # - horse_id_show_rate_distance_class_{distance_class}
            # [previous]
            # - horse_id_last1_goal_time
            # - horse_id_last1_goal_speed
            # - horse_id_last1_last_3f_time
            logger.info("Preprocessing horse features...")
            horse_race_result = preprocess_horse(df, feature_columns=feature_columns)
            horse_race_df = horse_race_result["data"]
            previous_feature_columns = horse_race_result["previous_feature_columns"]
            df = df.join(horse_race_df, on=[ResultColumn.HORSE_ID, ResultColumn.RACE_ID], how="left")

            # latest for prediction
            horse_df = df.group_by(ResultColumn.HORSE_ID).last().drop(ResultColumn.RACE_ID)

            # shift previous features
            df = df.with_columns(
                pl.col(c).sort_by(ResultColumn.RACE_DATE).shift(1).over(ResultColumn.HORSE_ID).alias(c)
                for c in previous_feature_columns
            )
        else:
            # TODO
            # raise ValueError("mode is not train, but horse_df is not provided")
            pass
    else:
        df = df.join(horse_df, on=ResultColumn.HORSE_ID, how="left")

    # Days since last race feature
    df = df.with_columns(
        (
            (
                pl.col(ResultColumn.RACE_DATE).str.strptime(pl.Date, "%Y%m%d")
                - pl.col(f"{ResultColumn.HORSE_ID}_last1_{ResultColumn.RACE_DATE}").str.strptime(pl.Date, "%Y%m%d")
            )
            .dt.total_days()
            .alias(f"{ResultColumn.HORSE_ID}_days_since_last_race")
        )
    )

    logger.info("After joining horse features, shape: %s", df.shape)

    # weight dev
    logger.info("Preprocessing horse weight difference average...")
    if ResultColumn.HORSE_WEIGHT_DIFF_DEV in feature_columns:
        df = df.with_columns(
            (
                pl.col(ResultColumn.HORSE_WEIGHT_DIFF)
                - pl.col(f"{ResultColumn.HORSE_ID}_last5_{ResultColumn.HORSE_WEIGHT_DIFF}_avg")
            ).alias(ResultColumn.HORSE_WEIGHT_DIFF_DEV)
        )
        logger.info("After joining weight diff avg, shape: %s", df.shape)
    else:
        logger.info("Skip.")

    if horse_df is not None:
        horse_df = horse_df.unique(subset=[ResultColumn.HORSE_ID])
    if jockey_df is not None:
        jockey_df = jockey_df.unique(subset=[ResultColumn.JOCKEY_ID])

    return {
        "feature": df,
        "horse": horse_df,
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
    importance: dict[str, dict[str, float]],
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

    # upload features list
    feature_columns_path = tmp_dir / "feature_columns.json"
    with open(feature_columns_path, "w") as fp:
        json.dump(feature_columns, fp)

    # upload metric
    metric_path = tmp_dir / "metrics.json"
    with open(metric_path, "w") as fp:
        json.dump(metric, fp)
    bucket.blob(f"{prefix}/metrics.json").upload_from_filename(str(metric_path))

    # upload importance
    importance_path = tmp_dir / "importance.json"
    with open(importance_path, "w") as fp:
        json.dump(importance, fp)
    bucket.blob(f"{prefix}/importance.json").upload_from_filename(str(importance_path))
