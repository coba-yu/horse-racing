from typing import Literal
import polars as pl

from horse_racing.app.runs.jobs.utils.enum import Mode
from horse_racing.usecase.race_result import (
    ResultColumn,
    GENDER_AGE_COLUMN,
    HORSE_WEIGHT_AND_DIFF_COLUMN,
)

NUM_CORNERS = 4


def calcurate_goal_speed(df: pl.DataFrame) -> pl.DataFrame:
    # 1:54.5 => 60 + 54.5 = 114.5 sec
    df = df.with_columns(
        pl.col(ResultColumn.GOAL_TIME).str.extract(r"(\d+):").cast(pl.Int32).alias(f"{ResultColumn.GOAL_TIME}_minute"),
        pl.col(ResultColumn.GOAL_TIME)
        .str.extract(r"\d+:(\d+\.?\d*)")
        .cast(pl.Float64)
        .alias(f"{ResultColumn.GOAL_TIME}_second"),
    )

    df = df.with_columns(
        (pl.col(f"{ResultColumn.GOAL_TIME}_minute") * 60.0 + pl.col(f"{ResultColumn.GOAL_TIME}_second")).alias(
            ResultColumn.GOAL_TIME
        )
    )
    df = df.with_columns(
        (pl.col(ResultColumn.DISTANCE) / pl.col(ResultColumn.GOAL_TIME)).alias(ResultColumn.GOAL_SPEED)
    )
    return df.drop(f"{ResultColumn.GOAL_TIME}_minute", f"{ResultColumn.GOAL_TIME}_second")


def select_base_columns(
    original_df: pl.DataFrame,
    feature_columns: list[str],
    mode: Literal["train", "predict"],
    num_corners: int = NUM_CORNERS,
) -> pl.DataFrame:
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

    # horse_weight_diff はレース直前までわからないため
    # feature_columns で指定していない場合は skip する.
    if ResultColumn.HORSE_WEIGHT_DIFF_DEV in feature_columns:
        select_exprs.append(
            pl.col(HORSE_WEIGHT_AND_DIFF_COLUMN)
            .str.extract(r"\(([-\+\d]+)\)")
            .cast(pl.Int32)
            .alias(ResultColumn.HORSE_WEIGHT_DIFF)
        )

    if mode != Mode.TRAIN:
        return original_df.select(select_exprs)

    # Target label
    df = original_df.filter(~pl.col(ResultColumn.RANK).is_in({"中止", "除外", "取消"}))
    select_exprs.append(pl.col(ResultColumn.RANK).cast(pl.Int32).alias(ResultColumn.RANK))

    select_exprs.append(pl.col(ResultColumn.GOAL_TIME).cast(pl.String).alias(ResultColumn.GOAL_TIME))

    # Last 3F time
    select_exprs.append(pl.col(ResultColumn.LAST_3F_TIME).cast(pl.Float64).alias(ResultColumn.LAST_3F_TIME))

    # Corner rank
    select_exprs.append(pl.col(ResultColumn.CORNER_RANK).cast(pl.String).alias(f"raw_{ResultColumn.CORNER_RANK}"))
    df = df.select(select_exprs)

    # Calcurate goal time [sec]
    df = calcurate_goal_speed(df)

    # Parse corner rank
    # 6-4-3-1
    # => corner_rank_1 = 6, corner_rank_2 = 4, ... , corner_rank_4 = 1
    df = df.with_columns(pl.col(f"raw_{ResultColumn.CORNER_RANK}").str.split("-").alias("corner_ranks"))
    df = df.with_columns(
        [
            pl.col("corner_ranks").list.get(i, null_on_oob=True).cast(pl.Int32).alias(f"corner_{i+1}_rank")
            for i in range(num_corners)
        ]
    )

    return df
