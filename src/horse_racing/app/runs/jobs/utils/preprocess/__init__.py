from typing import Literal
import polars as pl

from horse_racing.app.runs.jobs.utils.enum import Mode
from horse_racing.usecase.race_result import (
    ResultColumn,
    GENDER_AGE_COLUMN,
    HORSE_WEIGHT_AND_DIFF_COLUMN,
)


def select_base_columns(
    original_df: pl.DataFrame,
    feature_columns: list[str],
    mode: Literal["train", "predict"],
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

    # horse_weight_diff は
    if ResultColumn.HORSE_WEIGHT_DIFF_DEV in feature_columns:
        select_exprs.append(
            pl.col(HORSE_WEIGHT_AND_DIFF_COLUMN)
            .str.extract(r"\(([-\+\d]+)\)")
            .cast(pl.Int32)
            .alias(ResultColumn.HORSE_WEIGHT_DIFF)
        )

    if mode != Mode.TRAIN:
        return original_df.select(select_exprs)

    # target label
    df = original_df.filter(~pl.col(ResultColumn.RANK).is_in({"中止", "除外", "取消"}))
    select_exprs.append(pl.col(ResultColumn.RANK).cast(pl.Int32).alias(ResultColumn.RANK))

    select_exprs.append(pl.col(ResultColumn.GOAL_TIME).cast(pl.String).alias(ResultColumn.GOAL_TIME))
    select_exprs.append(pl.col(ResultColumn.LAST_3F_TIME).cast(pl.Float64).alias(ResultColumn.LAST_3F_TIME))
    select_exprs.append(pl.col(ResultColumn.CORNER_RANK).cast(pl.String).alias(f"raw_{ResultColumn.CORNER_RANK}"))

    return df.select(select_exprs)
