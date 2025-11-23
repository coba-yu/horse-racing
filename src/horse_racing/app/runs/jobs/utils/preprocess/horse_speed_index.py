import polars as pl
from collections.abc import Sequence

from horse_racing.usecase.race_result import ResultColumn

SPEED_INDEX_KEY_COLUMNS = (
    ResultColumn.RACE_PLACE,
    ResultColumn.FIELD_TYPE,
    ResultColumn.DISTANCE,
    ResultColumn.RACE_CLASS,
    ResultColumn.FIELD_CONDITION,
)
SPEED_INDEX_VALUE_COLUMNS = (
    ResultColumn.GOAL_TIME,
    ResultColumn.LAST_3F_TIME,
)


def calcurate_avg_for_base_index(
    df: pl.DataFrame,
    target_column: str,
    key_columns: Sequence[str] = SPEED_INDEX_KEY_COLUMNS,
) -> pl.DataFrame:
    return df.group_by(*key_columns).agg(pl.mean(target_column).alias(f"base_{target_column}_avg"))


def calcurate_std_for_base_index(
    df: pl.DataFrame,
    target_column: str,
    key_columns: Sequence[str] = SPEED_INDEX_KEY_COLUMNS,
) -> pl.DataFrame:
    return df.group_by(*key_columns).agg(pl.std(target_column).alias(f"base_{target_column}_std"))


def calcurate_base_index(
    df: pl.DataFrame,
    key_columns: Sequence[str] = SPEED_INDEX_KEY_COLUMNS,
    target_columns: Sequence[str] = SPEED_INDEX_VALUE_COLUMNS,
) -> pl.DataFrame:
    for target_column in target_columns:
        avg_df = calcurate_avg_for_base_index(df, target_column=target_column, key_columns=key_columns)
        std_df = calcurate_std_for_base_index(df, target_column=target_column, key_columns=key_columns)
        if target_column == target_columns[0]:
            base_index_df = avg_df.join(std_df, on=list(key_columns), how="left")
        else:
            base_index_df = base_index_df.join(avg_df, on=list(key_columns), how="left").join(
                std_df, on=list(key_columns), how="left"
            )
    return base_index_df


def calculate_speed_index(
    df: pl.DataFrame,
    base_df: pl.DataFrame,
    key_columns: Sequence[str] = SPEED_INDEX_KEY_COLUMNS,
    target_columns: Sequence[str] = SPEED_INDEX_VALUE_COLUMNS,
) -> pl.DataFrame:
    """Calculate speed index (Z-score) for race times.

    Speed Index = (time - mean) / std

    This normalizes times across different track conditions,
    allowing fair comparison between races on different days.
    """
    df = df.join(base_df, on=list(key_columns), how="left")

    for target_column in target_columns:
        value_col = f"{ResultColumn.HORSE_ID}_last1_{target_column}"
        avg_col = f"base_{target_column}_avg"
        std_col = f"base_{target_column}_std"
        index_col = f"{ResultColumn.HORSE_ID}_speed_index_{target_column}"

        df = df.with_columns(
            pl.when(pl.col(std_col) > 0)
            .then((pl.col(value_col) - pl.col(avg_col)) / pl.col(std_col))
            .otherwise(None)
            .alias(index_col)
        )

    return df
