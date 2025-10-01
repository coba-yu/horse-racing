import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Any, Optional

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


def _convert_distance_to_class(distance: int) -> str:
    if distance < 1400:
        return "sprint"
    if distance <= 1800:
        return "mile"
    if distance <= 2200:
        return "middle"
    if distance <= 2800:
        return "middle_to_long"
    return "long"


def _label_encode(df: pl.DataFrame, column: str, label_dict: dict[str, int]) -> pl.DataFrame:
    return df.with_columns(
        pl.col(column)
        .cast(pl.Utf8)
        .str.extract(rf'({"|".join(list(label_dict))})')
        .replace_strict(label_dict, default=-1)
        .cast(pl.Int8)
    )


def _calculate_grouped_stats(
    base_df: pl.DataFrame,
    group_column: str,
    group_value: int | str,
    rank_condition: pl.Expr,
    prefix: str,
    id_column: str,
) -> pl.DataFrame:
    count_df = (
        base_df.filter(pl.col(group_column) == group_value)
        .group_by(id_column)
        .agg(
            pl.col(ResultColumn.RANK)
            .filter(rank_condition)
            .count()
            .alias(f"{prefix}_count_{group_column}_{group_value}")
        )
    )

    rate_df = (
        base_df.filter(pl.col(group_column) == group_value)
        .group_by(id_column)
        .agg(
            (pl.col(ResultColumn.RANK).filter(rank_condition).count() / pl.col(ResultColumn.RANK).count()).alias(
                f"{prefix}_rate_{group_column}_{group_value}"
            )
        )
    )

    return count_df.join(rate_df, on=id_column, how="left")


def _calculate_overall_stats(
    base_df: pl.DataFrame,
    race_count_df: pl.DataFrame,
    rank_condition: pl.Expr,
    ticket_type: str,
    id_column: str,
) -> pl.DataFrame:
    """Calculate overall win/show statistics"""
    count_df = base_df.group_by(id_column).agg(
        pl.col(ResultColumn.RANK).filter(rank_condition).count().alias(f"{id_column}_{ticket_type}_count")
    )

    rate_df = race_count_df.join(count_df, on=id_column, how="left").select(
        pl.col(id_column),
        (pl.col(f"{id_column}_{ticket_type}_count") / pl.col("race_count")).alias(f"{id_column}_{ticket_type}_rate"),
    )
    rate_df = rate_df.unique(subset=[id_column])

    return count_df.join(rate_df, on=id_column, how="left")


def _calculate_field_type_stats(
    base_df: pl.DataFrame,
    rank_condition: pl.Expr,
    ticket_type: str,
    id_column: str,
    result_df: pl.DataFrame,
) -> pl.DataFrame:
    """Calculate field type specific statistics"""
    for field_type in FIELD_TYPES:
        prefix = f"{id_column}_{ticket_type}"
        actual_field_types = base_df[ResultColumn.FIELD_TYPE].unique().to_list()

        if field_type in actual_field_types:
            field_type_stat_df = _calculate_grouped_stats(
                base_df,
                group_column=ResultColumn.FIELD_TYPE,
                group_value=field_type,
                rank_condition=rank_condition,
                prefix=prefix,
                id_column=id_column,
            )
            result_df = result_df.join(field_type_stat_df, on=id_column, how="left")
        else:
            logger.info("%s not exists in %s and fill with null", field_type, actual_field_types)
            result_df[f"{prefix}_{field_type}_count"] = None

    return result_df


def _calculate_distance_stats(
    base_df: pl.DataFrame,
    rank_condition: pl.Expr,
    ticket_type: str,
    id_column: str,
    result_df: pl.DataFrame,
) -> pl.DataFrame:
    """Calculate distance specific statistics"""
    for distance_class in DISTANCE_CLASSES:
        prefix = f"{id_column}_{ticket_type}"
        actual_distance_classes = base_df[ResultColumn.DISTANCE_CLASS].unique().to_list()

        if distance_class in actual_distance_classes:
            distance_stat_df = _calculate_grouped_stats(
                base_df,
                group_column=ResultColumn.DISTANCE_CLASS,
                group_value=distance_class,
                rank_condition=rank_condition,
                prefix=prefix,
                id_column=id_column,
            )
            result_df = result_df.join(distance_stat_df, on=id_column, how="left")
        else:
            logger.info("%s not exists in %s and fill with null", distance_class, actual_distance_classes)
            result_df[f"{prefix}_{distance_class}_count"] = None

    return result_df


def _calculate_race_place_stats(
    base_df: pl.DataFrame,
    rank_condition: pl.Expr,
    ticket_type: str,
    id_column: str,
    result_df: pl.DataFrame,
) -> pl.DataFrame:
    """Calculate race place specific statistics"""
    for race_place in RACE_PLACES:
        prefix = f"{id_column}_{ticket_type}"
        actual_race_places = base_df[ResultColumn.RACE_PLACE].unique().to_list()

        if race_place in actual_race_places:
            race_place_stat_df = _calculate_grouped_stats(
                base_df,
                group_column=ResultColumn.RACE_PLACE,
                group_value=race_place,
                rank_condition=rank_condition,
                prefix=prefix,
                id_column=id_column,
            )
            result_df = result_df.join(race_place_stat_df, on=id_column, how="left")
        else:
            logger.info("%s not exists in %s and fill with null", race_place, actual_race_places)
            result_df[f"{prefix}_{race_place}_count"] = None

    return result_df


def _format_last_column_name(raw_column: str, prefix: str = "horse_id_", n_shift: int = 0) -> str:
    return f"{prefix}last{n_shift + 1}_{raw_column}"


def _shift_horse_result_expr(raw_column: str, prefix: str = "horse_id_", n_shift: int = 0) -> pl.Expr:
    alias_name = _format_last_column_name(raw_column=raw_column, prefix=prefix, n_shift=n_shift)
    return (
        pl.col(raw_column).sort_by(ResultColumn.RACE_DATE).shift(n_shift).over(ResultColumn.HORSE_ID).alias(alias_name)
    )


def _agg_horse_last_result(
    base_df: pl.DataFrame,
    race_categories: Sequence[int | str],
    race_category_column: str,
    last_prefix: str,
) -> dict[str, Any]:
    key_columns = (
        ResultColumn.HORSE_ID,
        ResultColumn.RACE_ID,
    )
    previous_feature_raw_columns = (
        ResultColumn.WIN_LABEL,
        ResultColumn.SHOW_LABEL,
        ResultColumn.RANK_CLIPPED,
    )
    rate_rename_dict = {
        ResultColumn.WIN_LABEL: "win_rate",
        ResultColumn.SHOW_LABEL: "show_rate",
    }

    all_df: Optional[pl.DataFrame] = None
    for category in race_categories:
        logger.info("horse last result by %s: %s", race_category_column, category)
        target_df = base_df.filter(pl.col(race_category_column) == category)
        target_df = target_df.select(
            *key_columns,
            ResultColumn.RACE_DATE,
            *previous_feature_raw_columns,
        )

        # 過去Nレースの平均を取るための材料
        max_past = 5
        target_df = target_df.with_columns(
            _shift_horse_result_expr(raw_column=c, n_shift=n_shift)
            for c in previous_feature_raw_columns
            for n_shift in range(max_past)
        )

        # mean
        target_df = target_df.with_columns(
            *(
                pl.mean_horizontal([pl.col(f"{last_prefix}{i + 1}_{c}") for i in range(num_mean)]).alias(
                    f"{ResultColumn.HORSE_ID}_last{num_mean}_{c}_avg"
                )
                for c in previous_feature_raw_columns
                for num_mean in (3, 5)
            )
        )

        previous_feature_columns = []
        previous_feature_exprs = []
        for c in previous_feature_raw_columns:
            # last 1 race
            last1_original = f"{ResultColumn.HORSE_ID}_last1_{c}"
            last1_alias = f"{last1_original}_{race_category_column}_{category}"
            previous_feature_columns.append(last1_alias)
            previous_feature_exprs.append(pl.col(last1_original).alias(last1_alias))

            # last 3, 5 races
            for num_mean in (3, 5):
                last_n_prefix = f"{ResultColumn.HORSE_ID}_last{num_mean}_"
                last_n_original = f"{last_n_prefix}{c}_avg"
                last_n_alias = f"{last_n_prefix}{rate_rename_dict.get(c, c)}_avg_{race_category_column}_{category}"
                previous_feature_columns.append(last_n_alias)
                previous_feature_exprs.append(pl.col(last_n_original).alias(last_n_alias))

        target_df = target_df.select(*key_columns, *previous_feature_exprs)
        if all_df is None:
            all_df = target_df
        else:
            suffix = "_right"
            all_df = all_df.join(target_df, on=key_columns, how="outer", suffix=suffix)
            all_df = all_df.drop([f"{c}{suffix}" for c in key_columns], strict=False)
    return {"data": all_df, "previous_feature_columns": previous_feature_columns}


def _agg_horse_last_result_by_distance_class(base_df: pl.DataFrame, last_prefix: str) -> dict[str, Any]:
    return _agg_horse_last_result(
        base_df=base_df,
        race_categories=DISTANCE_CLASSES,
        race_category_column=ResultColumn.DISTANCE_CLASS,
        last_prefix=last_prefix,
    )


def _agg_horse_last_result_by_field_type(base_df: pl.DataFrame, last_prefix: str) -> dict[str, Any]:
    return _agg_horse_last_result(
        base_df=base_df,
        race_categories=FIELD_TYPES,
        race_category_column=ResultColumn.FIELD_TYPE,
        last_prefix=last_prefix,
    )


def _extract_horse_last_result_same_field_type(
    base_df: pl.DataFrame,
    column_prefix: str,
) -> pl.DataFrame:
    alias_name = f"{column_prefix}field_type_same"
    result_df = base_df.with_columns(
        pl.when(pl.col(ResultColumn.FIELD_TYPE) == 0)
        .then(pl.col(f"{column_prefix}field_type_0"))
        .otherwise(None)
        .alias(alias_name)
    )
    for field_type in FIELD_TYPES[1:]:
        result_df = result_df.with_columns(
            pl.when(pl.col(ResultColumn.FIELD_TYPE) == field_type)
            .then(pl.col(f"{column_prefix}field_type_{field_type}"))
            .otherwise(pl.col(alias_name))
            .alias(alias_name)
        )
    return result_df


def preprocess_horse(df: pl.DataFrame, feature_columns: list[str]) -> pl.DataFrame:
    # [previous]
    # - horse_id_last1_rank
    # - horse_id_last1_distance
    # - horse_id_last1_corner_{i}_rank
    # - horse_id_last1_goal_time
    # - horse_id_last1_goal_speed
    # - horse_id_last1_last_3f_time
    logger.info("Calculating previous result...")
    previous_mean_feature_columns = [
        ResultColumn.GOAL_TIME,
        ResultColumn.GOAL_SPEED,
        ResultColumn.LAST_3F_TIME,
    ]
    num_mean_races = 3
    num_corners = 4
    corner_rank_columns = [f"corner_{i+1}_rank" for i in range(num_corners)]
    corner_rank_clipped_columns = [f"corner_{i+1}_rank_clipped" for i in range(num_corners)]
    mean_columns = [
        *previous_mean_feature_columns,
        *corner_rank_columns,
        *corner_rank_clipped_columns,
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

    # [previous.distance_class]
    distance_class_result = _agg_horse_last_result_by_distance_class(base_df=df, last_prefix=last_prefix)
    distance_class_df = distance_class_result["data"]
    distance_class_previous_columns = distance_class_result["previous_feature_columns"]

    # [previous.field_type]
    field_type_result = _agg_horse_last_result_by_field_type(base_df=df, last_prefix=last_prefix)
    field_type_df = field_type_result["data"]
    field_type_previous_columns = field_type_result["previous_feature_columns"]

    # Combine all stats
    logger.info("Combining win, show and previous stats...")
    horse_race_df = df.select(ResultColumn.HORSE_ID, ResultColumn.RACE_ID).unique()
    return {
        "data": (
            horse_race_df.join(last_race_df, on=[ResultColumn.HORSE_ID, ResultColumn.RACE_ID], how="left")
            .join(race_date_df, on=[ResultColumn.HORSE_ID, ResultColumn.RACE_ID], how="left")
            .join(distance_class_df, on=[ResultColumn.HORSE_ID, ResultColumn.RACE_ID], how="left")
            .join(field_type_df, on=[ResultColumn.HORSE_ID, ResultColumn.RACE_ID], how="left")
        ),
        "previous_feature_columns": [
            *race_previous_columns,
            *last3_mean_columns,
            *weight_diff_avg_previous_columns,
            *race_date_previous_columns,
            *distance_class_previous_columns,
            *field_type_previous_columns,
        ],
    }


def _calculate_jockey_stats(
    base_df: pl.DataFrame,
    race_count_df: pl.DataFrame,
    rank_condition: pl.Expr,
    ticket_type: str,
) -> pl.DataFrame:
    id_column = ResultColumn.JOCKEY_ID

    # Calculate overall stats
    result_df = _calculate_overall_stats(
        base_df=base_df,
        race_count_df=race_count_df,
        rank_condition=rank_condition,
        ticket_type=ticket_type,
        id_column=id_column,
    )

    # Calculate field type stats
    result_df = _calculate_field_type_stats(
        base_df=base_df,
        rank_condition=rank_condition,
        ticket_type=ticket_type,
        id_column=id_column,
        result_df=result_df,
    )

    # Calculate distance stats
    result_df = _calculate_distance_stats(
        base_df=base_df,
        rank_condition=rank_condition,
        ticket_type=ticket_type,
        id_column=id_column,
        result_df=result_df,
    )

    # Calculate race place stats
    result_df = _calculate_race_place_stats(
        base_df=base_df,
        rank_condition=rank_condition,
        ticket_type=ticket_type,
        id_column=id_column,
        result_df=result_df,
    )

    return result_df


def _agg_jockey(df: pl.DataFrame) -> pl.DataFrame:
    # Base dataframe with minimal columns
    jockey_base_df = df.select(
        [
            pl.col(ResultColumn.RANK).cast(pl.Int32),
            pl.col(ResultColumn.JOCKEY_ID).cast(pl.String),
            pl.col(ResultColumn.FIELD_TYPE),
            pl.col(ResultColumn.DISTANCE_CLASS),
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
    win_stats = _calculate_jockey_stats(
        base_df=jockey_base_df,
        race_count_df=jockey_race_count_df,
        rank_condition=win_rank_condition,
        ticket_type="win",
    )
    logger.info("win count and rate features calculated:\n%s", win_stats)

    # Calculate show stats
    logger.info("Calculating jockey show count and rate...")
    show_stats = _calculate_jockey_stats(
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
    horse_df: pl.DataFrame | None = None,
    jockey_df: pl.DataFrame | None = None,
    mode: Literal["train", "predict"] = "train",
) -> dict[str, pl.DataFrame]:
    # filter race
    df = _remove_debut_race(raw_df)
    df = df.drop("race_name")

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
    if ResultColumn.HORSE_WEIGHT_DIFF_DEV in feature_columns:
        select_exprs.append(
            pl.col(HORSE_WEIGHT_AND_DIFF_COLUMN)
            .str.extract(r"\(([-\+\d]+)\)")
            .cast(pl.Int32)
            .alias(ResultColumn.HORSE_WEIGHT_DIFF)
        )

    if mode == "train":
        # target label
        df = df.filter(~pl.col(ResultColumn.RANK).is_in({"中止", "除外", "取消"}))
        select_exprs.append(pl.col(ResultColumn.RANK).cast(pl.Int32))
        select_exprs.append(pl.col(ResultColumn.GOAL_TIME).cast(pl.String).alias(ResultColumn.GOAL_TIME))
        select_exprs.append(pl.col(ResultColumn.LAST_3F_TIME).cast(pl.Float64).alias(ResultColumn.LAST_3F_TIME))
        select_exprs.append(pl.col(ResultColumn.CORNER_RANK).cast(pl.String).alias(f"raw_{ResultColumn.CORNER_RANK}"))

        df = df.select(select_exprs)
        df = df.with_columns(
            get_win_label_expr(),
            get_show_label_expr(),
            get_rank_clipped_expr(),
        )

        # [goal speed]
        df = calcurate_goal_speed(df)

        # [corner rank]
        # 6-4-3-1
        # => corner_rank_1 = 6, corner_rank_2 = 4, ... , corner_rank_4 = 1
        num_corners = 4
        df = df.with_columns(pl.col(f"raw_{ResultColumn.CORNER_RANK}").str.split("-").alias("corner_ranks"))
        df = df.with_columns(
            [
                pl.col("corner_ranks").list.get(i, null_on_oob=True).cast(pl.Int32).alias(f"corner_{i+1}_rank")
                for i in range(num_corners)
            ]
        )
        # Clip large corner rank
        df = df.with_columns([get_rank_clipped_expr(f"corner_{i+1}_rank") for i in range(num_corners)])
        df = df.drop("corner_ranks", pl.col(f"raw_{ResultColumn.CORNER_RANK}"))
    else:
        df = df.select(select_exprs)
    logger.info("After selecting basic features, shape: %s", df.shape)

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
    logger.info("After preprocessing category type, shape: %s", df.shape)

    # jockey target encoding
    df = df.with_columns(
        pl.col(ResultColumn.DISTANCE).map_elements(_convert_distance_to_class).alias(ResultColumn.DISTANCE_CLASS)
    )
    if jockey_df is None:
        if mode == "train":
            logger.info("Preprocessing jockey features...")
            jockey_df = _agg_jockey(df)
        else:
            raise ValueError("mode is not train, but jockey_df is not provided")
    df = df.join(jockey_df, on=ResultColumn.JOCKEY_ID, how="left")
    logger.info("After joining jockey features, shape: %s", df.shape)

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

    for column_prefix in (
        f"{ResultColumn.HORSE_ID}_last1_win_label_",
        f"{ResultColumn.HORSE_ID}_last1_show_label_",
        f"{ResultColumn.HORSE_ID}_last1_rank_clipped_",
        f"{ResultColumn.HORSE_ID}_last3_win_rate_avg_",
        f"{ResultColumn.HORSE_ID}_last3_show_rate_avg_",
        f"{ResultColumn.HORSE_ID}_last3_rank_clipped_avg_",
        f"{ResultColumn.HORSE_ID}_last5_win_rate_avg_",
        f"{ResultColumn.HORSE_ID}_last5_show_rate_avg_",
        f"{ResultColumn.HORSE_ID}_last5_rank_clipped_avg_",
    ):
        df = _extract_horse_last_result_same_field_type(base_df=df, column_prefix=column_prefix)

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
