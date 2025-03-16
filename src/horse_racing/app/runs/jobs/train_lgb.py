from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from horse_racing.core.gcp.storage import StorageClient
from horse_racing.core.logging import logger
from horse_racing.infrastructure.netkeiba.race_result import RaceResultNetkeibaRepository
from horse_racing.usecase.race_result import RaceResultUsecase, Column, GENDER_AGE_COLUMN, HORSE_WEIGHT_AND_DIFF_COLUMN


@dataclass
class TrainConfig:
    train_first_date: str = ""
    train_last_date: str = ""
    valid_last_date: str = ""
    data_version: str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # constants
    model: str = "lightgbm"
    model_version: str = datetime.now().strftime("%Y%m%d_%H%M%S")


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


def preprocess(raw_df: pl.DataFrame) -> pl.DataFrame:
    # filter race
    df = raw_df.filter(~pl.col(Column.RANK).is_in({"中止", "除外", "取消"}))
    df = _remove_debut_race(df)
    df = df.drop("race_name")

    df = df.select(
        pl.col(Column.RANK).cast(pl.Int32),
        pl.col(Column.HORSE_NUMBER).cast(pl.Int32),
        pl.col(GENDER_AGE_COLUMN).alias(Column.GENDER),
        pl.col(GENDER_AGE_COLUMN).str.extract(r"(\d+)").cast(pl.Int32).alias(Column.AGE),
        pl.col(Column.TOTAL_WEIGHT).cast(pl.Float64).alias(Column.TOTAL_WEIGHT),
        # race
        pl.col(Column.RACE_NUMBER).str.extract(r"(\d+)").cast(pl.Int32).alias(Column.RACE_NUMBER),
        pl.col(Column.START_AT).str.extract(r"^(\d+)").cast(pl.Int32).alias(Column.START_AT),
        pl.col(Column.DISTANCE).str.extract(r"(\d+)").cast(pl.Int32).alias(Column.DISTANCE),
        pl.col(Column.DISTANCE).alias(Column.ROTATE),
        pl.col(Column.DISTANCE).alias(Column.FIELD_TYPE),
        pl.col(Column.WEATHER),
        pl.col(Column.FIELD_CONDITION),
        # fresh
        pl.col(Column.POPULAR).cast(pl.Int32).alias(Column.POPULAR),
        pl.col(Column.ODDS).cast(pl.Float32).alias(Column.ODDS),
        (
            pl.col(HORSE_WEIGHT_AND_DIFF_COLUMN)
            .str.extract(r"\(([-\+\d]+)\)")
            .cast(pl.Int32)
            .alias(Column.HORSE_WEIGHT_DIFF)
        ),
        # not feature
        pl.col(Column.RACE_ID),
        pl.col(Column.RACE_DATE),
        pl.col(Column.HORSE_NAME),
        pl.col(Column.HORSE_ID),
        pl.col(Column.JOCKEY_ID),
        pl.col(Column.TRAINER_ID),
    )

    # label encoding
    gender_label_dict = {"牝": 0, "牡": 1, "セ": 2}
    df = _label_encode(df=df, column=Column.GENDER, label_dict=gender_label_dict)
    df = df.drop(Column.GENDER)

    df = _label_encode(df=df, column=Column.ROTATE, label_dict={"左": 0, "右": 1})
    df = _label_encode(df=df, column=Column.FIELD_TYPE, label_dict={"芝": 0, "ダ": 1, "障": 2})
    df = _label_encode(
        df=df,
        column=Column.WEATHER,
        label_dict={"晴": 0, "曇": 1, "小雨": 2, "雨": 3, "小雪": 4, "雪": 5},
    )
    df = _label_encode(
        df=df,
        column=Column.FIELD_CONDITION,
        label_dict={"良": 0, "稍": 1, "重": 2, "不": 3, "未": 4},
    )

    # weight dev
    weight_diff_avg_df = df.group_by("horse_id").agg(pl.mean(Column.HORSE_WEIGHT_DIFF).alias("weight_diff_avg"))
    df = df.join(weight_diff_avg_df, on="horse_id", how="left")
    df = df.with_columns(
        (pl.col(Column.HORSE_WEIGHT_DIFF) - pl.col("weight_diff_avg")).alias(Column.HORSE_WEIGHT_DIFF_DEV)
    )
    return df


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
        logger.info(raw_df)

    # preprocess
    processed_df = preprocess(raw_df=raw_df)
    logger.info(processed_df)

    # split train and valid

    # hyper param tuning

    # train

    # save model


if __name__ == "__main__":
    main()
