import os
from argparse import ArgumentParser

import polars as pl
from tqdm import tqdm

from horse_racing.core.chrome import ChromeDriver
from horse_racing.core.logging import logger
from horse_racing.usecase.race_schedule import RaceScheduleUsecase
from horse_racing.usecase.horse import HorseUsecase


def get_and_save_race_results(
    driver: ChromeDriver,
    year: int,
    month: int,
    race_dates: list[str] | None,
) -> pl.DataFrame:
    race_schedule_usecase = RaceScheduleUsecase(driver=driver)

    if race_dates is None:
        race_dates = race_schedule_usecase.get_race_dates(year=year, month=month)
    logger.info(race_dates)

    race_date_to_ids_dict = {}
    for race_date in race_dates:
        race_ids_per_date = race_schedule_usecase.get_race_ids(race_date=race_date)
        logger.info(f"{race_date=}, {race_ids_per_date=}")
        race_date_to_ids_dict[race_date] = race_ids_per_date

    all_df: pl.DataFrame | None = None
    for race_date, race_ids in race_date_to_ids_dict.items():
        for race_id in tqdm(race_ids, desc=f"{race_date=}"):
            df = race_schedule_usecase.get_race_result(race_id=race_id, race_date=race_date)
            if all_df is None:
                all_df = df
            else:
                all_df = pl.concat([all_df, df])

    if all_df is None:
        raise ValueError(f"No data ({year=}, {month=}).")
    logger.info(all_df)

    monthly_results_dir = os.path.join("data", "cache", "parquet", "race_results")
    os.makedirs(monthly_results_dir, exist_ok=True)
    all_df.write_parquet(os.path.join(monthly_results_dir, f"{year:04}{month:02}.parquet"))

    return all_df


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=int, default=1)

    parser.add_argument("--race-dates", type=str)

    args = parser.parse_args()
    year = args.year
    month = args.month
    if args.race_dates is None:
        race_dates = None
    else:
        race_dates = args.race_dates.split(",")

    driver = ChromeDriver()

    race_df = get_and_save_race_results(driver=driver, year=year, month=month, race_dates=race_dates)
    horse_id_per_date_df = race_df.group_by("race_date").agg(pl.col("horse_id").alias("horse_ids"))
    date_to_horse_ids_dict = dict(
        zip(horse_id_per_date_df["race_date"].to_list(), horse_id_per_date_df["horse_ids"].to_list())
    )
    print(date_to_horse_ids_dict)

    horse_usecase = HorseUsecase()

    for race_date, horse_ids in date_to_horse_ids_dict.items():
        for horse_id in tqdm(horse_ids, desc=f"{race_date=}"):
            horse_df = horse_usecase.get_horse_results(horse_id=horse_id, race_date=race_date)
            print(horse_df)
