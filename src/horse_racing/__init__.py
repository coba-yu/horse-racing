import os
import polars as pl
from argparse import ArgumentParser
from tqdm import tqdm

from horse_racing.core.chrome import ChromeDriver
from horse_racing.core.logging import logger
from horse_racing.usecase.race_schedule import RaceScheduleUsecase


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=int, default=1)

    args = parser.parse_args()
    year = args.year
    month = args.month

    driver = ChromeDriver()
    race_schedule_usecase = RaceScheduleUsecase(driver=driver)

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
