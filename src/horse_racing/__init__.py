import polars as pl
from tqdm import tqdm

from horse_racing.core.chrome import ChromeDriver
from horse_racing.usecase.race_schedule import RaceScheduleUsecase


def main() -> None:
    driver = ChromeDriver()
    race_schedule_usecase = RaceScheduleUsecase(driver=driver)

    year = 2024
    month = 1
    race_dates = race_schedule_usecase.get_race_dates(year=year, month=month)
    print(race_dates)

    race_date_to_ids_dict = {}
    for race_date in race_dates:
        race_ids_per_date = race_schedule_usecase.get_race_ids(race_date=race_date)
        print(f"{race_date=}, {race_ids_per_date=}")
        race_date_to_ids_dict[race_date] = race_ids_per_date

    all_df: pl.DataFrame | None = None
    for race_date, race_ids in race_date_to_ids_dict.items():
        for race_id in tqdm(race_ids, desc=f"{race_date=}"):
            df = race_schedule_usecase.get_race_result(race_id=race_id, race_date=race_date)
            if all_df is None:
                all_df = df
            else:
                all_df = pl.concat([df, all_df])
    print(all_df)
