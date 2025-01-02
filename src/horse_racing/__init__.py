from horse_racing.core.chrome import ChromeDriver
from horse_racing.usecase.race_schedule import RaceScheduleUsecase


def main() -> None:
    driver = ChromeDriver()
    race_schedule_usecase = RaceScheduleUsecase(driver=driver)

    year = 2024
    month = 1
    race_dates = race_schedule_usecase.get_race_dates(year=year, month=month)
    print(race_dates)
    for race_date in race_dates:
        race_ids = race_schedule_usecase.get_race_ids(race_date=race_date)
        print(f"{race_date=}, {race_ids=}")
