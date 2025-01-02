from horse_racing.usecase.race_schedule import RaseScheduleUsecase


def main() -> None:
    year = 2024
    month = 1
    rase_schedule_usecase = RaseScheduleUsecase()
    race_dates = rase_schedule_usecase.get_race_dates(year=year, month=month)
    print(race_dates)
