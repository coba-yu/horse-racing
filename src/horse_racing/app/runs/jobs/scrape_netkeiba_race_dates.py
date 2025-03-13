from argparse import ArgumentParser
from dataclasses import dataclass

from tqdm import tqdm

from horse_racing.app.runs.jobs.scrape_netkeiba import scrape_by_race_date
from horse_racing.core.chrome import ChromeDriver
from horse_racing.core.gcp.storage import StorageClient
from horse_racing.core.logging import logger
from horse_racing.usecase.race_schedule import RaceScheduleUsecase


@dataclass
class Argument:
    year: int | None = None
    month: int | None = None


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--year", type=int)
    parser.add_argument("--month", type=int)

    args, _ = parser.parse_known_args(namespace=Argument())
    logger.info(f"{args=}")

    if args.year is None:
        raise ValueError("year is required.")
    if args.month is None:
        raise ValueError("month is required.")

    driver = ChromeDriver()
    storage_client = StorageClient()
    schedule_usecase = RaceScheduleUsecase(driver=driver)

    race_dates = schedule_usecase.get_race_dates(year=args.year, month=args.month)
    logger.info(f"race_dates: {race_dates}")

    for race_date in tqdm(race_dates, mininterval=60.0, maxinterval=180.0):
        scrape_by_race_date(race_date=race_date, driver=driver, storage_client=storage_client)


if __name__ == "__main__":
    main()
