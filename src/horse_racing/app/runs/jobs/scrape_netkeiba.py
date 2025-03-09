from dataclasses import dataclass
from argparse import ArgumentParser

from horse_racing.core.chrome import ChromeDriver
from horse_racing.core.logging import logger
from horse_racing.usecase.race_schedule import RaceScheduleUsecase


@dataclass(frozen=True)
class Argument:
    race_date: str = ""

    def validate(self) -> None:
        if len(self.race_date) == 0:
            raise ValueError("race_date is required.")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--race-date", type=str)

    args, _ = parser.parse_known_args(namespace=Argument())
    logger.info(f"{args=}")
    args.validate()

    driver = ChromeDriver()
    race_schedule_usecase = RaceScheduleUsecase(driver=driver)
    race_ids = race_schedule_usecase.get_race_ids(race_date=args.race_date)
    logger.info(f"race_ids: {race_ids}")


if __name__ == "__main__":
    main()
