from dataclasses import dataclass
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory

from tqdm import tqdm

from horse_racing.core.chrome import ChromeDriver
from horse_racing.core.gcp.storage import StorageClient
from horse_racing.core.logging import logger
from horse_racing.infrastructure.netkeiba.race_result import RaceResultNetkeibaRepository
from horse_racing.usecase.race_result import RaceResultUsecase
from horse_racing.usecase.race_schedule import RaceScheduleUsecase


@dataclass
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
    storage_client = StorageClient()

    with TemporaryDirectory() as tmp_dir:
        result_repository = RaceResultNetkeibaRepository(
            driver=driver,
            storage_client=storage_client,
            root_dir=Path(tmp_dir),
        )
        schedule_usecase = RaceScheduleUsecase(driver=driver)
        result_usecase = RaceResultUsecase(race_result_repository=result_repository)

        race_date = args.race_date
        race_ids = schedule_usecase.get_race_ids(race_date=race_date)
        logger.info(f"race_ids: {race_ids}")

        for race_id in tqdm(race_ids, mininterval=60.0, maxinterval=180.0):
            logger.info(f"race_id: {race_id}")
            result_usecase.get(race_date=race_date, race_id=race_id)  # TODO: not yet used returned html


if __name__ == "__main__":
    main()
