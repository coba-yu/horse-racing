from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from horse_racing.core.gcp.storage import StorageClient
from horse_racing.core.logging import logger
from horse_racing.infrastructure.netkeiba.race_result import RaceResultNetkeibaRepository
from horse_racing.usecase.race_result import RaceResultUsecase


@dataclass
class TrainConfig:
    # constants
    model: str = "lightgbm"
    data_version: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_version: str = datetime.now().strftime("%Y%m%d_%H%M%S")


def collect_data(storage_client: StorageClient, version: str, tmp_dir: Path) -> pl.DataFrame:
    result_repository = RaceResultNetkeibaRepository(
        storage_client=storage_client,
        root_dir=tmp_dir,
    )
    result_usecase = RaceResultUsecase(race_result_repository=result_repository, root_dir=tmp_dir)
    return result_usecase.get(version=version)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--data-version", type=str)

    args, _ = parser.parse_known_args(namespace=TrainConfig())
    logger.info(f"{args=}")

    storage_client = StorageClient()

    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        raw_df = collect_data(storage_client=storage_client, version=args.data_version, tmp_dir=tmp_dir)
        logger.info(raw_df)

    # preprocess

    # split train and valid

    # hyper param tuning

    # train

    # save model


if __name__ == "__main__":
    main()
