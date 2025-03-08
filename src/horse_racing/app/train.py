from argparse import ArgumentParser
from dataclasses import dataclass

from horse_racing.core.chrome import ChromeDriver
from horse_racing.core.logging import logger
from horse_racing.usecase.race_card import RaceCardUsecase


@dataclass
class Args:
    dt: str = ""
    race_id: str = ""


def train() -> None:
    # setup
    parser = ArgumentParser()
    parser.add_argument("--dt", type=str, required=True)
    parser.add_argument("--race_id", type=str, required=True)
    args = parser.parse_args(namespace=Args())

    logger.info(f"args: {args}")

    chrome_driver = ChromeDriver()

    # get race info
    race_card_usecase = RaceCardUsecase(driver=chrome_driver)
    race_info = race_card_usecase.get_race_info(race_id=args.race_id)
    logger.info(f"race_info: {race_info}")

    # training

    # save model


if __name__ == "__main__":
    train()
