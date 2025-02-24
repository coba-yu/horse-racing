from dataclasses import dataclass
from argparse import ArgumentParser

from horse_racing.core.logging import logger


@dataclass(frozen=True)
class Argument:
    dt: str | None = None


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--dt", type=str)

    args = Argument()
    args, _ = parser.parse_known_args(namespace=args)
    logger.info(f"{args=}")


if __name__ == "__main__":
    main()
