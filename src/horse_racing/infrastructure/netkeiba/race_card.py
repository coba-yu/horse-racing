from pathlib import Path

from horse_racing.core.chrome import ChromeDriver
from horse_racing.core.html import make_cache_dir
from horse_racing.core.logging import logger


class RaceCardNetkeibaRepository:
    def __init__(
        self,
        root_dir: Path,
        driver: ChromeDriver,
    ) -> None:
        self.driver = driver
        self.cache_dir = Path(make_cache_dir(sub_dir="race_card", root_dir=root_dir))

    def get_cache_path(self, race_date: str, race_id: str) -> Path:
        return self.cache_dir / f"race_date={race_date}" / f"{race_id}.html"

    def _download_from_netkeiba(self, race_date: str, race_id: str) -> str:
        cache_path = self.get_cache_path(race_date=race_date, race_id=race_id)
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"

        logger.info(f"Downloading {url} to {cache_path}")
        html = self.driver.get_page_source(url=url, skip_sleep=True)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            f.write(html)
        return html

    def get_by_race_id(self, race_date: str, race_id: str) -> str:
        cache_path = self.get_cache_path(race_date=race_date, race_id=race_id)
        if cache_path.exists():
            with open(cache_path, "r") as fp:
                html = fp.read()
            if len(html) > 0:
                logger.info(f"Local cache found: {cache_path}")
                return html

        return self._download_from_netkeiba(race_date=race_date, race_id=race_id)
