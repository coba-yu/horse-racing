from pathlib import Path

from horse_racing.core.chrome import ChromeDriver
from horse_racing.core.gcp.storage import StorageClient
from horse_racing.core.html import make_cache_dir
from horse_racing.core.logging import logger


class RaceResultNetkeibaRepository:
    def __init__(
        self,
        driver: ChromeDriver,
        storage_client: StorageClient,
        root_dir: Path,
        bucket_name: str = "yukob-netkeiba-htmls",
    ) -> None:
        self.driver = driver
        self.storage_client = storage_client
        self.bucket_name = bucket_name
        self.cache_dir = Path(make_cache_dir(sub_dir="race_result", root_dir=root_dir))

    def get_cache_path(self, race_date: str, race_id: str) -> Path:
        return self.cache_dir / f"race_date={race_date}" / f"{race_id}.html"

    def download(self, race_date: str, race_id: str) -> str:
        cache_path = self.get_cache_path(race_date=race_date, race_id=race_id)
        url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"

        logger.info(f"Downloading {url} to {cache_path}")
        html = self.driver.get_page_source(url=url)
        with open(cache_path, "w") as f:
            f.write(html)
        return html

    def upload_to_storage(self, race_date: str, race_id: str) -> None:
        cache_path = self.get_cache_path(race_date=race_date, race_id=race_id)
        bucket = self.storage_client.get_bucket(bucket_name=self.bucket_name)
        blob = bucket.blob(f"race_result/race_date={race_date}/{race_id}.html")
        logger.info(f"Uploading {cache_path} to {blob.path}")
        blob.upload_from_filename(filename=str(cache_path))
