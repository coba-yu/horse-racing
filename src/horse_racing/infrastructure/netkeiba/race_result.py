import os.path
import re
from collections.abc import Generator
from pathlib import Path
from typing import Any

from google.cloud.storage import Blob, Bucket

from horse_racing.core.chrome import ChromeDriver
from horse_racing.core.gcp.storage import StorageClient
from horse_racing.core.html import make_cache_dir
from horse_racing.core.logging import logger


class RaceResultNetkeibaRepository:
    def __init__(
        self,
        storage_client: StorageClient,
        root_dir: Path,
        driver: ChromeDriver | None = None,
        html_bucket_name: str = "yukob-netkeiba-htmls",
        data_bucket_name: str = "yukob-netkeiba-data",
    ) -> None:
        self.driver = driver
        self.storage_client = storage_client
        self.html_bucket_name = html_bucket_name
        self.data_bucket_name = data_bucket_name
        self.cache_dir = Path(make_cache_dir(sub_dir="race_result", root_dir=root_dir))

    def get_cache_path(self, race_date: str, race_id: str) -> Path:
        return self.cache_dir / f"race_date={race_date}" / f"{race_id}.html"

    @property
    def _html_bucket(self) -> Bucket:
        return self.storage_client.get_bucket(bucket_name=self.html_bucket_name)

    def get_html_blob(self, race_date: str, race_id: str) -> Blob:
        return self._html_bucket.blob(f"race_result/race_date={race_date}/{race_id}.html")

    def upload_html_to_storage(self, race_date: str, race_id: str) -> None:
        cache_path = self.get_cache_path(race_date=race_date, race_id=race_id)
        blob = self.get_html_blob(race_date=race_date, race_id=race_id)
        logger.info(f"Uploading {cache_path} to {blob.path}")
        blob.upload_from_filename(filename=str(cache_path))

    def _download_from_netkeiba(self, race_date: str, race_id: str) -> str:
        if self.driver is None:
            raise ValueError("missing driver")

        cache_path = self.get_cache_path(race_date=race_date, race_id=race_id)
        url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"

        logger.info(f"Downloading {url} to {cache_path}")
        html = self.driver.get_page_source(url=url)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            f.write(html)
        self.upload_html_to_storage(race_date=race_date, race_id=race_id)
        return html

    def get_by_race_id(self, race_date: str, race_id: str) -> str:
        cache_path = self.get_cache_path(race_date=race_date, race_id=race_id)
        if cache_path.exists():
            with open(cache_path, "r") as fp:
                html = fp.read()
            if len(html) > 0:
                logger.info(f"Local cache found: {cache_path}")
                return html

        blob = self.get_html_blob(race_date=race_date, race_id=race_id)
        if blob.exists():
            html = blob.download_as_text()
            if len(html) > 0:
                logger.info(f"GCS cache found: {blob.path}")
                return html

        return self._download_from_netkeiba(race_date=race_date, race_id=race_id)

    def get_iter(
        self,
        first_date: str | None = None,
        last_date: str | None = None,
    ) -> Generator[dict[str, str], Any, None]:
        for blob in self._html_bucket.list_blobs(prefix="race_result"):
            # race_result/race_date={race_date}/{race_id}.html
            blob_name = blob.name

            # extract race_date
            race_date_match = re.search(r"race_date=\d+", blob_name)
            if race_date_match is None:
                logger.warning(f"Unmatch race_date pattern: {blob_name}")
                continue
            race_date = race_date_match.group().replace("race_date=", "")

            if first_date is not None and race_date < first_date:
                continue
            if last_date is not None and race_date > last_date:
                continue

            # extract race_id
            race_id, _ = os.path.splitext(os.path.basename(blob.name))

            yield {
                "race_id": race_id,
                "race_date": race_date,
                "html": blob.download_as_text(),
            }

    def _get_data_blob(self, version: str, data_name: str) -> Blob:
        return self.storage_client.get_bucket(self.data_bucket_name).blob(f"data_version={version}/{data_name}.parquet")

    def _exists_data_blob(self, version: str, data_name: str) -> bool:
        return bool(self._get_data_blob(version=version, data_name=data_name).exists())

    def _download_data_from_storage(self, version: str, data_name: str, data_dir: Path) -> Path:
        data_dir.mkdir(parents=True, exist_ok=True)
        data_path = data_dir / f"{data_name}.parquet"
        blob = self._get_data_blob(version=version, data_name=data_name)
        blob.download_to_filename(str(data_path))
        return data_path

    def upload_data_to_storage(self, path: Path, version: str) -> None:
        data_name, _ = os.path.splitext(os.path.basename(path))
        blob = self._get_data_blob(version=version, data_name=data_name)
        blob.upload_from_filename(str(path))

    def exists_result_data_blob(self, version: str) -> bool:
        return self._exists_data_blob(version=version, data_name="race_result")

    def download_result_data_from_storage(self, version: str, data_dir: Path) -> Path:
        return self._download_data_from_storage(version=version, data_name="race_result", data_dir=data_dir)
