from pathlib import Path
from typing import Any, Iterable

from google.cloud.storage import Blob, Bucket

from horse_racing.core.chrome import ChromeDriver
from horse_racing.core.datetime import get_current_yyyymmdd_hhmmss
from horse_racing.core.gcp.storage import StorageClient
from horse_racing.core.html import make_cache_dir
from horse_racing.core.logging import logger


class BaseNetkeibaRepository:
    html_bucket_name: str = "yukob-netkeiba-htmls"
    data_bucket_name: str = "yukob-netkeiba-data"

    def __init__(
        self,
        storage_client: StorageClient,
        url_template: str,
        root_dir: Path,
        sub_dir_name: str,
        driver: ChromeDriver | None = None,
    ) -> None:
        self.driver = driver
        self.storage_client = storage_client

        self.url_template = url_template

        self.sub_dir_name = sub_dir_name
        self.cache_dir = Path(make_cache_dir(sub_dir=sub_dir_name, root_dir=root_dir))

    @property
    def _html_bucket(self) -> Bucket:
        return self.storage_client.get_bucket(bucket_name=self.html_bucket_name)

    def get_cache_path(
        self,
        partition: Iterable[tuple[str, Any]] = (),
        file_stem: str | None = None,
    ) -> Path:
        dir_path = Path(self.cache_dir)
        for k, v in partition:
            dir_path /= f"{k}={v}"

        if file_stem is None:
            existing_paths = list(dir_path.glob("*.html"))
            if len(existing_paths) > 0:
                file_stem = sorted([p.stem for p in existing_paths])[-1]
            else:
                file_stem = get_current_yyyymmdd_hhmmss()

        return dir_path / f"{file_stem}.html"

    def get_html_blob(
        self,
        partition: Iterable[tuple[str, Any]] = (),
        file_stem: str | None = None,
    ) -> Blob:
        sub_dirs = [self.sub_dir_name]
        for k, v in partition:
            sub_dirs.append(f"{k}={v}")

        bucket = self._html_bucket
        if file_stem is None:
            blob_prefix = "/".join(sub_dirs)
            blobs = list(bucket.list_blobs(prefix=blob_prefix))
            if len(blobs) == 0:
                return None
            return sorted(blobs, key=lambda b: b.updated)[-1]

        blob_name = "/".join((*sub_dirs, f"{file_stem}.html"))
        return bucket.blob(blob_name)

    def upload_html_to_storage(
        self,
        partition: Iterable[tuple[str, Any]] = (),
        file_stem: str | None = None,
    ) -> None:
        cache_path = self.get_cache_path(partition=partition, file_stem=file_stem)
        blob = self.get_html_blob(partition=partition, file_stem=file_stem)
        logger.info(f"Uploading {cache_path} to {blob.path}")
        blob.upload_from_filename(filename=str(cache_path))

    def _download_from_netkeiba(
        self,
        partition: Iterable[tuple[str, Any]] = (),
        file_stem: str | None = None,
        url_params: dict[str, Any] | None = None,
    ) -> str:
        if self.driver is None:
            raise ValueError("missing driver")

        cache_path = self.get_cache_path(partition=partition, file_stem=file_stem)

        if url_params is None:
            url_params = {}
        url = self.url_template.format(**url_params)

        logger.info(f"Downloading {url} to {cache_path}")
        html = str(self.driver.get_page_source(url=url))

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            f.write(html)
        self.upload_html_to_storage(partition=partition, file_stem=cache_path.stem)
        return html

    def _get_by_id(
        self,
        partition: Iterable[tuple[str, Any]] = (),
        file_stem: str | None = None,
        url_params: dict[str, Any] | None = None,
        force_netkeiba: bool = False,
    ) -> str:
        if not force_netkeiba:
            cache_path = self.get_cache_path(partition=partition, file_stem=file_stem)
            if cache_path.exists():
                with open(cache_path, "r") as fp:
                    html = fp.read()
                if len(html) > 0:
                    logger.info(f"Local cache found: {cache_path}")
                    return html

            blob = self.get_html_blob(partition=partition, file_stem=file_stem)
            if blob is not None and blob.exists():
                html = blob.download_as_text()
                if len(html) > 0:
                    logger.info(f"GCS cache found: {blob.path}")
                    return html

        return self._download_from_netkeiba(
            partition=partition,
            file_stem=file_stem,
            url_params=url_params,
        )
