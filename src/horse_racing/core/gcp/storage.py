import os

from google.cloud.storage import (
    Blob,
    Bucket,
    Client,
)

from horse_racing.core.logging import logger


class StorageClient:
    def __init__(self) -> None:
        project = os.getenv("GOOGLE_PROJECT")
        if project is None:
            logger.warning("GOOGLE_PROJECT is not set.")
        self._client = Client(project=project)

    def get_bucket(self, bucket_name: str) -> Bucket:
        return self._client.get_bucket(bucket_name)

    def get_blob(self, bucket_name: str, blob_name: str) -> Blob:
        return self.get_bucket(bucket_name).get_blob(blob_name)
