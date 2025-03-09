import os

from google.cloud.storage import (
    Blob,
    Bucket,
    Client,
)


class StorageClient:
    def __init__(self) -> None:
        self._client = Client(project=os.environ["GOOGLE_PROJECT"])

    def get_bucket(self, bucket_name: str) -> Bucket:
        return self._client.get_bucket(bucket_name)

    def get_blob(self, bucket_name: str, blob_name: str) -> Blob:
        return self.get_bucket(bucket_name).get_blob(blob_name)
