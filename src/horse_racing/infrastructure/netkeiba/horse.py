from pathlib import Path

from horse_racing.core.chrome import ChromeDriver
from horse_racing.core.gcp.storage import StorageClient
from horse_racing.infrastructure.netkeiba.base import BaseNetkeibaRepository


class HorseNetkeibaRepository(BaseNetkeibaRepository):
    def __init__(
        self,
        storage_client: StorageClient,
        root_dir: Path,
        driver: ChromeDriver | None = None,
    ) -> None:
        super().__init__(
            driver=driver,
            storage_client=storage_client,
            root_dir=root_dir,
            url_template="https://db.netkeiba.com/horse/{horse_id}",
            sub_dir_name="horse",
        )

    def get_by_id(
        self,
        horse_id: str,
        race_date: str | None = None,
        force_netkeiba: bool = False,
    ) -> str:
        return self._get_by_id(
            partition=[("horse_id", horse_id)],
            url_params={"horse_id": horse_id},
            file_stem=race_date,
            force_netkeiba=force_netkeiba,
        )
