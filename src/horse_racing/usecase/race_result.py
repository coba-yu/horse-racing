from io import StringIO
from pathlib import Path

import pandas as pd
import polars as pl

from horse_racing.infrastructure.netkeiba.race_result import RaceResultNetkeibaRepository


class RaceResultUsecase:
    def __init__(
        self,
        race_result_repository: RaceResultNetkeibaRepository,
        root_dir: Path,
    ) -> None:
        self.race_result_repository = race_result_repository
        self.root_dir = root_dir

    def get_raw_html(self, race_date: str, race_id: str) -> str:
        return self.race_result_repository.get_by_race_id(race_date=race_date, race_id=race_id)

    def get(self, version: str) -> pl.DataFrame:
        data_dir = Path(self.root_dir, "data")
        data_dir.mkdir(parents=True, exist_ok=True)

        if self.race_result_repository.exists_result_data_blob(version=version):
            result_path = self.race_result_repository.download_result_data_from_storage(
                version=version, data_dir=data_dir
            )
            return pl.read_parquet(result_path)

        for data in self.race_result_repository.get_iter():
            sub_dir = f'race_date={data["race_date"]}'

            table_pdf_list = pd.read_html(StringIO(data["html"]))

            result_df = pl.from_pandas(table_pdf_list[0])
            result_dir = data_dir / "race_results" / sub_dir
            result_dir.mkdir(parents=True, exist_ok=True)
            result_df.write_parquet(result_dir / f'{data["race_id"]}.parquet')
        df = pl.read_parquet(data_dir)

        # cache to storage
        result_path = data_dir / "race_result.parquet"
        df.write_parquet(result_path)
        self.race_result_repository.upload_data_to_storage(path=result_path, version=version)

        return df
