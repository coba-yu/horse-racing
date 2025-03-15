from io import StringIO
from pathlib import Path

import pandas as pd
import polars as pl
from tqdm import tqdm

from horse_racing.core.logging import logger
from horse_racing.infrastructure.netkeiba.race_result import RaceResultNetkeibaRepository

# raw -> renamed
_RESULT_COLUMN_RENAME_DICT = {
    "着 順": "rank",
    "枠": "frame",
    "馬 番": "horse_number",
    "馬名": "horse_name",
    "性齢": "gender_age",
    "斤量": "total_weight",
    "騎手": "jockey_name",
    "タイム": "goal_time",
    "着差": "goal_diff",
    "人 気": "popular",
    "単勝 オッズ": "odds",
    "後3F": "last_3f_time",
    "コーナー 通過順": "corner_rank",
    "厩舎": "trainer_name",
    "馬体重 (増減)": "horse_weight_and_diff",
}


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

    def get(self, version: str, first_date: str | None = None, last_date: str | None = None) -> pl.DataFrame:
        data_dir = Path(self.root_dir, "data")
        data_dir.mkdir(parents=True, exist_ok=True)

        if self.race_result_repository.exists_result_data_blob(version=version):
            result_path = self.race_result_repository.download_result_data_from_storage(
                version=version, data_dir=data_dir
            )
            return pl.read_parquet(result_path)

        for data in tqdm(self.race_result_repository.get_iter(first_date=first_date, last_date=last_date)):
            race_id = data["race_id"]
            sub_dir = f'race_date={data["race_date"]}'

            try:
                table_pdf_list = pd.read_html(
                    StringIO(data["html"]),
                    converters={c: str for c, _ in _RESULT_COLUMN_RENAME_DICT.items()},
                )

                result_df = pl.from_pandas(table_pdf_list[0])
                result_df = result_df.rename(_RESULT_COLUMN_RENAME_DICT)

                result_dir = data_dir / "race_results" / sub_dir
                result_dir.mkdir(parents=True, exist_ok=True)
                result_df.write_parquet(result_dir / f"{race_id}.parquet")
            except ValueError:
                logger.error(f"Error: {race_id=}")
                raise
        df = pl.read_parquet(data_dir)

        # cache to storage
        result_path = data_dir / "race_result.parquet"
        df.write_parquet(result_path)
        self.race_result_repository.upload_data_to_storage(path=result_path, version=version)

        return df
