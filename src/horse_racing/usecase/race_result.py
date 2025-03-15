import re
from io import StringIO
from pathlib import Path
from traceback import format_exc
from typing import Any

import pandas as pd
import polars as pl
from bs4 import BeautifulSoup
from tqdm import tqdm

from horse_racing.core.html import get_soup
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


def extract_race_info(soup: BeautifulSoup) -> dict[str, Any]:
    race_info: dict[str, Any] = {}

    # => "1R"
    race_list_name_box = soup.find("div", class_="RaceList_NameBox")
    race_num_tag = race_list_name_box.select_one(".RaceList_Item01 .RaceNum")
    if race_num_tag is None:
        race_info["race_number"] = None
    else:
        race_info["race_number"] = race_num_tag.get_text(strip=True)

    # => "3歳未勝利"
    race_name_tag = race_list_name_box.select_one(".RaceList_Item02 .RaceName")
    if race_num_tag is None:
        race_info["race_name"] = None
    else:
        race_info["race_name"] = race_name_tag.get_text(strip=True)

    # => "10:10発走 / ダ1400m (左) / 天候:晴 / 馬場:良"
    race_data_1_tag = race_list_name_box.select_one(".RaceList_Item02 .RaceData01")
    if race_data_1_tag is None:
        race_info_texts = []
    else:
        race_info_texts = re.sub(r"\s+", "", race_data_1_tag.get_text(strip=True)).split("/")

    for i, k in enumerate(("start_at", "distance", "weather", "field_condition")):
        if len(race_info_texts) >= i + 1:
            race_info[k] = race_info_texts[i]
        else:
            race_info[k] = None

    return dict(**race_info)


def convert_html_to_dataframe(html: str, race_date: str, race_id: str) -> pl.DataFrame:
    # base dataframe
    table_pdf_list = pd.read_html(
        StringIO(html),
        converters={c: str for c, _ in _RESULT_COLUMN_RENAME_DICT.items()},
    )
    df = pl.from_pandas(table_pdf_list[0])
    df = df.rename(_RESULT_COLUMN_RENAME_DICT)

    df = df.with_columns(race_id=pl.lit(race_id), race_date=pl.lit(race_date))

    # extract info
    soup = get_soup(html)
    race_info = extract_race_info(soup=soup)
    df = df.with_columns([pl.lit(v).alias(k) for k, v in race_info.items()])

    return df


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

        for data in tqdm(
            self.race_result_repository.get_iter(first_date=first_date, last_date=last_date),
            mininterval=60.0,
            maxinterval=180.0,
        ):
            race_id = data["race_id"]
            race_date = data["race_date"]
            sub_dir = f"race_date={race_date}"

            try:
                result_df = convert_html_to_dataframe(html=data["html"], race_date=race_date, race_id=race_id)

                result_dir = data_dir / "race_results" / sub_dir
                result_dir.mkdir(parents=True, exist_ok=True)
                result_df.write_parquet(result_dir / f"{race_id}.parquet")
            except ValueError:
                logger.error(f"Error: {race_id=}, {race_date=}\n{format_exc()}")
        df = pl.read_parquet(data_dir)

        # cache to storage
        result_path = data_dir / "race_result.parquet"
        df.write_parquet(result_path)
        self.race_result_repository.upload_data_to_storage(path=result_path, version=version)

        return df
