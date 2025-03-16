import re
from io import StringIO
from pathlib import Path
from traceback import format_exc
from typing import Any

import pandas as pd
import polars as pl
from bs4 import BeautifulSoup
from bs4.element import Tag
from tqdm import tqdm

from horse_racing.core.html import get_soup
from horse_racing.core.logging import logger
from horse_racing.infrastructure.netkeiba.race_result import RaceResultNetkeibaRepository


class Column:
    RANK: str = "rank"
    FRAME: str = "frame"
    HORSE_NUMBER: str = "horse_number"
    GENDER: str = "gender"
    AGE: str = "age"
    TOTAL_WEIGHT: str = "total_weight"
    HORSE_WEIGHT: str = "horse_weight"
    GOAL_TIME: str = "goal_time"
    GOAL_DIFF: str = "goal_diff"
    POPULAR: str = "popular"
    ODDS: str = "odds"
    LAST_3F_TIME: str = "last_3f_time"
    CORNER_RANK: str = "corner_rank"

    # race info
    RACE_NUMBER: str = "race_number"
    RACE_NAME: str = "race_name"
    START_AT: str = "start_at"
    DISTANCE: str = "distance"
    ROTATE: str = "rotate"
    FIELD_TYPE: str = "field_type"
    WEATHER: str = "weather"
    FIELD_CONDITION: str = "field_condition"

    WIN_PAYOUT: str = "win_payout"
    PLACE_PAYOUT: str = "place_payout"

    # id
    HORSE_ID: str = "horse_id"
    JOCKEY_ID: str = "jockey_id"
    TRAINER_ID: str = "trainer_id"

    # not feature
    HORSE_NAME: str = "horse_name"
    JOCKEY_NAME: str = "jockey_name"
    TRAINER_NAME: str = "trainer_name"


GENDER_AGE_COLUMN = f"{Column.GENDER}_{Column.AGE}"
HORSE_WEIGHT_AND_DIFF_COLUMN = "horse_weight_and_diff"

# raw -> renamed
_RESULT_COLUMN_RENAME_DICT = {
    "着 順": Column.RANK,
    "枠": Column.FRAME,
    "馬 番": Column.HORSE_NUMBER,
    "馬名": Column.HORSE_NAME,
    "性齢": GENDER_AGE_COLUMN,
    "斤量": Column.TOTAL_WEIGHT,
    "騎手": Column.JOCKEY_NAME,
    "タイム": Column.GOAL_TIME,
    "着差": Column.GOAL_DIFF,
    "人 気": Column.POPULAR,
    "単勝 オッズ": Column.ODDS,
    "後3F": Column.LAST_3F_TIME,
    "コーナー 通過順": Column.CORNER_RANK,
    "厩舎": Column.TRAINER_NAME,
    "馬体重 (増減)": HORSE_WEIGHT_AND_DIFF_COLUMN,
}


def extract_race_info(soup: BeautifulSoup) -> dict[str, Any]:
    race_info: dict[str, Any] = {}

    # => "1R"
    race_list_name_box = soup.find("div", class_="RaceList_NameBox")
    race_num_tag = race_list_name_box.select_one(".RaceList_Item01 .RaceNum")
    if race_num_tag is None:
        race_info[Column.RACE_NUMBER] = None
    else:
        race_info[Column.RACE_NUMBER] = race_num_tag.get_text(strip=True)

    # => "3歳未勝利"
    race_name_tag = race_list_name_box.select_one(".RaceList_Item02 .RaceName")
    if race_num_tag is None:
        race_info[Column.RACE_NAME] = None
    else:
        race_info[Column.RACE_NAME] = race_name_tag.get_text(strip=True)

    # => "10:10発走 / ダ1400m (左) / 天候:晴 / 馬場:良"
    race_data_1_tag = race_list_name_box.select_one(".RaceList_Item02 .RaceData01")
    if race_data_1_tag is None:
        race_info_texts = []
    else:
        race_info_texts = re.sub(r"\s+", "", race_data_1_tag.get_text(strip=True)).split("/")

    for i, k in enumerate((Column.START_AT, Column.DISTANCE, Column.WEATHER, Column.FIELD_CONDITION)):
        if len(race_info_texts) >= i + 1:
            race_info[k] = race_info_texts[i]
        else:
            race_info[k] = None

    return dict(**race_info)


def _remove_whitespace(df: pl.DataFrame, column: str) -> pl.DataFrame:
    df = df.with_columns(pl.col(column).str.replace(r"^\s+", "").alias(column))
    return df.with_columns(pl.col(column).str.replace(r"\s+$", "").alias(column))


def _extracted_id_df(tag: Tag, href_key: str, id_column_prefix: str, name_column: str) -> pl.DataFrame:
    a_list = tag.find_all("a", href=re.compile(rf"{href_key}/[\d\w]+"))
    name_values: list[str | None] = []
    id_values: list[str | None] = []
    label_values: list[str | None] = []
    for a in a_list:
        name = a.text
        if id_column_prefix == "trainer":
            # e.g.
            # <span class="Label1">美浦</span>
            # <a href="https://db.netkeiba.com/trainer/result/recent/01169/" target="_blank" title="加藤士">加藤士</a>
            label = a.parent.find("span").text
            label_values.append(label)
            name = "".join((label, name))
        name_values.append(name)

        href = a.get("href")
        if href is None:
            id_values.append(None)
            continue

        ids = re.findall(rf"{href_key}/([\d\w]+)", href)
        if ids is None or len(ids) < 1:
            id_values.append(None)
            continue
        id_values.append(ids[0])

    df = pl.DataFrame(
        {
            name_column: name_values,
            f"{id_column_prefix}_id": id_values,
        },
    )
    if len(label_values) > 0:
        df = df.with_columns(tmp_label=pl.Series(label_values))
        df = df.rename({"tmp_label": f"{id_column_prefix}_label"})

    return df


def extract_payout_df(soup: BeautifulSoup) -> pl.DataFrame:
    payout_pdfs = []
    for t in soup.find_all("table", class_="Payout_Detail_Table"):
        pdf_list = pd.read_html(StringIO(str(t)))
        if len(pdf_list) <= 0:
            continue
        pdf = pdf_list[0]
        pdf.columns = ["ticket_type", "horse_numbers", "payouts", "populars"]
        payout_pdfs.append(pl.from_pandas(pdf))

    payout_df = pl.concat(payout_pdfs, how="vertical")
    payout_df = payout_df.drop("populars")
    payout_df = payout_df.with_columns(
        pl.col("horse_numbers").str.split(" "),
        pl.col("payouts").str.replace_all(",", "").str.replace_all("円", "").str.extract_all(r"[0-9]+"),
    )

    win_column = "単勝"
    win_payout_df = payout_df.filter(pl.col("ticket_type") == win_column).explode(["horse_numbers", "payouts"])
    win_payout_df = win_payout_df.select(
        pl.col("horse_numbers").alias(Column.HORSE_NUMBER),
        pl.col("payouts").cast(pl.Int32).alias("win_payout"),
    )

    place_column = "複勝"
    place_payout_df = payout_df.filter(pl.col("ticket_type") == place_column).explode(["horse_numbers", "payouts"])
    place_payout_df = place_payout_df.select(
        pl.col("horse_numbers").alias(Column.HORSE_NUMBER),
        pl.col("payouts").cast(pl.Int32).alias("place_payout"),
    )
    return win_payout_df.join(place_payout_df, on=Column.HORSE_NUMBER, how="outer")


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

    # horse / jockey / trainer id
    table = soup.find("table", class_="RaceTable01")
    for href_key, id_column_prefix, name_column in (
        ("horse", "horse", Column.HORSE_NAME),
        ("jockey/result/recent", "jockey", Column.JOCKEY_NAME),
        ("trainer/result/recent", "trainer", Column.TRAINER_NAME),
    ):
        id_df = _extracted_id_df(
            table,
            href_key=href_key,
            id_column_prefix=id_column_prefix,
            name_column=name_column,
        )
        id_df = _remove_whitespace(id_df, column=name_column)

        df = _remove_whitespace(df, column=name_column)
        df = df.join(id_df, on=name_column, how="left")

    # payout
    suffix = "_right"
    df = df.join(extract_payout_df(soup=soup), on=Column.HORSE_NUMBER, how="left", suffix=suffix)
    df = df.drop(f"{Column.HORSE_NUMBER}{suffix}")

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

                result_dir = data_dir / "race_result" / sub_dir
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
