import os.path
import re
from io import StringIO
from typing import Any

import pandas as pd
import polars as pl
from bs4 import BeautifulSoup
from bs4.element import Tag

from horse_racing.core.chrome import ChromeDriver
from horse_racing.core.html import get_html, get_soup, make_cache_dir


def extract_race_date(href: str) -> str | None:
    date_match = re.search(r"kaisai_date=(\d{8})", href)
    if date_match is None:
        return None
    _, _, race_date = date_match.group().partition("=")
    return race_date


class RaceScheduleUsecase:
    horse_number_raw_column = "馬 番"
    horse_number_column = "horse_number"
    horse_name_column = "馬名"
    jockey_name_column = "騎手"
    trainer_name_column = "厩舎"
    race_result_columns = [
        "着 順",
        "枠",
        horse_number_raw_column,
        horse_name_column,
        "性齢",
        "斤量",
        jockey_name_column,
        "タイム",
        "着差",
        "人 気",
        "単勝 オッズ",
        "後3F",
        "コーナー 通過順",
        trainer_name_column,
        "馬体重 (増減)",
    ]

    def __init__(
        self,
        driver: ChromeDriver | None = None,
        root_dir: str = ".",
    ) -> None:
        self.driver = driver
        self.root_dir = root_dir

    def _remove_whitespace(self, df: pl.DataFrame, column: str) -> pl.DataFrame:
        df = df.with_columns(pl.col(column).str.replace(r"^\s+", "").alias(column))
        return df.with_columns(pl.col(column).str.replace(r"\s+$", "").alias(column))

    def _extracted_id_df(self, tag: Tag, href_key: str, id_column_prefix: str, name_column: str) -> pl.DataFrame:
        a_list = tag.find_all("a", href=re.compile(rf"{href_key}/[\d\w]+"))
        name_values: list[str | None] = []
        id_values: list[str | None] = []
        for a in a_list:
            name_values.append(a.text)

            href = a.get("href")
            if href is None:
                id_values.append(None)
                continue

            ids = re.findall(rf"{href_key}/([\d\w]+)", href)
            if ids is None or len(ids) < 1:
                id_values.append(None)
                continue
            id_values.append(ids[0])

        return pl.DataFrame(
            {
                name_column: name_values,
                f"{id_column_prefix}_id": id_values,
            },
        )

    def get_race_dates(self, year: int, month: int) -> list[str]:
        url = f"https://race.netkeiba.com/top/calendar.html?year={year}&month={month}"
        html = get_html(url, root_dir=self.root_dir)
        soup = get_soup(html)

        table = soup.find("table", class_="Calendar_Table")
        a_tags = table.find_all("a")
        href_list = [tag.get("href") for tag in a_tags if tag.get("href") is not None]

        race_dates = []
        for href in href_list:
            race_date = extract_race_date(href)
            if race_date is None:
                continue
            race_dates.append(race_date)
        return race_dates

    def get_race_ids(self, race_date: str) -> list[str]:
        cache_dir = make_cache_dir(sub_dir="race_schedule", root_dir=self.root_dir)
        cache_path = os.path.join(cache_dir, f"{race_date}.html")
        if os.path.isfile(cache_path):
            with open(cache_path, "r") as f:
                html = f.read()
        else:
            if self.driver is None:
                raise ValueError(f"Cache is not found but self.driver is None.\n{cache_path=}")
            url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={race_date}"
            html = self.driver.get_page_source(url=url)
            with open(cache_path, "w") as f:
                f.write(html)

        soup = get_soup(html)
        race_list_items = soup.find_all("li", class_="RaceList_DataItem")
        race_ids = []
        for race_item in race_list_items:
            href = race_item.find("a").get("href")
            if href is None:
                continue

            race_id_query_match = re.search(r"race_id=[\d\w]+", href)
            if race_id_query_match is None:
                continue
            race_id_query = race_id_query_match.group()
            _, _, race_id = race_id_query.partition("=")
            race_ids.append(race_id)

        return race_ids

    @staticmethod
    def _get_race_info(soup: BeautifulSoup) -> dict[str, Any]:
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

    @classmethod
    def _get_payout_df(cls, soup: BeautifulSoup) -> pl.DataFrame:
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
            pl.col("horse_numbers").alias(cls.horse_number_column),
            pl.col("payouts").cast(pl.Int32).alias("win_payout"),
        )

        place_column = "複勝"
        place_payout_df = payout_df.filter(pl.col("ticket_type") == place_column).explode(["horse_numbers", "payouts"])
        place_payout_df = place_payout_df.select(
            pl.col("horse_numbers").alias(cls.horse_number_column),
            pl.col("payouts").cast(pl.Int32).alias("place_payout"),
        )
        return win_payout_df.join(place_payout_df, on=cls.horse_number_column, how="outer")

    def get_race_result(self, race_id: str, race_date: str) -> pl.DataFrame:
        url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
        html = get_html(
            url=url,
            root_dir=self.root_dir,
            cache_sub_path=os.path.join("race_results", f"race_date={race_date}", f"{race_id}.html"),
        )

        pdf_list = pd.read_html(StringIO(html), converters={c: str for c in self.race_result_columns})
        if len(pdf_list) < 1:
            return pl.DataFrame()
        df = pl.from_pandas(pdf_list[0])
        df = df.with_columns(race_id=pl.lit(race_id), race_date=pl.lit(race_date))
        df = df.rename({self.horse_number_raw_column: self.horse_number_column})

        soup = get_soup(html)

        race_info = self._get_race_info(soup=soup)
        df = df.with_columns([pl.lit(v).alias(k) for k, v in race_info.items()])

        table = soup.find("table", class_="RaceTable01")
        horse_id_df = self._extracted_id_df(
            table,
            href_key="horse",
            id_column_prefix="horse",
            name_column=self.horse_name_column,
        )
        df = self._remove_whitespace(df, column=self.horse_name_column)
        horse_id_df = self._remove_whitespace(horse_id_df, column=self.horse_name_column)
        df = df.join(horse_id_df, on=self.horse_name_column, how="left")

        jockey_id_df = self._extracted_id_df(
            table,
            href_key="jockey/result/recent",
            id_column_prefix="jockey",
            name_column=self.jockey_name_column,
        )
        df = self._remove_whitespace(df, column=self.jockey_name_column)
        jockey_id_df = self._remove_whitespace(jockey_id_df, column=self.jockey_name_column)
        df = df.join(jockey_id_df, on=self.jockey_name_column, how="left")

        trainer_id_df = self._extracted_id_df(
            table,
            href_key="trainer/result/recent",
            id_column_prefix="trainer",
            name_column=self.trainer_name_column,
        )
        df = self._remove_whitespace(df, column=self.trainer_name_column)
        trainer_id_df = self._remove_whitespace(trainer_id_df, column=self.trainer_name_column)
        a_list = table.find_all("a", href=re.compile(r"trainer/result/recent/[\d\w]+"))
        trainer_id_df = trainer_id_df.with_columns(
            trainer_label=pl.Series([a.parent.find("span").text for a in a_list])
        )
        trainer_id_df = self._remove_whitespace(trainer_id_df, column="trainer_label")
        trainer_id_df = trainer_id_df.with_columns(
            pl.concat_str(pl.col("trainer_label"), pl.col(self.trainer_name_column)).alias(self.trainer_name_column)
        )
        df = df.join(trainer_id_df, on=self.trainer_name_column, how="left")

        # payout
        suffix = "_right"
        df = df.join(self._get_payout_df(soup=soup), on=self.horse_number_column, how="left", suffix=suffix)
        df = df.drop(f"{self.horse_number_column}{suffix}")

        return df
