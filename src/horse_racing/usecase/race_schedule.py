import os.path
import re
from io import StringIO
from time import sleep

import pandas as pd
import polars as pl
from horse_racing.core.chrome import ChromeDriver
from horse_racing.core.html import get_html, get_soup


def extract_race_date(href: str) -> str | None:
    date_match = re.search(r"kaisai_date=(\d{8})", href)
    if date_match is None:
        return None
    _, _, race_date = date_match.group().partition("=")
    return race_date


class RaceScheduleUsecase:
    race_result_columns = [
        "着 順",
        "枠",
        "馬 番",
        "馬名",
        "性齢",
        "斤量",
        "騎手",
        "タイム",
        "着差",
        "人 気",
        "単勝 オッズ",
        "後3F",
        "コーナー 通過順",
        "厩舎",
        "馬体重 (増減)",
    ]

    def __init__(self, driver: ChromeDriver) -> None:
        self.driver = driver

    def _make_tmp_dir(self, sub_dir: str) -> str:
        tmp_dir = os.path.join("data", "tmp", "html", sub_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        return tmp_dir

    @staticmethod
    def get_race_dates(year: int, month: int) -> list[str]:
        url = f"https://race.netkeiba.com/top/calendar.html?year={year}&month={month}"
        html = get_html(url)
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
        tmp_dir = self._make_tmp_dir(sub_dir="race_list")
        tmp_html_path = os.path.join(tmp_dir, f"{race_date}.html")
        if os.path.isfile(tmp_html_path):
            with open(tmp_html_path, "r") as f:
                html = f.read()
        else:
            url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={race_date}"
            html = self.driver.get_page_source(url=url)
            with open(tmp_html_path, "w") as f:
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

    def get_race_result(self, race_id: str, race_date: str) -> pl.DataFrame:
        tmp_dir = self._make_tmp_dir(sub_dir=os.path.join("race_results", f"race_date={race_date}"))
        tmp_html_path = os.path.join(tmp_dir, f"{race_id}.html")

        if os.path.isfile(tmp_html_path):
            with open(tmp_html_path, "r") as f:
                html = f.read()
        else:
            url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
            html = get_html(url)
            sleep(1.0)
            with open(tmp_html_path, "w") as f:
                f.write(html)

        pdf_list = pd.read_html(StringIO(html), converters={c: str for c in self.race_result_columns})
        if len(pdf_list) < 1:
            return pl.DataFrame()
        df = pl.from_pandas(pdf_list[0])
        return df.with_columns(race_id=pl.lit(race_id), race_date=pl.lit(race_date))
