from pathlib import Path

import pandas as pd
import polars as pl
from bs4 import BeautifulSoup

from horse_racing.core.html import get_html, get_soup


class HorseUsecase:
    result_columns = [
        "日付",
        "開催",
        "天 気",
        "R",
        "レース名",
        "頭 数",
        "枠 番",
        "馬 番",
        "オ ッ ズ",
        "人 気",
        "着 順",
        "騎手",
        "斤 量",
        "距離",
        "馬 場",
        "タイム",
        "着差",
        "通過",
        "ペース",
        "上り",
        "馬体重",
        "勝ち馬 (2着馬)",
        "賞金",
    ]

    def _get_horse_html(self, horse_id: str) -> BeautifulSoup:
        url = f"https://db.netkeiba.com/horse/{horse_id}"
        html = get_html(url=url, cache_sub_path=Path("horses", f"{horse_id}.html"))
        return get_soup(html)

    def get_horse_profile(self, horse_id: str) -> pl.DataFrame:
        soup = self._get_horse_html(horse_id=horse_id)
        table = soup.find("table", class_="db_prof_table")
        pdf_list = pd.read_html(table.decode())
        if len(pdf_list) < 1:
            return pl.DataFrame()
        return pl.from_pandas(pdf_list[0])

    def get_horse_results(self, horse_id: str) -> pl.DataFrame:
        soup = self._get_horse_html(horse_id=horse_id)
        table = soup.find("table", class_="db_h_race_results")
        pdf_list = pd.read_html(table.decode(), converters={c: str for c in self.result_columns})
        if len(pdf_list) < 1:
            return pl.DataFrame()
        return pl.from_pandas(pdf_list[0]).select(self.result_columns)
