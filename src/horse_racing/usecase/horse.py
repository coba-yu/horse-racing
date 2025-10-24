from pathlib import Path

import pandas as pd
import polars as pl
from bs4 import BeautifulSoup

from horse_racing.core.html import get_soup
from horse_racing.infrastructure.netkeiba.horse import HorseNetkeibaRepository


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

    def __init__(
        self,
        horse_repository: HorseNetkeibaRepository,
        root_dir: Path,
    ) -> None:
        self._horse_repository = horse_repository
        self.root_dir = root_dir

    def get_raw_html(
        self,
        horse_id: str,
        race_date: str | None = None,
        force_netkeiba: bool = False,
    ) -> BeautifulSoup:
        return self._horse_repository.get_by_id(horse_id=horse_id, race_date=race_date, force_netkeiba=force_netkeiba)

    def _get_soup(
        self,
        horse_id: str,
        race_date: str | None = None,
        force_netkeiba: bool = False,
    ) -> BeautifulSoup:
        html = self.get_raw_html(horse_id=horse_id, race_date=race_date, force_netkeiba=force_netkeiba)
        return get_soup(html)

    def get_horse_profile(
        self,
        horse_id: str,
        race_date: str | None = None,
        force_netkeiba: bool = False,
    ) -> pl.DataFrame:
        soup = self._get_soup(horse_id=horse_id, race_date=race_date, force_netkeiba=force_netkeiba)
        table = soup.find("table", class_="db_prof_table")
        pdf_list = pd.read_html(table.decode())
        if len(pdf_list) < 1:
            return pl.DataFrame()
        return pl.from_pandas(pdf_list[0])

    def get_horse_results(
        self,
        horse_id: str,
        race_date: str | None = None,
        force_netkeiba: bool = False,
    ) -> pl.DataFrame:
        soup = self._get_soup(horse_id=horse_id, race_date=race_date, force_netkeiba=force_netkeiba)
        table = soup.find("table", class_="db_h_race_results")
        pdf_list = pd.read_html(table.decode(), converters={c: str for c in self.result_columns})
        if len(pdf_list) < 1:
            return pl.DataFrame()
        return pl.from_pandas(pdf_list[0]).select(self.result_columns)
