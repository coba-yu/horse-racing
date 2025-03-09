from io import StringIO

import pandas as pd
import polars as pl
from bs4 import BeautifulSoup

from horse_racing.core.chrome import ChromeDriver
from horse_racing.core.html import get_soup
from horse_racing.domain.race import RaceInfo


class RaceCardUsecase:
    def __init__(
        self,
        driver: ChromeDriver | None = None,
        root_dir: str = ".",
    ) -> None:
        self.driver = driver
        self.root_dir = root_dir

        self._soup: BeautifulSoup | None = None

    def _get_soup(self, race_id: str) -> BeautifulSoup:
        if self._soup is not None:
            return self._soup

        if self.driver is None:
            raise ValueError("driver is not set")

        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        html = self.driver.get_page_source(url=url)
        self._soup = get_soup(html)
        return self._soup

    def get_race_info(self, race_id: str) -> RaceInfo:
        soup = self._get_soup(race_id)

        race_number = int(race_id[-2:])

        race_list_name_box = soup.find("div", class_="RaceList_NameBox")
        race_text = race_list_name_box.select_one(".RaceList_Item02 .RaceData01").get_text(strip=True)
        race_texts = race_text.replace(" ", "").split("/")

        start_hour = int(race_texts[0].split(":")[0])
        distance = race_texts[1]
        if len(race_texts) >= 3:
            weather = race_texts[2]
        else:
            weather = None
        if len(race_texts) >= 4:
            field_condition = race_texts[3]
        else:
            field_condition = None

        return RaceInfo(
            race_number=race_number,
            start_hour=start_hour,
            distance=distance,
            weather=weather,
            field_condition=field_condition,
        )

    def get_race_card(self, race_id: str) -> pl.DataFrame:
        soup = self._get_soup(race_id)
        pdf_list = pd.read_html(StringIO(str(soup)))
        if len(pdf_list) < 1:
            return pl.DataFrame()
        race_pdf = pdf_list[0]
        race_pdf.columns = [c for c, _ in race_pdf.columns]
        for c in ("お気に入り馬", "馬メモ切替"):
            if c in list(race_pdf):
                race_pdf.drop(c, axis=1, inplace=True)

        race_df = pl.from_pandas(race_pdf)
        return race_df
