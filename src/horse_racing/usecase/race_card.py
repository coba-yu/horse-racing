from io import StringIO

import pandas as pd
import polars as pl

from horse_racing.core.html import get_soup
from horse_racing.infrastructure.netkeiba.race_card import RaceCardNetkeibaRepository
from horse_racing.usecase.race_result import (
    ResultColumn,
    GENDER_AGE_COLUMN,
    HORSE_WEIGHT_AND_DIFF_COLUMN,
    extract_race_info,
    extract_id_columns,
)

# raw -> renamed
_RESULT_COLUMN_RENAME_DICT = {
    "枠": ResultColumn.FRAME,
    "馬 番": ResultColumn.HORSE_NUMBER,
    "馬名": ResultColumn.HORSE_NAME,
    "性齢": GENDER_AGE_COLUMN,
    "斤量": ResultColumn.TOTAL_WEIGHT,
    "騎手": ResultColumn.JOCKEY_NAME,
    "厩舎": ResultColumn.TRAINER_NAME,
    "馬体重 (増減)": HORSE_WEIGHT_AND_DIFF_COLUMN,
    "オッズ 更新": ResultColumn.ODDS,
    "人気": ResultColumn.POPULAR,
}


def convert_html_to_dataframe(html: str, race_date: str, race_id: str) -> pl.DataFrame:
    # base dataframe
    table_pdf_list = pd.read_html(StringIO(html))

    pdf = table_pdf_list[0]
    pdf = pdf.astype({c: str for c, _ in _RESULT_COLUMN_RENAME_DICT.items()})
    pdf.columns = [c for c, _ in pdf.columns]
    pdf.drop([c for c in pdf if c not in _RESULT_COLUMN_RENAME_DICT], axis=1, inplace=True)

    df = pl.from_pandas(pdf)
    df = df.rename(_RESULT_COLUMN_RENAME_DICT)

    df = df.with_columns(race_id=pl.lit(race_id), race_date=pl.lit(race_date))

    # extract info
    soup = get_soup(html)
    race_info = extract_race_info(soup=soup)
    df = df.with_columns([pl.lit(v).alias(k) for k, v in race_info.items()])

    # horse / jockey / trainer id
    df = extract_id_columns(df=df, soup=soup)

    return df


class RaceCardUsecase:
    def __init__(self, race_card_repository: RaceCardNetkeibaRepository) -> None:
        self.race_card_repository = race_card_repository

    def get_by_race_id(self, race_date: str, race_id: str) -> pl.DataFrame:
        html = self.race_card_repository.get_by_race_id(race_date=race_date, race_id=race_id)
        df = convert_html_to_dataframe(html=html, race_date=race_date, race_id=race_id)
        return df
