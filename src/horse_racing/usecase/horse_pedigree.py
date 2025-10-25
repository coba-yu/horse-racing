import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from horse_racing.core.html import get_soup
from horse_racing.core.logging import logger
from horse_racing.infrastructure.netkeiba.horse_pedigree import HorsePedigreeNetkeibaRepository


def extract_gender(td: Tag) -> str:
    td_classes = td.get("class", [])
    if len(td_classes) == 0:
        return "unknown"
    td_class = td_classes[0]
    td_class_map = {
        "b_ml": "male",
        "b_fml": "female",
    }
    return td_class_map.get(td_class, "unknown")


def extract_row_span(td: Tag) -> int:
    row_span = td.get("rowspan")
    if row_span is None:
        return 1
    return int(row_span)


class HorsePedigreeUsecase:
    def __init__(
        self,
        horse_pedigree_repository: HorsePedigreeNetkeibaRepository,
        root_dir: Path,
    ) -> None:
        self._horse_pedigree_repository = horse_pedigree_repository
        self.root_dir = root_dir

    def get_raw_html(
        self,
        horse_id: str,
        date: str | None = None,
        force_netkeiba: bool = False,
    ) -> str:
        return self._horse_pedigree_repository.get_by_id(
            horse_id=horse_id,
            date=date,
            force_netkeiba=force_netkeiba,
        )

    def _get_soup(
        self,
        horse_id: str,
        date: str | None = None,
        force_netkeiba: bool = False,
    ) -> BeautifulSoup:
        html = self.get_raw_html(horse_id=horse_id, date=date, force_netkeiba=force_netkeiba)
        return get_soup(html)

    def get(
        self,
        horse_id: str,
        date: str | None = None,
        force_netkeiba: bool = False,
    ) -> dict[str, str] | None:
        soup = self._get_soup(horse_id=horse_id, date=date, force_netkeiba=force_netkeiba)

        # Find pedigree table
        table: Tag = soup.find("table", class_="blood_table detail")
        if table is None:
            return None

        pedigree: dict[str, str] = {}
        gen_memo: dict[int, str] = {}
        for td in table.find_all("td"):
            gender = extract_gender(td)
            row_span = extract_row_span(td)

            a_tag = td.find("a")
            if a_tag is None:
                logger.info("Skip because no a tag: horse_id = %s", horse_id)
                continue
            url = a_tag.get("href")
            if url is None:
                logger.info("Skip because no href: horse_id = %s", horse_id)
                continue
            ancestor_horse_id_match = re.search(r"horse/([a-zA-Z0-9]+)/", url)
            if ancestor_horse_id_match is None:
                logger.info("Skip because no ancestor horse id: horse_id = %s", horse_id)
                continue
            ancestor_horse_id = ancestor_horse_id_match.group(1)

            # 同世代以前の記録を clear.
            gen_memo = {k: v for k, v in gen_memo.items() if k > row_span}
            gen_memo[row_span] = gender

            key = "_".join([v for k, v in sorted(gen_memo.items(), key=lambda x: -x[-0])])
            pedigree[key] = ancestor_horse_id

        return pedigree
