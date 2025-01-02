import os.path
import re

from horse_racing.core.chrome import ChromeDriver
from horse_racing.core.html import get_html, get_soup


def extract_race_date(href: str) -> str | None:
    date_match = re.search(r"kaisai_date=(\d{8})", href)
    if date_match is None:
        return None
    _, _, race_date = date_match.group().partition("=")
    return race_date


class RaceScheduleUsecase:
    def __init__(self, driver: ChromeDriver) -> None:
        self.driver = driver

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
        tmp_dir = os.path.join("data", "tmp", "html", "race_list")
        os.makedirs(tmp_dir, exist_ok=True)

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
