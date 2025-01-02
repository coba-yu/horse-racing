import re

from horse_racing.core.html import get_html, get_soup


def extract_race_date(href: str) -> str | None:
    date_match = re.search(r"kaisai_date=(\d{8})", href)
    if date_match is None:
        return None
    _, _, race_date = date_match.group().partition("=")
    return race_date


class RaseScheduleUsecase:
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
