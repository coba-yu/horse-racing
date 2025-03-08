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

    def _get_page_source(self, url: str) -> str:
        if self.driver is None:
            raise ValueError("driver is not set")
        return self.driver.get_page_source(url=url)

    def get_race_info(self, race_id: str) -> RaceInfo:
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        html = self._get_page_source(url=url)
        soup = get_soup(html)

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
