from time import sleep

from selenium.webdriver import Chrome, ChromeOptions

from horse_racing.core.user_agent import random_choice_user_agent


class ChromeDriver:
    def __init__(self) -> None:
        self._options = ChromeOptions()
        self._options.add_argument("--headless")
        self._options.add_argument("--no-sandbox")
        self._options.add_argument("--disable-gpu")

    def get_page_source(self, url: str) -> str:
        self._options.add_argument(f"--user-agent={random_choice_user_agent()}")
        with Chrome(options=self._options) as driver:
            driver.get(url=url)
            sleep(1.0)
            return str(driver.page_source)
