from time import sleep

import tenacity
from selenium.webdriver import Chrome, ChromeOptions

from horse_racing.core.html import DEFAULT_SLEEP_SECONDS, random_choice_user_agent


class ChromeDriver:
    def __init__(self, sleep_seconds: float = DEFAULT_SLEEP_SECONDS) -> None:
        self._options = ChromeOptions()
        self._options.add_argument("--headless")
        self._options.add_argument("--no-sandbox")
        self._options.add_argument("--disable-gpu")
        self._options.add_argument("--disable-dev-shm-usage")

        self._sleep_seconds = sleep_seconds

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(10),
        wait=tenacity.wait_exponential_jitter(initial=1.0, max=60.0),
    )  # type: ignore[misc]
    def get_page_source(self, url: str, skip_sleep: bool = False) -> str:
        self._options.add_argument(f"--user-agent={random_choice_user_agent()}")

        with Chrome(options=self._options) as driver:
            driver.get(url=url)
            if not skip_sleep:
                sleep(self._sleep_seconds)
            return str(driver.page_source)
