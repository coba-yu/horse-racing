from time import sleep

from selenium.webdriver import Chrome, ChromeOptions


class ChromeDriver:
    def __init__(self) -> None:
        self.options = ChromeOptions()
        self.options.add_argument("--headless")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-gpu")
        self.options.add_argument("--user-agent=Mozilla/5.0")

    def get_page_source(self, url: str) -> str:
        with Chrome(options=self.options) as driver:
            driver.get(url=url)
            sleep(1.0)
            return str(driver.page_source)
