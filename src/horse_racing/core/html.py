import requests
from bs4 import BeautifulSoup


def get_html(url: str) -> str:
    headers = {"user-agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()

    # workaround
    r.encoding = "EUC-JP"
    return r.text


def get_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")
