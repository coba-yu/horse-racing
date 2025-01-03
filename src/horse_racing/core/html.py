import os
import random
from functools import wraps
from pathlib import Path
from time import sleep
from typing import Callable, Concatenate, ParamSpec

import requests
from bs4 import BeautifulSoup

from horse_racing.core.logging import logger

P = ParamSpec("P")

DEFAULT_SLEEP_SECONDS = 10.0
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:115.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 OPR/85.0.4341.72",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 OPR/85.0.4341.72",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Vivaldi/5.3.2679.55",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Vivaldi/5.3.2679.55",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Brave/1.40.107",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Brave/1.40.107",
]


def random_choice_user_agent() -> str:
    return random.choice(USER_AGENTS)


def make_cache_dir(sub_dir: str) -> str:
    tmp_dir = os.path.join("data", "cache", "html", sub_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


def get_html_with_cache(
    func: Callable[Concatenate[str, P], str],
) -> Callable[..., str]:
    @wraps(func)
    def wrapper(
        url: str,
        *args: P.args,
        sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
        cache_sub_path: str | Path | None = None,
        **kwds: P.kwargs,
    ) -> str:
        if cache_sub_path is not None:
            cache_sub_dir = os.path.dirname(cache_sub_path)
            cache_dir = make_cache_dir(sub_dir=cache_sub_dir)

            cache_file_name = os.path.basename(cache_sub_path)
            cache_path = os.path.join(cache_dir, cache_file_name)
            logger.info(f"Try to get cache: {cache_path}")
            if os.path.isfile(cache_path):
                logger.info("Cache found")
                with open(cache_path, "r") as f:
                    return f.read()
            logger.info("Cache not found")

        html = func(url, *args, **kwds)
        sleep(sleep_seconds)

        if cache_sub_dir is not None:
            logger.info(f"Save cache: {cache_path}")
            with open(cache_path, "w") as f:
                f.write(html)

        return html

    return wrapper


@get_html_with_cache
def get_html(url: str) -> str:
    headers = {"user-agent": random_choice_user_agent()}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()

    # workaround
    r.encoding = "EUC-JP"
    return r.text


def get_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")
