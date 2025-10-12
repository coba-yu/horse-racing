from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=+9), "JST")


def get_current() -> datetime:
    return datetime.now(tz=JST)


def get_current_yyyymmdd_hhmmss() -> str:
    return get_current().strftime("%Y%m%d_%H%M%S")


def get_current_timestamp() -> int:
    return int(get_current().timestamp())
