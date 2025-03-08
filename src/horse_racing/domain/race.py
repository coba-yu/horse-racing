from dataclasses import dataclass


@dataclass
class RaceInfo:
    race_number: int
    start_hour: int
    distance: str
    weather: str
    field_condition: str
