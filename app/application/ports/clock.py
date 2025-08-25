from typing import Protocol
import datetime


class Clock(Protocol):
    start_time: datetime.datetime
    end_time: datetime.datetime

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...
