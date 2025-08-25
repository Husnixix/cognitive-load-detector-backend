import datetime

from app.application.ports.clock import Clock
from app.infrastructure.capture.detectors.timer import Timer


class SystemClock(Clock):
    def __init__(self):
        self._timer = Timer()

    @property
    def start_time(self) -> datetime.datetime:
        return self._timer.start_time

    @property
    def end_time(self) -> datetime.datetime:
        return self._timer.end_time

    def start(self) -> None:
        self._timer.start_timer()

    def stop(self) -> None:
        self._timer.stop_timer()
