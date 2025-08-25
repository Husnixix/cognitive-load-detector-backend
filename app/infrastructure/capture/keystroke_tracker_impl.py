from typing import Dict

from app.application.ports.keystroke_tracker import KeystrokeTracker
from app.infrastructure.capture.keystroke_analyzer import KeystrokeAnalyzer


class SystemKeystrokeTracker(KeystrokeTracker):
    def __init__(self):
        self._impl = KeystrokeAnalyzer()

    def start_keystroke_tracker(self) -> None:
        self._impl.start_keystroke_tracker()

    def stop_keystroke_tracker(self) -> Dict:
        return self._impl.stop_keystroke_tracker()

    def keystroke_snap_shot_and_reset(self) -> Dict:
        return self._impl.keystroke_snap_shot_and_reset()
