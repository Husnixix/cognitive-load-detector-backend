from typing import Protocol, Dict


class KeystrokeTracker(Protocol):
    def start_keystroke_tracker(self) -> None:
        ...

    def stop_keystroke_tracker(self) -> Dict:
        ...

    def keystroke_snap_shot_and_reset(self) -> Dict:
        ...
