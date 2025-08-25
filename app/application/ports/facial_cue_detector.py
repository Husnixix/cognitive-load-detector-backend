from typing import Protocol, Dict


class FacialCueDetector(Protocol):
    def start_facial_cue_detector(self) -> None:
        ...

    def stop_facial_cue_detector(self) -> None:
        ...

    def facial_cue_snap_shot_and_reset(self) -> Dict:
        ...
