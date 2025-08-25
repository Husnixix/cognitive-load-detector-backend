from typing import Dict

from app.application.ports.facial_cue_detector import FacialCueDetector
from app.infrastructure.capture.facial_cue_analyzer import FacialCueAnalyzer


class OpenCVFacialCueDetector(FacialCueDetector):
    def __init__(self):
        self._impl = FacialCueAnalyzer()

    def start_facial_cue_detector(self) -> None:
        self._impl.start_facial_cue_detector()

    def stop_facial_cue_detector(self) -> None:
        self._impl.stop_facial_cue_detector()

    def facial_cue_snap_shot_and_reset(self) -> Dict:
        return self._impl.facial_cue_snap_shot_and_reset()
