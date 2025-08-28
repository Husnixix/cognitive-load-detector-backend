import threading
from datetime import datetime
from app.domain.detectors.facial_cue_detector import FacialCueDetector
from app.domain.detectors.keystroke_detector import KeystrokeDetector
from app.domain.algorithm.cognitive_load_algorithm import CognitiveLoadAlgorithm
from app.infrastructure.entities.cognitive_load_entity import CognitiveState
from app.infrastructure.repository.cognitive_load_respository import CognitiveLoadRepository

class CognitiveLoadDetector:
    def __init__(self):
        self.facial_cue_detector = FacialCueDetector()
        self.keystroke_detector = KeystrokeDetector()
        self.algorithm = CognitiveLoadAlgorithm()
        self.repository = CognitiveLoadRepository()
        self.start_time = None
        self.end_time = None
        self.interval = 60
        self.is_detecting = False
        self._facial_cue_thread = None
        self._keystroke_thread = None
        self._detection_thread = None
        self._stop_event = threading.Event()

    def start_detectors(self):
        if self.is_detecting:
            return
        self.is_detecting = True
        self._stop_event.clear()
        self.start_time = datetime.now()
        
        self._facial_cue_thread = threading.Thread(target=self.facial_cue_detector.start_facial_cue_detector,
                                                   daemon=True)
        self._facial_cue_thread.start()
        self._keystroke_thread = threading.Thread(target=self.keystroke_detector.start_keystroke_tracker,
                                                   daemon=True)
        self._keystroke_thread.start()
        self._detection_thread = threading.Thread(target=self.detect_state, daemon=True)
        self._detection_thread.start()

    def detect_state(self):
        while self.is_detecting:
            if self._stop_event.wait(self.interval):
                break
            end_time = datetime.now()
            start_time = self.start_time or end_time

            facial_cue_data = self.facial_cue_detector.facial_cue_snap_shot_and_reset()
            keystroke_data = self.keystroke_detector.keystroke_snap_shot_and_reset()
            cognitive_score = self.algorithm.score_feature(facial_cue_data, keystroke_data)
            cognitive_state_data = self.algorithm.get_score_and_label(cognitive_score)

            print(facial_cue_data)
            print(keystroke_data)
            print(cognitive_state_data)
            cognitive_state = CognitiveState(
                start_time=start_time,
                end_time=end_time,
                facial_cue_data=facial_cue_data,
                keystroke_data=keystroke_data,
                cognitive_state_data=cognitive_state_data,
            )
            self.repository.save(cognitive_state)
            self.start_time = end_time

    def stop_detectors(self):
        self.is_detecting = False
        self._stop_event.set()

        self.facial_cue_detector.stop_facial_cue_detector()
        self.keystroke_detector.stop_keystroke_tracker()

        if self._facial_cue_thread and self._facial_cue_thread.is_alive():
            self._facial_cue_thread.join(timeout=2.0)
        if self._keystroke_thread and self._keystroke_thread.is_alive():
            self._keystroke_thread.join(timeout=2.0)
        if self._detection_thread and self._detection_thread.is_alive():
            self._detection_thread.join(timeout=2.0)

        self._facial_cue_thread = None
        self._keystroke_thread = None
        self._detection_thread = None

    def get_latest_state(self):
        return self.repository.get_latest_cognitive_state()

    def get_cognitive_state_history(self):
        return self.repository.get_cognitive_state_history()