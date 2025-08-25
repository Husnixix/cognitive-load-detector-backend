import logging
import time
import threading


from app.model.detectors.timer import Timer
from app.model.facial_cue_analyzer import FacialCueAnalyzer
from app.model.keystroke_analyzer import KeystrokeAnalyzer
from app.model.cognitive_load_analyzer import CognitiveLoadAnalyzer

from app.domain.entities.cognitive_state import CognitiveState
from app.infrastructure.repositories.cognitive_state_repository import MongoCognitiveStateRepository


class CognitiveLoadService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.facial_cue_analyzer = FacialCueAnalyzer()
        self.keystroke_analyzer = KeystrokeAnalyzer()
        self.cognitive_load_analyzer = CognitiveLoadAnalyzer()
        self.respositary = MongoCognitiveStateRepository()
        self.timer = Timer()
        self.is_detecting = False
        self.interval = 60
        self.snapshot_thread = None

    def start_detecting_cognitive_load(self):
        if self.is_detecting:
            return
        self.is_detecting = True

        self.timer.start_timer()

        threading.Thread(target=self.facial_cue_analyzer.start_facial_cue_detector, daemon=True).start()
        threading.Thread(target=self.keystroke_analyzer.start_keystroke_tracker, daemon=True).start()
        self.snapshot_thread = threading.Thread(target=self.snapshot_loop, daemon=True)
        self.snapshot_thread.start()

    def snapshot_loop(self):
        while self.is_detecting:
            time.sleep(self.interval)
            self.timer.stop_timer()
            start_time = self.timer.start_time
            end_time = self.timer.end_time

            facial_data = self.facial_cue_analyzer.facial_cue_snap_shot_and_reset()
            keystroke_data = self.keystroke_analyzer.keystroke_snap_shot_and_reset()

            cognitive_score = self.cognitive_load_analyzer.score_feature(facial_data, keystroke_data)
            cognitive_status = self.cognitive_load_analyzer.get_score_and_label(cognitive_score)

            cognitive_state = CognitiveState(
                start_time=start_time,
                end_time=end_time,
                facial_cue_data=facial_data,
                keystroke_data=keystroke_data,
                cognitive_state_data=cognitive_status,
            )
            self.respositary.save(cognitive_state)
            self.logger.info("Database done")
            self.logger.info(f"[SNAPSHOT] {start_time} → {end_time}")
            self.logger.info(f"Facial: {facial_data}")
            self.logger.info(f"Keystrokes: {keystroke_data}")
            self.logger.info(f"Status: {cognitive_status}")

    def stop_detecting_cognitive_load(self):
        self.is_detecting = False
        self.timer.stop_timer()

        start_time = self.timer.start_time
        end_time = self.timer.end_time

        facial_data = self.facial_cue_analyzer.facial_cue_snap_shot_and_reset()
        keystroke_data = self.keystroke_analyzer.keystroke_snap_shot_and_reset()

        cognitive_score = self.cognitive_load_analyzer.score_feature(facial_data, keystroke_data)
        cognitive_status = self.cognitive_load_analyzer.get_score_and_label(cognitive_score)

        self.facial_cue_analyzer.stop_facial_cue_detector()
        self.keystroke_analyzer.stop_keystroke_tracker()

        cognitive_state = CognitiveState(
            start_time=start_time,
            end_time=end_time,
            facial_cue_data=facial_data,
            keystroke_data=keystroke_data,
            cognitive_state_data=cognitive_status,
        )
        self.respositary.save(cognitive_state)
        self.logger.info("Database done")

        self.logger.info(f"[FINAL SNAPSHOT] {start_time} → {end_time}")
        self.logger.info(f"Facial: {facial_data}")
        self.logger.info(f"Keystrokes: {keystroke_data}")
        self.logger.info(f"Status: {cognitive_status}")

    def get_latest(self):
        """Read-only accessor for latest cognitive state."""
        return self.respositary.get_latest_cognitive_state()

    def get_history(self):
        """Read-only accessor for full cognitive state history (most recent first)."""
        return self.respositary.get_cognitive_state_history()

