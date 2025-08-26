import logging
import time
import threading

from app.domain.entities.cognitive_state import CognitiveState
from app.domain.services.cognitive_load_analyzer import CognitiveLoadAnalyzer
from app.application.ports.facial_cue_detector import FacialCueDetector
from app.application.ports.keystroke_tracker import KeystrokeTracker
from app.application.ports.cognitive_state_repository import CognitiveStateRepository
from app.application.ports.clock import Clock


class CognitiveLoadService:
    def __init__(
        self,
        repository: CognitiveStateRepository,
        facial_detector: FacialCueDetector,
        keystroke_tracker: KeystrokeTracker,
        analyzer: CognitiveLoadAnalyzer,
        clock: Clock,
        interval: int = 60,
    ):
        self.logger = logging.getLogger(__name__)
        self.facial_cue_analyzer = facial_detector
        self.keystroke_analyzer = keystroke_tracker
        self.cognitive_load_analyzer = analyzer
        self.repository = repository
        self.clock = clock
        self.is_detecting = False
        self.interval = interval
        self.snapshot_thread = None
        self._facial_thread = None
        self._keystroke_thread = None
        self._stop_event = threading.Event()

    def start_detecting_cognitive_load(self):
        if self.is_detecting:
            return
        self.is_detecting = True

        self.clock.start()
        self._stop_event.clear()

        self._facial_thread = threading.Thread(target=self.facial_cue_analyzer.start_facial_cue_detector, daemon=True)
        self._facial_thread.start()
        self._keystroke_thread = threading.Thread(target=self.keystroke_analyzer.start_keystroke_tracker, daemon=True)
        self._keystroke_thread.start()
        self.snapshot_thread = threading.Thread(target=self.snapshot_loop, daemon=True)
        self.snapshot_thread.start()

    def snapshot_loop(self):
        while self.is_detecting:
            # Wait but allow early wake-up when stopping
            if self._stop_event.wait(self.interval):
                break
            self.clock.stop()
            start_time = self.clock.start_time
            end_time = self.clock.end_time

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
            self.repository.save(cognitive_state)
            self.logger.info("Database done")
            self.logger.info(f"[SNAPSHOT] {start_time} → {end_time}")
            self.logger.info(f"Facial: {facial_data}")
            self.logger.info(f"Keystrokes: {keystroke_data}")
            self.logger.info(f"Status: {cognitive_status}")
            # Start the next interval
            self.clock.start()

    def stop_detecting_cognitive_load(self):
        # Signal loops to stop and wake sleepers
        self.is_detecting = False
        self._stop_event.set()
        self.clock.stop()

        self.facial_cue_analyzer.stop_facial_cue_detector()
        self.keystroke_analyzer.stop_keystroke_tracker()

        # Give worker threads a brief chance to cleanup resources
        if self._facial_thread and self._facial_thread.is_alive():
            self._facial_thread.join(timeout=2.0)
        if self._keystroke_thread and self._keystroke_thread.is_alive():
            self._keystroke_thread.join(timeout=2.0)
        if self.snapshot_thread and self.snapshot_thread.is_alive():
            self.snapshot_thread.join(timeout=1.0)
        self._facial_thread = None
        self._keystroke_thread = None
        self.snapshot_thread = None
        # Final snapshot intentionally removed for fast stop

    def get_latest(self):
        """Read-only accessor for latest cognitive state."""
        return self.repository.get_latest_cognitive_state()

    def get_history(self):
        """Read-only accessor for full cognitive state history (most recent first)."""
        return self.repository.get_cognitive_state_history()

