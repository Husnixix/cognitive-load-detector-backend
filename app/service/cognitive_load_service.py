from app.model.detectors.timer import Timer
from app.model.facial_cue_analyzer import FacialCueAnalyzer
from app.model.keystroke_analyzer import KeystrokeAnalyzer
from app.model.cognitive_load_anayzler import CognitiveLoadAnalyzer
import time
import threading
from datetime import datetime
import threading


class CognitiveLoadService:
    def __init__(self):
        self.facial_cue_analyzer = FacialCueAnalyzer()
        self.keystroke_analyzer = KeystrokeAnalyzer()
        self.cognitive_load_analyzer = CognitiveLoadAnalyzer()
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
        self.snapshot_thread = threading.Thread(target=self.snapshot_thread, daemon=True)
        self.snapshot_thread.start()

    def snapshot_loop(self):
        while self.is_detecting:
            time.sleep(self.interval)
            start_time = self.timer.start_time
            end_time = self.timer.end_time

            facial_data = self.facial_cue_analyzer.facial_cue_snap_shot_and_reset()
            keystroke_data = self.keystroke_analyzer.keystroke_snap_shot_and_reset()

            cognitive_score = self.cognitive_load_analyzer.score_feature(facial_data, keystroke_data)
            cognitive_status = self.cognitive_load_analyzer.give_label(cognitive_score)

            print(f"\n[SNAPSHOT] {start_time} → {end_time}")
            print("Facial:", facial_data)
            print("Keystrokes:", keystroke_data)
            print("Score:", cognitive_score, "Status:", cognitive_status)

    def stop_detecting_cognitive_load(self):
        self.is_detecting = False
        self.timer.stop_timer()

        start_time = self.timer.start_time
        end_time = self.timer.end_time

        facial_data = self.facial_cue_analyzer.facial_cue_snap_shot_and_reset()
        keystroke_data = self.keystroke_analyzer.keystroke_snap_shot_and_reset()

        cognitive_score = self.cognitive_load_analyzer.score_feature(facial_data, keystroke_data)
        cognitive_status = self.cognitive_load_analyzer.give_label(cognitive_score)

        self.facial_cue_analyzer.stop_facial_cue_detector()
        self.keystroke_analyzer.stop_keystroke_tracker()

        print(f"\n[FINAL SNAPSHOT] {start_time} → {end_time}")
        print("Facial:", facial_data)
        print("Keystrokes:", keystroke_data)
        print("Score:", cognitive_score, "Status:", cognitive_status)


