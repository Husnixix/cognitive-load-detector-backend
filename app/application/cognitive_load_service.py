from app.domain.cognitive_load_detector import CognitiveLoadDetector

class CognitiveLoadService:
    def __init__(self):
        self.cognitive_load_detector = CognitiveLoadDetector()

    def start_detection(self):
        self.cognitive_load_detector.start_detectors()

    def stop_detection(self):
        self.cognitive_load_detector.stop_detectors()

    def get_latest_cognitive_state(self):
        return self.cognitive_load_detector.get_latest_state()

    def get_cognitive_state_history(self):
        return self.cognitive_load_detector.get_cognitive_state_history()