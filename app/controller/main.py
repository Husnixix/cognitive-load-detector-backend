import time

from app.application.services.cognitive_load_service import CognitiveLoadService
from app.domain.services.cognitive_load_analyzer import CognitiveLoadAnalyzer
from app.infrastructure.repositories.mongo_cognitive_state_repository import MongoCognitiveStateRepository
from app.infrastructure.capture.opencv_facial_cue_detector import OpenCVFacialCueDetector
from app.infrastructure.capture.keystroke_tracker_impl import SystemKeystrokeTracker
from app.infrastructure.time.system_clock import SystemClock
from dotenv import load_dotenv, find_dotenv


def main():
    load_dotenv(find_dotenv())
    service = CognitiveLoadService(
        repository=MongoCognitiveStateRepository(),
        facial_detector=OpenCVFacialCueDetector(),
        keystroke_tracker=SystemKeystrokeTracker(),
        analyzer=CognitiveLoadAnalyzer(),
        clock=SystemClock(),
        interval=60,
    )
    print("Starting cognitive load detector... ")
    service.start_detecting_cognitive_load()
    try:
        time.sleep(90)
    except KeyboardInterrupt:
        print("Stopping cognitive load detector...")
    service.stop_detecting_cognitive_load()
    print("Finished cognitive load detector...")

if __name__ == "__main__":
    main()