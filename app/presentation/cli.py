import time
from dotenv import load_dotenv, find_dotenv
from app.application.cognitive_load_service import CognitiveLoadService


def main():
    load_dotenv(find_dotenv())
    service = CognitiveLoadService()
    print("Starting cognitive load detector... ")
    service.start_detection()
    try:
        time.sleep(90)
    except KeyboardInterrupt:
        print("Stopping cognitive load detector...")
    service.stop_detection()
    print("Finished cognitive load detector...")

if __name__ == "__main__":
    main()