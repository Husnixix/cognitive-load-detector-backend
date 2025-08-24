
from app.config.configure_database import ConfigureDatabase
from app.service.cognitive_load_service import CognitiveLoadService
import time



def main():

    cognitive_detector = CognitiveLoadService()
    print("Starting cognitive load detector... ")
    cognitive_detector.start_detecting_cognitive_load()
    try:
        time.sleep(90)
    except KeyboardInterrupt:
        print("Stopping cognitive load detector...")


    cognitive_detector.stop_detecting_cognitive_load()
    print("Finished cognitive load detector...")

if __name__ == "__main__":
    main()