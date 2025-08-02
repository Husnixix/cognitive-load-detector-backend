import time
from datetime import datetime
from app.model.keystroke_analyzer import KeystrokeAnalyzer

analyzer = KeystrokeAnalyzer()
analyzer.start_tracker()
print(f"Keystroke analyzer started at {datetime.fromtimestamp(analyzer.start_time).strftime('%Y-%m-%d %H:%M:%S')}")
time.sleep(60)
analyzer.stop_tracker()
print(f"Keystroke analyzer ended at {datetime.fromtimestamp(analyzer.end_time).strftime('%Y-%m-%d %H:%M:%S')}")

print(f"Typing Speed: {analyzer.typing_speed} keys/sec")
print(f"Error Rate: {analyzer.error_rate}% Backspace usage")
print(f"Pause Rate: {analyzer.pause_rate} sec between keys")
