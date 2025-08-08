import time
from pynput import keyboard


class KeystrokeAnalyzer:
    def __init__(self):
        self.keystrokes = []
        self.mistakes = []
        self.start_time = 0
        self.end_time = 0
        self.listener = None
        self.keystroke_data = {
            "typing_speed": 0,
            "error_rate": 0,
            "pause_rate": 0,
        }

    def start_tracker(self):
        self.start_time = time.time()

        def on_press(key):
            try:
                key_name = key.char
            except AttributeError:
                key_name = str(key)
            self.record_key(key_name, time.time())

        self.listener = keyboard.Listener(on_press=on_press)
        self.listener.start()

    def stop_tracker(self):
        if self.listener:
            self.listener.stop()
        self.end_time = time.time()

        self.calculate_typing_speed()
        self.calculate_error_rate()
        self.calculate_pause_rate()

    def record_key(self, key, timestamp):
        self.keystrokes.append((key, timestamp))
        if key == 'Key.backspace':
            self.mistakes.append(timestamp)

    def calculate_typing_speed(self):
        if len(self.keystrokes) < 2:
            self.keystroke_data["typing_speed"] = 0
            return self.keystroke_data["typing_speed"]

        duration = self.end_time - self.start_time
        if duration == 0:
            self.keystroke_data["typing_speed"] = 0
        else:
            self.keystroke_data["typing_speed"] = round(len(self.keystrokes) / duration, 2)
        return self.keystroke_data["typing_speed"]

    def calculate_error_rate(self):
        if len(self.keystrokes) == 0:
            self.keystroke_data["error_rate"] = 0
            return self.keystroke_data["error_rate"]

        total_keys = len(self.keystrokes)
        total_errors = len(self.mistakes)

        self.keystroke_data["error_rate"] = round((total_errors / total_keys) * 100, 2)
        return self.keystroke_data["error_rate"]

    def calculate_pause_rate(self):
        if len(self.keystrokes) < 2:
            self.keystroke_data["pause_rate"] = 0
            return self.keystroke_data["pause_rate"]

        total_gap = 0
        for i in range(1, len(self.keystrokes)):
            total_gap += self.keystrokes[i][1] - self.keystrokes[i - 1][1]

        self.keystroke_data["pause_rate"] = round(total_gap / (len(self.keystrokes) - 1), 2)
        return self.keystroke_data["pause_rate"]

