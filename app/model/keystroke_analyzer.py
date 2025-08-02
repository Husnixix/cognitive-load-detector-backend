import time
from pynput import keyboard


class KeystrokeAnalyzer:
    def __init__(self):
        self.keystrokes = []
        self.mistakes = []
        self.typing_speed = 0
        self.error_rate = 0
        self.pause_rate = 0
        self.start_time = 0
        self.end_time = 0


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
            self.typing_speed = 0
            return self.typing_speed

        duration = self.end_time - self.start_time
        if duration == 0:
            self.typing_speed = 0
        else:
            self.typing_speed = round(len(self.keystrokes) / duration, 2)
        return self.typing_speed

    def calculate_error_rate(self):
        if len(self.keystrokes) == 0:
            self.error_rate = 0
            return self.error_rate

        total_keys = len(self.keystrokes)
        total_errors = len(self.mistakes)

        self.error_rate = round((total_errors / total_keys) * 100, 2)
        return self.error_rate

    def calculate_pause_rate(self):
        if len(self.keystrokes) < 2:
            self.pause_rate = 0
            return self.pause_rate

        total_gap = 0
        for i in range(1, len(self.keystrokes)):
            total_gap += self.keystrokes[i][1] - self.keystrokes[i - 1][1]

        self.pause_rate = round(total_gap / (len(self.keystrokes) - 1), 2)
        return self.pause_rate

