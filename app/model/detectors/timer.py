import time

from datetime import datetime, timedelta


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start_timer(self):
        self.start_time = datetime.now()
        return self.start_time.strftime("%Y-%m-%d %H:%M:%S")

    def stop_timer(self):
        self.end_time = datetime.now()
        return self.start_time.strftime("%Y-%m-%d %H:%M:%S")

