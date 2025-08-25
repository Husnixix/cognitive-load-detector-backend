import datetime
from dataclasses import dataclass
from typing import Optional


@dataclass
class CognitiveState:
    start_time: datetime.datetime
    end_time: datetime.datetime
    facial_cue_data: dict
    keystroke_data: dict
    cognitive_state_data: dict
    id: Optional[str] = None
