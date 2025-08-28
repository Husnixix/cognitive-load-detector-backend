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

def cognitive_state_to_dict(state: CognitiveState) -> Optional[dict]:
    if state is None:
        return None
    return {
        "id": state.id,
        "start_time": state.start_time.isoformat() if hasattr(state.start_time, "isoformat") else state.start_time,
        "end_time": state.end_time.isoformat() if hasattr(state.end_time, "isoformat") else state.end_time,
        "facial_cue_data": state.facial_cue_data,
        "keystroke_data": state.keystroke_data,
        "cognitive_state_data": state.cognitive_state_data,
    }
