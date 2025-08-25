from typing import Optional
from app.domain.entities.cognitive_state import CognitiveState


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
