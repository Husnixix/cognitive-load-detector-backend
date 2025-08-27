from typing import Optional, List
from app.infrastructure.database.ConfigureDatabase import ConfigureDatabase
from app.infrastructure.entities.cognitive_load_entity import CognitiveState

class CognitiveLoadRepository:
    def __init__(self):
        self.collection = ConfigureDatabase().get_cognitive_states_collection("cognitive_states")

    def save(self, cognitive_state: CognitiveState) -> CognitiveState:
        document = {
            "start_time": cognitive_state.start_time,
            "end_time": cognitive_state.end_time,
            "facial_cue_data": cognitive_state.facial_cue_data,
            "keystroke_data": cognitive_state.keystroke_data,
            "cognitive_state_data": cognitive_state.cognitive_state_data,
        }
        result = self.collection.insert_one(document)
        cognitive_state.id = str(result.inserted_id)
        return cognitive_state

    def get_latest_cognitive_state(self) -> Optional[CognitiveState]:
        document = self.collection.find_one(sort=[("_id", -1)])
        if document:
            return CognitiveState(
                id=str(document["_id"]),
                start_time=document["start_time"],
                end_time=document["end_time"],
                facial_cue_data=document["facial_cue_data"],
                keystroke_data=document["keystroke_data"],
                cognitive_state_data=document["cognitive_state_data"],
            )
        return None

    def get_cognitive_state_history(self) -> List[CognitiveState]:
        documents = self.collection.find().sort("_id", -1)
        return [
            CognitiveState(
                id=str(document["_id"]),
                start_time=document["start_time"],
                end_time=document["end_time"],
                facial_cue_data=document["facial_cue_data"],
                keystroke_data=document["keystroke_data"],
                cognitive_state_data=document["cognitive_state_data"],
            )
            for document in documents
        ]