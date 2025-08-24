from abc import ABC, abstractmethod

from app.repositary.cognitive_load_entity import CognitiveState

class CognitiveStateRespositary(ABC):
    @abstractmethod
    def save(self, cognitive_state: CognitiveState) -> CognitiveState:
        pass

    @abstractmethod
    def get_latest_cognitive_state(self) -> CognitiveState:
        pass

    @abstractmethod
    def get_cognitive_state_history(self) -> list[CognitiveState]:
        pass

