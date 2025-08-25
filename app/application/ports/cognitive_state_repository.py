from typing import Protocol, List, Optional

from app.domain.entities.cognitive_state import CognitiveState


class CognitiveStateRepository(Protocol):
    def save(self, cognitive_state: CognitiveState) -> CognitiveState:
        ...

    def get_latest_cognitive_state(self) -> Optional[CognitiveState]:
        ...

    def get_cognitive_state_history(self) -> List[CognitiveState]:
        ...
