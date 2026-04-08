from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class SupportAction(Action):
    action_type: Literal["classify", "search_kb", "respond", "escalate"] = Field(
        ..., description="Action category taken by the agent"
    )
    content: str = Field(default="", description="Action payload text")


class SupportObservation(Observation):
    ticket_id: int = Field(..., description="Current ticket identifier")
    task_id: str = Field(..., description="Current task id")
    difficulty: Literal["easy", "medium", "hard"] = Field(...)
    query: str = Field(..., description="Customer support query")
    kb_id: str = Field(..., description="Expected knowledge-base identifier")
    requires_escalation: bool = Field(..., description="Whether escalation is required")
    history: List[Dict[str, Any]] = Field(default_factory=list)
    feedback: str = Field(default="", description="Reward-shaping feedback")


class SupportState(State):
    difficulty_filter: Optional[str] = Field(default=None)
    current_index: int = Field(default=0)
    task_id: str = Field(default="")
    score_so_far: float = Field(default=0.0)
    final_score: float = Field(default=0.0)
    done: bool = Field(default=False)