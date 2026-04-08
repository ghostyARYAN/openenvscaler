from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import SupportAction, SupportObservation, SupportState


class CustomerSupportEnvClient(EnvClient[SupportAction, SupportObservation, SupportState]):
    def _step_payload(self, action: SupportAction) -> Dict:
        return {
            "action_type": action.action_type,
            "content": action.content,
            "metadata": action.metadata,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SupportObservation]:
        obs_data = payload.get("observation", {})
        observation = SupportObservation(
            ticket_id=obs_data.get("ticket_id", 0),
            task_id=obs_data.get("task_id", ""),
            difficulty=obs_data.get("difficulty", "easy"),
            query=obs_data.get("query", ""),
            kb_id=obs_data.get("kb_id", ""),
            requires_escalation=obs_data.get("requires_escalation", False),
            history=obs_data.get("history", []),
            feedback=obs_data.get("feedback", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> SupportState:
        return SupportState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            difficulty_filter=payload.get("difficulty_filter"),
            current_index=payload.get("current_index", 0),
            task_id=payload.get("task_id", ""),
            score_so_far=payload.get("score_so_far", 0.0),
            final_score=payload.get("final_score", 0.0),
            done=payload.get("done", False),
        )