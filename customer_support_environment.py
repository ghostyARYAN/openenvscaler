from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from data_loader import build_expected, load_dataset, split_difficulty
from kb import build_knowledge_base
from models import SupportAction, SupportObservation, SupportState
from tasks import grade_task, task_for_difficulty


class CustomerSupportEnvironment(Environment[SupportAction, SupportObservation, SupportState]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, csv_path: str = "dataset.csv", max_steps: int = 6):
        super().__init__()
        self.df = load_dataset(csv_path)
        self.splits = split_difficulty(self.df)
        self.kb = build_knowledge_base(self.df)
        self.max_steps = max_steps

        self._episodes = self.df.reset_index(drop=True)
        self._cursor = 0
        self._expected: dict[str, Any] | None = None
        self._history: list[dict[str, Any]] = []
        self._difficulty_filter: Optional[str] = None
        self._final_score = 0.0
        self._done = False
        self._task_id = ""
        self._state = SupportState(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        index: Optional[int] = None,
        **kwargs: Any,
    ) -> SupportObservation:
        del seed, kwargs
        if difficulty is not None:
            normalized = difficulty.strip().lower()
            if normalized not in self.splits:
                raise ValueError(f"Unknown difficulty: {difficulty}")
            self._difficulty_filter = normalized
            self._episodes = self.splits[normalized].reset_index(drop=True)
        elif self._difficulty_filter is None:
            self._episodes = self.df.reset_index(drop=True)

        if len(self._episodes) == 0:
            raise ValueError("No episodes found for the requested filter")

        if index is None:
            self._cursor = self._cursor % len(self._episodes)
        else:
            self._cursor = int(index) % len(self._episodes)

        row = self._episodes.iloc[self._cursor]
        self._cursor = (self._cursor + 1) % len(self._episodes)
        self._expected = build_expected(row)
        self._task_id = task_for_difficulty(self._expected["difficulty"]).task_id
        self._history = []
        self._done = False
        self._final_score = 0.0

        self._state = SupportState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            difficulty_filter=self._difficulty_filter,
            current_index=self._cursor,
            task_id=self._task_id,
            score_so_far=0.0,
            final_score=0.0,
            done=False,
        )

        return self._make_observation(reward=0.0, done=False, feedback="Environment reset")

    def step(
        self,
        action: SupportAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SupportObservation:
        del timeout_s, kwargs
        if self._expected is None:
            # HTTP mode can be stateless across requests; initialize lazily.
            self.reset()
        if self._done:
            return self._make_observation(reward=0.0, done=True, feedback="Episode already done")

        self._state.step_count += 1
        reward = -0.01 * self._state.step_count
        feedback_parts: list[str] = ["time_penalty"]

        if action.action_type == "classify":
            if _norm(action.content) == _norm(self._expected["expected_category"]):
                reward += 0.3
                feedback_parts.append("classification_correct")
            else:
                reward -= 0.1
                feedback_parts.append("classification_incorrect")

        elif action.action_type == "search_kb":
            if _norm(action.content) == _norm(self._expected["kb_id"]):
                reward += 0.2
                feedback_parts.append("kb_match")
            else:
                reward -= 0.05
                feedback_parts.append("kb_mismatch")

        elif action.action_type == "respond":
            similarity = SequenceMatcher(
                None,
                _norm(action.content),
                _norm(self._expected["expected_response"]),
            ).ratio()
            reward += 0.45 * similarity
            if _is_polite(action.content):
                reward += 0.1
                feedback_parts.append("polite")
            if similarity < 0.25:
                reward -= 0.2
                feedback_parts.append("hallucination_risk")
            self._done = True
            feedback_parts.append("terminal_respond")

        elif action.action_type == "escalate":
            if bool(self._expected["requires_escalation"]):
                reward += 0.3
                feedback_parts.append("escalation_correct")
            else:
                reward -= 0.2
                feedback_parts.append("unnecessary_escalation")
            self._done = True
            feedback_parts.append("terminal_escalate")

        if self._state.step_count >= self.max_steps:
            self._done = True
            feedback_parts.append("max_steps")

        event = {
            "step": self._state.step_count,
            "action_type": action.action_type,
            "content": action.content,
            "reward": round(reward, 4),
        }
        self._history.append(event)

        if self._done:
            self._final_score = grade_task(self._task_id, self._expected, self._history)
            self._state.final_score = self._final_score

        self._state.score_so_far = max(0.0, min(1.0, self._state.score_so_far + max(reward, 0.0) / 2.0))
        self._state.done = self._done

        return self._make_observation(
            reward=round(reward, 4),
            done=self._done,
            feedback=",".join(feedback_parts),
        )

    @property
    def state(self) -> SupportState:
        return self._state

    def _make_observation(self, reward: float, done: bool, feedback: str) -> SupportObservation:
        if self._expected is None:
            raise RuntimeError("Environment is not initialized")
        return SupportObservation(
            ticket_id=self._expected["ticket_id"],
            task_id=self._task_id,
            difficulty=self._expected["difficulty"],
            query=self._expected["query"],
            kb_id=self._expected["kb_id"],
            requires_escalation=self._expected["requires_escalation"],
            history=list(self._history),
            done=done,
            reward=reward,
            feedback=feedback,
            metadata={
                "expected_category": self._expected["expected_category"],
                "expected_action": self._expected["expected_action"],
                "expected_response": self._expected["expected_response"],
                "kb_id": self._expected["kb_id"],
                "requires_escalation": self._expected["requires_escalation"],
                "final_score": self._final_score,
            },
        )


def _norm(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _is_polite(text: str) -> bool:
    lowered = _norm(text)
    return any(token in lowered for token in ("thank", "please", "assist", "apolog"))