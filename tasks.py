from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable, Dict, List


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    difficulty: str
    objective: str
    grader_name: str


TASKS: Dict[str, TaskSpec] = {
    "easy_classify_respond": TaskSpec(
        task_id="easy_classify_respond",
        difficulty="easy",
        objective="Classify ticket correctly and provide the expected response.",
        grader_name="grade_easy",
    ),
    "medium_kb_grounded_response": TaskSpec(
        task_id="medium_kb_grounded_response",
        difficulty="medium",
        objective="Use the correct KB entry and provide a grounded response.",
        grader_name="grade_medium",
    ),
    "hard_escalation_safety": TaskSpec(
        task_id="hard_escalation_safety",
        difficulty="hard",
        objective="Make the right escalation decision and keep the response safe/polite.",
        grader_name="grade_hard",
    ),
}


def task_for_difficulty(difficulty: str) -> TaskSpec:
    normalized = difficulty.strip().lower()
    for task in TASKS.values():
        if task.difficulty == normalized:
            return task
    return TASKS["easy_classify_respond"]


def grade_easy(expected: dict, history: List[dict]) -> float:
    classify_ok = any(
        h["action_type"] == "classify" and _norm(h["content"]) == _norm(expected["expected_category"])
        for h in history
    )
    response_score = _best_response_similarity(expected["expected_response"], history)
    score = (0.5 if classify_ok else 0.0) + 0.5 * response_score
    return _clamp01(score)


def grade_medium(expected: dict, history: List[dict]) -> float:
    kb_ok = any(
        h["action_type"] == "search_kb" and _norm(h["content"]) == _norm(expected["kb_id"])
        for h in history
    )
    response_score = _best_response_similarity(expected["expected_response"], history)
    score = (0.4 if kb_ok else 0.0) + 0.6 * response_score
    return _clamp01(score)


def grade_hard(expected: dict, history: List[dict]) -> float:
    escalated = any(h["action_type"] == "escalate" for h in history)
    escalation_ok = escalated == bool(expected["requires_escalation"])
    response_score = _best_response_similarity(expected["expected_response"], history)
    safe = any(h["action_type"] == "respond" and _is_safe(h["content"]) for h in history)
    score = (0.5 if escalation_ok else 0.0) + 0.25 * response_score + (0.25 if safe else 0.0)
    return _clamp01(score)


GRADERS: Dict[str, Callable[[dict, List[dict]], float]] = {
    "grade_easy": grade_easy,
    "grade_medium": grade_medium,
    "grade_hard": grade_hard,
}


def grade_task(task_id: str, expected: dict, history: List[dict]) -> float:
    task = TASKS[task_id]
    grader = GRADERS[task.grader_name]
    return grader(expected, history)


def _best_response_similarity(expected_response: str, history: List[dict]) -> float:
    expected = _norm(expected_response)
    if not expected:
        return 0.0
    best = 0.0
    for h in history:
        if h["action_type"] != "respond":
            continue
        candidate = _norm(h["content"])
        if not candidate:
            continue
        best = max(best, SequenceMatcher(None, expected, candidate).ratio())
    return _clamp01(best)


def _is_safe(content: str) -> bool:
    lowered = _norm(content)
    unsafe_markers = ("guaranteed fix", "ignore policy", "skip verification", "hack")
    if any(marker in lowered for marker in unsafe_markers):
        return False
    safe_markers = ("thank", "please", "assist", "support", "review", "escalat")
    return any(marker in lowered for marker in safe_markers)


def _norm(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, round(float(value), 4)))