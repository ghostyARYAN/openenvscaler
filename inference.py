from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from openai import OpenAI

from models import SupportAction
from server.customer_support_environment import CustomerSupportEnvironment
from tasks import grade_task


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None = None) -> None:
    error_value = "null" if error is None else error.replace("\n", " ")
    print(
        f"[STEP] step={step} action={action!r} reward={reward:.4f} done={done} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    reward_text = ",".join(f"{r:.4f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards=[{reward_text}]",
        flush=True,
    )


def get_model_plan(client: OpenAI, model_name: str, task_prompt: str) -> Dict[str, str | bool]:
    completion = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a customer support policy agent. Reply with strict JSON only and no markdown. "
                    "Schema: {\"category\": string, \"search_kb\": bool, \"escalate\": bool, \"response\": string}."
                ),
            },
            {"role": "user", "content": task_prompt},
        ],
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content or "{}"
    data = json.loads(raw)
    return {
        "category": str(data.get("category", "technical_issue")),
        "search_kb": bool(data.get("search_kb", True)),
        "escalate": bool(data.get("escalate", False)),
        "response": str(data.get("response", "Thank you for contacting support. We will assist shortly.")),
    }


def heuristic_plan(observation: object) -> Dict[str, str | bool]:
    query = str(getattr(observation, "query")).lower()
    requires_escalation = bool(getattr(observation, "requires_escalation"))

    category = "technical_issue"
    if any(t in query for t in ("refund", "charge", "invoice", "billing")):
        category = "billing"
    elif any(t in query for t in ("cancel", "termination", "close account", "cancellation")):
        category = "cancellation"
    elif any(t in query for t in ("warranty", "buy", "purchase", "availability", "product")):
        category = "product_inquiry"
    elif any(t in query for t in ("refund not received", "return", "reimburse")):
        category = "refund"

    return {
        "category": category,
        "search_kb": True,
        "escalate": requires_escalation,
        "response": "Thank you for contacting us. We will review your case and assist you shortly.",
    }


def build_actions(observation: object, plan: Dict[str, str | bool]) -> List[SupportAction]:
    actions = [
        SupportAction(action_type="classify", content=str(plan["category"])),
    ]

    if bool(plan.get("search_kb", True)):
        actions.append(SupportAction(action_type="search_kb", content=str(getattr(observation, "kb_id"))))

    if bool(plan.get("escalate", False)):
        actions.append(SupportAction(action_type="escalate", content="Escalating to specialist support for manual review."))
    else:
        actions.append(SupportAction(action_type="respond", content=str(plan.get("response", "Thank you for contacting support."))))
    return actions


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenEnv baseline inference for customer support benchmark")
    parser.add_argument("--csv", default="dataset.csv")
    parser.add_argument("--limit-per-task", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--offline", action="store_true", help="Use heuristic policy without API calls")
    args = parser.parse_args()

    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    hf_token = os.getenv("HF_TOKEN")

    if not args.offline:
        missing = [name for name, value in (("API_BASE_URL", api_base_url), ("MODEL_NAME", model_name), ("HF_TOKEN", hf_token)) if not value]
        if missing:
            raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

    client = OpenAI(base_url=api_base_url, api_key=hf_token) if not args.offline else None
    env = CustomerSupportEnvironment(csv_path=args.csv)

    all_rewards: List[float] = []
    task_scores: List[float] = []
    task_labels: List[str] = []
    total_steps = 0

    log_start(task="all", env="customer_support_benchmark", model=model_name or "offline-heuristic")
    difficulties = ["easy", "medium", "hard"]
    for difficulty in difficulties:
        for idx in range(args.limit_per_task):
            obs = env.reset(difficulty=difficulty, index=idx)
            rewards: List[float] = []
            done = False

            prompt = (
                "Return the best support plan for this ticket. "
                "Prefer policy-safe responses and escalate only when needed.\n"
                f"TicketID={obs.ticket_id}\n"
                f"Difficulty={obs.difficulty}\n"
                f"Task={obs.task_id}\n"
                f"Query={obs.query}\n"
                f"KB={obs.kb_id}\n"
                f"RequiresEscalation={obs.requires_escalation}\n"
            )

            error = None
            try:
                plan = heuristic_plan(obs)
                if client is not None and model_name is not None:
                    plan = get_model_plan(client, model_name, prompt)
            except Exception as exc:
                error = str(exc)
                plan = heuristic_plan(obs)

            planned_actions = build_actions(obs, plan)

            for step, action in enumerate(planned_actions[: args.max_steps], start=1):
                if done:
                    break
                obs = env.step(action)

                reward = float(obs.reward or 0.0)
                rewards.append(reward)
                all_rewards.append(reward)
                total_steps += 1
                done = bool(obs.done)
                step_error = error if step == 1 else None
                log_step(step=step, action=f"{action.action_type}|{action.content}", reward=reward, done=done, error=step_error)

            final_score = grade_task(obs.task_id, obs.metadata, obs.history)
            task_scores.append(final_score)
            task_labels.append(difficulty)
            log_step(
                step=total_steps,
                action=f"episode_score|difficulty={difficulty}|index={idx}",
                reward=final_score,
                done=True,
                error=None,
            )

    benchmark_score = sum(task_scores) / len(task_scores) if task_scores else 0.0
    by_task: dict[str, list[float]] = {"easy": [], "medium": [], "hard": []}
    for label, score in zip(task_labels, task_scores):
        by_task[label].append(score)
    for label in ("easy", "medium", "hard"):
        values = by_task[label]
        avg = sum(values) / len(values) if values else 0.0
        log_step(
            step=total_steps,
            action=f"task_average|difficulty={label}",
            reward=avg,
            done=False,
            error=None,
        )

    print(
        f"[END] success={str(benchmark_score >= 0.7).lower()} steps={total_steps} score={benchmark_score:.4f} rewards_count={len(all_rewards)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())