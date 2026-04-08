import asyncio

from client import CustomerSupportEnvClient
from models import SupportAction


async def run_episode(difficulty: str, index: int) -> dict:
    env = CustomerSupportEnvClient(base_url="http://127.0.0.1:8000")
    await env.connect()
    reset_result = await env.reset(difficulty=difficulty, index=index)

    rewards: list[float] = []
    done = bool(reset_result.done)
    observation = reset_result.observation

    actions = [
        SupportAction(action_type="classify", content="technical_issue"),
        SupportAction(action_type="search_kb", content=observation.kb_id),
        SupportAction(
            action_type="respond",
            content="Thank you for contacting us. We will assist you shortly.",
        ),
    ]

    for action in actions:
        if done:
            break
        step_result = await env.step(action)
        rewards.append(float(step_result.reward or 0.0))
        done = bool(step_result.done)
        observation = step_result.observation

    await env.close()
    return {
        "difficulty": difficulty,
        "index": index,
        "ticket_id": observation.ticket_id,
        "done": done,
        "steps": len(rewards),
        "reward_sum": round(sum(rewards), 4),
    }


async def main() -> None:
    for difficulty in ("easy", "medium", "hard"):
        for index in (0, 1):
            result = await run_episode(difficulty, index)
            print(result)


if __name__ == "__main__":
    asyncio.run(main())
