---
title: Customer Support OpenEnv Benchmark
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# Customer Support OpenEnv Benchmark

This project is a real-world customer support simulation environment for evaluating agent behavior in ticket triage, policy grounding, and escalation handling.

## Motivation

Support teams handle large volumes of incoming tickets where bad automation creates operational and safety risk. This environment evaluates whether an agent can:

- classify issue type correctly,
- ground responses in the right knowledge base entry,
- make safe escalation decisions,
- avoid low-value loops while being polite and policy-consistent.

## OpenEnv compliance

- Full OpenEnv manifest in openenv.yaml
- Typed models with Pydantic: action, observation, and state
- Environment API exposed through OpenEnv FastAPI server
- Standard endpoints available: reset, step, state, schema, health
- OpenEnv validator passes with openenv validate

## Action and observation spaces

### Action

SupportAction fields:

- action_type: classify | search_kb | respond | escalate
- content: text payload for that action

### Observation

SupportObservation fields:

- ticket_id, task_id, difficulty
- query, kb_id, requires_escalation
- history (trajectory events)
- reward, done, feedback, metadata

## Tasks and graders

Three deterministic task families are implemented, each with a programmatic grader returning a score in [0.0, 1.0].

1. easy_classify_respond
Objective: correct category classification and high-fidelity response.

2. medium_kb_grounded_response
Objective: correct KB retrieval followed by grounded response.

3. hard_escalation_safety
Objective: correct escalation decision plus safe/polite response behavior.

## Reward design

Reward shaping provides dense trajectory signal:

- positive for correct classify/search_kb decisions
- response similarity rewards for grounded answers
- politeness bonus
- penalties for hallucination risk and unnecessary escalation
- per-step loop penalty

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run server locally:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Run baseline inference:

```bash
set API_BASE_URL=https://api.openai.com/v1
set MODEL_NAME=gpt-4o-mini
set HF_TOKEN=your_api_token
python inference.py --csv dataset.csv --limit-per-task 5
```

Offline smoke run (no API calls):

```bash
python inference.py --csv dataset.csv --limit-per-task 3 --offline
```

## Strict inference log format

Inference emits only structured markers:

- [START]
- [STEP]
- [END]

## Baseline score reproducibility

- Deterministic decoding (temperature=0)
- fixed task ordering (easy, medium, hard)
- deterministic grader criteria
- one model planning call per episode to reduce runtime variance and cost

## Pre-submission validation

Run all checks in one command:

```bash
python pre_submission_validate.py --skip-docker
```

Checks included:

- required env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN
- openenv validate
- POST /reset returns HTTP 200
- optional docker build

For full parity with judge checks, run without skip:

```bash
python pre_submission_validate.py
```

## Docker

```bash
docker build -t customer-support-openenv .
docker run -p 8000:8000 customer-support-openenv
```

## Hugging Face Spaces

- Space SDK: Docker
- Required tag: openenv
- Health endpoint: /health
- Reset endpoint: POST /reset

Deploy command (after huggingface login):

```bash
openenv push
```