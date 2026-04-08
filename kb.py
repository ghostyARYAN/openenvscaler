from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd


def build_knowledge_base(df: pd.DataFrame) -> dict[str, dict]:
    grouped: dict[str, list] = defaultdict(list)
    for _, row in df.iterrows():
        grouped[str(row["knowledge_base_id"])].append(row)

    kb: dict[str, dict] = {}
    for kb_id, rows in grouped.items():
        kb[kb_id] = {
            "title": _most_common([str(row.get("Subject", kb_id)) for row in rows]),
            "category": _most_common([str(row["expected_category"]) for row in rows]),
            "canonical_action": _most_common([str(row["expected_action"]) for row in rows]),
            "canonical_response": _most_common([str(row["expected_response"]) for row in rows]),
            "summary": "Ground responses in policy and avoid unsupported claims.",
        }
    return kb


def _most_common(items: list[str]) -> str:
    if not items:
        return ""
    return Counter(items).most_common(1)[0][0]