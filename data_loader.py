from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

DIFFICULTIES = ("easy", "medium", "hard")
DESCRIPTION_COLUMNS = ("Description", "Ticket Description")

REQUIRED_COLUMNS = {
    "TicketID",
    "expected_category",
    "expected_action",
    "expected_response",
    "requires_escalation",
    "knowledge_base_id",
    "difficulty",
}


def load_dataset(csv_path: str | Path = "dataset.csv") -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists() and not path.is_absolute():
        candidate = Path(__file__).resolve().parent / path
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)
    desc_col = next((c for c in DESCRIPTION_COLUMNS if c in df.columns), None)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if desc_col is None:
        missing.append("Description or Ticket Description")
    if missing:
        raise ValueError(f"Missing required dataset columns: {', '.join(missing)}")

    df = df.copy()
    if desc_col != "Description":
        df["Description"] = df[desc_col]
    df["difficulty"] = df["difficulty"].astype(str).str.lower().str.strip()
    df["requires_escalation"] = (
        df["requires_escalation"].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)
    )
    df["TicketID"] = df["TicketID"].astype(int)
    return df


def split_difficulty(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {d: df[df["difficulty"] == d].reset_index(drop=True).copy() for d in DIFFICULTIES}


def build_expected(row: pd.Series) -> dict:
    return {
        "ticket_id": int(row["TicketID"]),
        "query": str(row["Description"]),
        "expected_category": str(row["expected_category"]),
        "expected_action": str(row["expected_action"]),
        "expected_response": str(row["expected_response"]),
        "requires_escalation": bool(row["requires_escalation"]),
        "kb_id": str(row["knowledge_base_id"]),
        "difficulty": str(row["difficulty"]),
    }