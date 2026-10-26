"""Shared state dictionary passed between all agents (step 6)."""
from __future__ import annotations
from typing import Any, Dict


def new_state(user_input: Dict[str, Any], demo_revision: bool = False) -> Dict[str, Any]:
    return {
        "user_input": user_input,      # raw input: {"measurements": {...}|None, "question": str}
        "question": "",
        "measurements": None,          # validated numeric measurements
        "route": None,                 # recommend | knowledge | invalid | unsupported
        "route_reason": "",
        "validation_errors": [],
        "warnings": [],
        "soil_analysis": None,
        "prediction": None,            # output of the dataset-derived predictor tool
        "rationale": [],
        "retrieved": [],               # retrieved passages [{id, source, crop, text, score}]
        "draft": None,
        "review": None,                # {"verdict", "issues", "by"}
        "review_status": "pending",
        "retry_count": 0,
        "limitations": [],
        "final_response": None,
        "report_text": "",
        "demo_revision": demo_revision,
        "trace": [],
    }


def log(state: Dict[str, Any], agent: str, message: str) -> None:
    state["trace"].append(f"[{agent}] {message}")
