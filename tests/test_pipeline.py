"""Run with:  pytest -q      (uses offline mode, so Ollama is NOT required)"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crop_agents.data import DatasetError, load_dataset  # noqa: E402
from crop_agents.orchestrator import build_system  # noqa: E402
from crop_agents.validation import validate_measurements  # noqa: E402

VALID = {"N": 90, "P": 42, "K": 43, "temperature": 20.9, "humidity": 82, "ph": 6.5, "rainfall": 203}


@pytest.fixture(scope="module")
def system():
    try:
        load_dataset()
    except DatasetError:
        pytest.skip("Crop_recommendation.csv not found (download it or use scripts/make_sample_data.py)")
    return build_system(force_offline=True, verbose=False)


def test_validation_rejects_bad_values():
    _, errors, _ = validate_measurements({**VALID, "ph": 17, "humidity": 150})
    assert len(errors) == 2
    _, errors, _ = validate_measurements({"N": 1})
    assert any("Missing" in e for e in errors)
    _, errors, _ = validate_measurements({**VALID, "N": "abc"})
    assert errors


def test_valid_route_returns_model_crop_and_evidence(system):
    s = system.run({"measurements": VALID, "question": ""})
    assert s["route"] == "recommend"
    assert s["prediction"]["label"] in system.stats.crops()
    assert s["retrieved"] and s["review_status"] in ("approved", "approved_after_revision")
    assert "LIMITATIONS" in s["report_text"]


def test_invalid_input_gets_safe_response(system):
    s = system.run({"measurements": {**VALID, "ph": 17}, "question": ""})
    assert s["route"] == "invalid" and s["prediction"] is None
    assert "no recommendation" in s["report_text"].lower()


def test_out_of_scope_question(system):
    s = system.run({"measurements": None, "question": "Who won the cricket match yesterday?"})
    assert s["route"] == "unsupported" and s["prediction"] is None


def test_at_most_one_revision(system):
    s = system.run({"measurements": VALID, "question": ""}, demo_revision=True)
    assert s["retry_count"] == 1
    assert s["review_status"] == "approved_after_revision"
