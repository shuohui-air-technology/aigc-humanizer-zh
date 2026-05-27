import json
from pathlib import Path

import pytest

from scripts.evaluate_red_blue import (
    HARD_CONSTRAINTS,
    PATTERN_IDS,
    RedBlueCase,
    coverage_failures,
    evaluate,
    load_cases,
    validate_record,
)


FIXTURES = Path("tests/fixtures/red_blue")


def test_red_blue_fixtures_load_and_meet_coverage_floor() -> None:
    cases = load_cases(FIXTURES)
    result = evaluate(cases)

    assert len(cases) >= 50
    assert coverage_failures(result, min_positive=3, min_negative=2) == []


def test_red_blue_current_detector_passes_v03_thresholds() -> None:
    result = evaluate(load_cases(FIXTURES))

    assert result["bounded_scores"] is True
    assert result["negative_false_positive_rate"] <= 0.15
    assert min(item.f1 for item in result["pattern_stats"].values()) >= 0.70
    assert min(item.f1 for item in result["hard_stats"].values()) >= 0.70
    assert result["failures"] == []


def test_fixture_schema_rejects_invalid_case_type(tmp_path: Path) -> None:
    record = {
        "id": "bad",
        "case_type": "other",
        "text": "样例文本。",
        "expected_patterns": [],
        "expected_hard_constraints": [],
        "expected_risk_level": "low",
    }

    with pytest.raises(ValueError, match="case_type"):
        validate_record(record, tmp_path / "bad.jsonl", 1)


def test_fixture_schema_rejects_unknown_pattern_id(tmp_path: Path) -> None:
    record = {
        "id": "bad-pattern",
        "case_type": "positive",
        "text": "样例文本。",
        "expected_patterns": [99],
        "expected_hard_constraints": [],
        "expected_risk_level": "low",
    }

    with pytest.raises(ValueError, match="invalid expected_patterns"):
        validate_record(record, tmp_path / "bad.jsonl", 1)


def test_all_fixture_lines_are_json_objects() -> None:
    for path in FIXTURES.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            assert isinstance(json.loads(line), dict)


def test_negative_fixture_targets_all_labels() -> None:
    cases = load_cases(FIXTURES)
    negative_cases = [
        case for case in cases if case.case_type in {"negative", "near_miss"}
    ]

    assert negative_cases
    assert set(PATTERN_IDS).issubset(
        set().union(*(case.negative_patterns for case in negative_cases))
    )
    assert set(HARD_CONSTRAINTS).issubset(
        set().union(*(case.negative_hard_constraints for case in negative_cases))
    )


def test_evaluate_accepts_minimal_in_memory_case() -> None:
    case = RedBlueCase(
        id="memory-positive",
        text="专家认为，平台机制会影响参与方式。",
        expected_patterns={8},
        expected_hard_constraints={"HC-7"},
        expected_risk_level="high",
        case_type="positive",
    )

    result = evaluate([case])

    assert result["pattern_stats"][8].tp == 1
    assert result["hard_stats"]["HC-7"].tp == 1
