#!/usr/bin/env python3
"""Red-blue evaluation harness for aigc-humanizer-zh.

The harness intentionally has no network or LLM dependency. Red-team examples
are generated or collected elsewhere, reviewed by humans, then stored as JSONL
fixtures for deterministic blue-team regression checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.patterns import PatternDetector

PATTERN_IDS = tuple(range(1, 17))
HARD_CONSTRAINTS = tuple(f"HC-{i}" for i in range(1, 8))
CASE_TYPES = {"positive", "negative", "near_miss", "adversarial"}
RISK_MAP = {
    "低": "low",
    "中": "medium",
    "高": "high",
    "low": "low",
    "medium": "medium",
    "high": "high",
}


@dataclass(frozen=True)
class RedBlueCase:
    id: str
    text: str
    expected_patterns: set[int]
    expected_hard_constraints: set[str]
    expected_risk_level: str
    case_type: str
    negative_patterns: set[int] = field(default_factory=set)
    negative_hard_constraints: set[str] = field(default_factory=set)
    source_path: Path | None = None


@dataclass
class LabelStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def normalize_risk(value: str) -> str:
    for key, normalized in RISK_MAP.items():
        if key in value:
            return normalized
    raise ValueError(f"unknown risk level: {value!r}")


def _require_list(record: dict[str, Any], key: str) -> list[Any]:
    value = record.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def validate_record(record: dict[str, Any], source: Path, line_no: int) -> RedBlueCase:
    prefix = f"{source}:{line_no}"
    required = {
        "id",
        "text",
        "expected_patterns",
        "expected_hard_constraints",
        "expected_risk_level",
        "case_type",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise ValueError(f"{prefix}: missing required fields: {', '.join(missing)}")

    if not isinstance(record["id"], str) or not record["id"]:
        raise ValueError(f"{prefix}: id must be a non-empty string")
    if not isinstance(record["text"], str) or not record["text"].strip():
        raise ValueError(f"{prefix}: text must be a non-empty string")
    if record["case_type"] not in CASE_TYPES:
        raise ValueError(f"{prefix}: case_type must be one of {sorted(CASE_TYPES)}")

    expected_patterns = {int(item) for item in _require_list(record, "expected_patterns")}
    invalid_patterns = expected_patterns - set(PATTERN_IDS)
    if invalid_patterns:
        raise ValueError(f"{prefix}: invalid expected_patterns: {sorted(invalid_patterns)}")

    expected_hard = set(_require_list(record, "expected_hard_constraints"))
    invalid_hard = expected_hard - set(HARD_CONSTRAINTS)
    if invalid_hard:
        raise ValueError(f"{prefix}: invalid expected_hard_constraints: {sorted(invalid_hard)}")

    negative_patterns = {int(item) for item in record.get("negative_patterns", [])}
    invalid_negative_patterns = negative_patterns - set(PATTERN_IDS)
    if invalid_negative_patterns:
        raise ValueError(f"{prefix}: invalid negative_patterns: {sorted(invalid_negative_patterns)}")

    negative_hard = set(record.get("negative_hard_constraints", []))
    invalid_negative_hard = negative_hard - set(HARD_CONSTRAINTS)
    if invalid_negative_hard:
        raise ValueError(f"{prefix}: invalid negative_hard_constraints: {sorted(invalid_negative_hard)}")

    return RedBlueCase(
        id=record["id"],
        text=record["text"],
        expected_patterns=expected_patterns,
        expected_hard_constraints=expected_hard,
        expected_risk_level=normalize_risk(record["expected_risk_level"]),
        case_type=record["case_type"],
        negative_patterns=negative_patterns,
        negative_hard_constraints=negative_hard,
        source_path=source,
    )


def load_cases(fixtures: Path) -> list[RedBlueCase]:
    paths = [fixtures] if fixtures.is_file() else sorted(fixtures.glob("*.jsonl"))
    if not paths:
        raise ValueError(f"no JSONL fixtures found under {fixtures}")

    cases: list[RedBlueCase] = []
    seen_ids: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                record = json.loads(stripped)
                case = validate_record(record, path, line_no)
                if case.id in seen_ids:
                    raise ValueError(f"{path}:{line_no}: duplicate case id {case.id!r}")
                seen_ids.add(case.id)
                cases.append(case)
    return cases


def predicted_labels(detector: PatternDetector, case: RedBlueCase) -> tuple[set[int], set[str], str]:
    report = detector.analyze(case.text)
    patterns = {
        match.pattern_id
        for paragraph in report.paragraph_risks
        for match in paragraph.matches
    }
    hard = {item["constraint"] for item in report.hard_violations}
    return patterns, hard, normalize_risk(report.overall_risk)


def update_stats(
    stats: dict[Any, LabelStats],
    labels: tuple[Any, ...],
    expected: set[Any],
    actual: set[Any],
) -> None:
    for label in labels:
        is_expected = label in expected
        is_actual = label in actual
        if is_expected and is_actual:
            stats[label].tp += 1
        elif is_expected and not is_actual:
            stats[label].fn += 1
        elif not is_expected and is_actual:
            stats[label].fp += 1


def coverage_counts(cases: list[RedBlueCase]) -> tuple[Counter, Counter, Counter, Counter]:
    pattern_pos: Counter[int] = Counter()
    pattern_neg: Counter[int] = Counter()
    hard_pos: Counter[str] = Counter()
    hard_neg: Counter[str] = Counter()

    for case in cases:
        pattern_pos.update(case.expected_patterns)
        hard_pos.update(case.expected_hard_constraints)

        if case.case_type in {"negative", "near_miss"}:
            pattern_targets = case.negative_patterns or set(PATTERN_IDS) - case.expected_patterns
            hard_targets = case.negative_hard_constraints or set(HARD_CONSTRAINTS) - case.expected_hard_constraints
            pattern_neg.update(pattern_targets)
            hard_neg.update(hard_targets)

    return pattern_pos, pattern_neg, hard_pos, hard_neg


def evaluate(cases: list[RedBlueCase]) -> dict[str, Any]:
    detector = PatternDetector()
    pattern_stats = {pid: LabelStats() for pid in PATTERN_IDS}
    hard_stats = {cid: LabelStats() for cid in HARD_CONSTRAINTS}
    risk_total = 0
    risk_correct = 0
    bounded_scores = True
    negative_fp = 0
    negative_total = 0
    failures: list[str] = []

    for case in cases:
        actual_patterns, actual_hard, actual_risk = predicted_labels(detector, case)
        update_stats(pattern_stats, PATTERN_IDS, case.expected_patterns, actual_patterns)
        update_stats(hard_stats, HARD_CONSTRAINTS, case.expected_hard_constraints, actual_hard)

        risk_total += 1
        risk_correct += int(case.expected_risk_level == actual_risk)

        report = detector.analyze(case.text)
        for paragraph in report.paragraph_risks:
            bounded_scores = bounded_scores and 0 <= paragraph.score <= paragraph.max_score

        if case.case_type in {"negative", "near_miss"}:
            negative_total += len(PATTERN_IDS) + len(HARD_CONSTRAINTS)
            negative_fp += len(actual_patterns - case.expected_patterns)
            negative_fp += len(actual_hard - case.expected_hard_constraints)

        missing_patterns = case.expected_patterns - actual_patterns
        missing_hard = case.expected_hard_constraints - actual_hard
        if missing_patterns or missing_hard:
            failures.append(
                f"{case.id}: missing patterns={sorted(missing_patterns)} "
                f"hard={sorted(missing_hard)}"
            )

    pattern_pos, pattern_neg, hard_pos, hard_neg = coverage_counts(cases)

    return {
        "case_count": len(cases),
        "pattern_stats": pattern_stats,
        "hard_stats": hard_stats,
        "risk_accuracy": risk_correct / risk_total if risk_total else 0.0,
        "bounded_scores": bounded_scores,
        "negative_false_positive_rate": (
            negative_fp / negative_total if negative_total else 0.0
        ),
        "coverage": {
            "pattern_positive": pattern_pos,
            "pattern_negative": pattern_neg,
            "hard_positive": hard_pos,
            "hard_negative": hard_neg,
        },
        "failures": failures,
    }


def min_f1(stats: dict[Any, LabelStats]) -> float:
    return min(item.f1 for item in stats.values()) if stats else 0.0


def format_stats(title: str, stats: dict[Any, LabelStats]) -> str:
    lines = [title]
    for label in sorted(stats):
        item = stats[label]
        lines.append(
            f"  {label}: P={item.precision:.2f} R={item.recall:.2f} "
            f"F1={item.f1:.2f} TP={item.tp} FP={item.fp} FN={item.fn}"
        )
    return "\n".join(lines)


def coverage_failures(result: dict[str, Any], min_positive: int, min_negative: int) -> list[str]:
    coverage = result["coverage"]
    failures: list[str] = []
    for pid in PATTERN_IDS:
        if coverage["pattern_positive"][pid] < min_positive:
            failures.append(f"pattern {pid} positive coverage < {min_positive}")
        if coverage["pattern_negative"][pid] < min_negative:
            failures.append(f"pattern {pid} negative coverage < {min_negative}")
    for cid in HARD_CONSTRAINTS:
        if coverage["hard_positive"][cid] < min_positive:
            failures.append(f"{cid} positive coverage < {min_positive}")
        if coverage["hard_negative"][cid] < min_negative:
            failures.append(f"{cid} negative coverage < {min_negative}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate red-blue JSONL fixtures.")
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--min-f1", type=float, default=0.70)
    parser.add_argument("--max-fpr", type=float, default=0.15)
    parser.add_argument("--min-positive", type=int, default=3)
    parser.add_argument("--min-negative", type=int, default=2)
    args = parser.parse_args(argv)

    cases = load_cases(args.fixtures)
    result = evaluate(cases)
    pattern_min_f1 = min_f1(result["pattern_stats"])
    hard_min_f1 = min_f1(result["hard_stats"])
    coverage_errors = coverage_failures(result, args.min_positive, args.min_negative)

    print(f"cases: {result['case_count']}")
    print(format_stats("patterns", result["pattern_stats"]))
    print(format_stats("hard_constraints", result["hard_stats"]))
    print(f"risk_accuracy: {result['risk_accuracy']:.2f}")
    print(f"bounded_scores: {result['bounded_scores']}")
    print(f"negative_false_positive_rate: {result['negative_false_positive_rate']:.2f}")

    errors: list[str] = []
    if pattern_min_f1 < args.min_f1:
        errors.append(f"pattern min F1 {pattern_min_f1:.2f} < {args.min_f1:.2f}")
    if hard_min_f1 < args.min_f1:
        errors.append(f"hard min F1 {hard_min_f1:.2f} < {args.min_f1:.2f}")
    if result["negative_false_positive_rate"] > args.max_fpr:
        errors.append(
            "negative/near_miss false positive rate "
            f"{result['negative_false_positive_rate']:.2f} > {args.max_fpr:.2f}"
        )
    if not result["bounded_scores"]:
        errors.append("paragraph scores are not bounded")
    errors.extend(coverage_errors)
    errors.extend(result["failures"])

    if errors:
        print("FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
