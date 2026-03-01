"""Output parsers for agent responses.

Extracted from pipeline.py so handlers can reuse them independently.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from agenix.storage.models import (
    Difficulty,
    InsightCard,
    KnowledgeCard,
    Problem,
    ReflectionCard,
    ReflectionCategory,
    TestCase,
    TestResult,
    Trajectory,
)


def extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from agent output text.

    Handles:
    1. Raw JSON
    2. JSON wrapped in markdown code blocks (```json ... ```)
    3. Prose-prefixed JSON (text before the first '{' or '[')
    """
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from code block
    if "```" in text:
        for block in text.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue

    # Try scanning for first '{' (prose-prefixed JSON)
    brace = text.find("{")
    if brace >= 0:
        try:
            return json.loads(text[brace:])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from agent output: {text[:200]}")


def coerce_str(value: Any) -> str:
    """Coerce a value to string — JSON-encode non-string types."""
    if isinstance(value, str):
        return value
    return json.dumps(value)


def parse_problem(output: str) -> Problem:
    """Parse curator output into a Problem."""
    data = extract_json(output)
    test_cases = []
    for tc in data.get("test_cases", []):
        test_cases.append(TestCase(
            input=coerce_str(tc.get("input", "")),
            expected_output=coerce_str(tc.get("expected_output", "")),
            description=tc.get("description", ""),
        ))
    difficulty = Difficulty(data.get("difficulty", "medium"))
    return Problem(
        title=data["title"],
        description=data["description"],
        test_cases=test_cases,
        domain=data.get("domain", "general"),
        difficulty=difficulty,
    )


def parse_trajectory(output: str, problem_id: str) -> Trajectory:
    """Parse solver output into a Trajectory."""
    data = extract_json(output)
    test_results = [
        TestResult(
            test_case=TestCase(**tr["test_case"]),
            passed=tr["passed"],
            actual_output=tr.get("actual_output", ""),
            error=tr.get("error", ""),
        )
        for tr in data.get("test_results", [])
    ]
    return Trajectory(
        problem_id=problem_id,
        code_solution=data.get("code_solution", ""),
        final_answer=data.get("final_answer", ""),
        is_correct=data.get("is_correct", False),
        test_results=test_results,
        completed_at=datetime.now(timezone.utc),
    )


def parse_reflection_cards(
    output: str, trajectory_id: str
) -> list[ReflectionCard]:
    """Parse critic output into ReflectionCards."""
    data = extract_json(output)
    cards = []
    for rc in data.get("reflection_cards", []):
        try:
            category = ReflectionCategory(rc.get("category", "general"))
        except ValueError:
            category = ReflectionCategory.GENERAL
        cards.append(ReflectionCard(
            title=rc["title"],
            content=rc["content"],
            trajectory_id=trajectory_id,
            category=category,
            confidence=rc.get("confidence", 0.5),
            tags=rc.get("tags", []),
            supporting_steps=rc.get("supporting_steps", []),
        ))
    return cards


def parse_knowledge_actions(output: str) -> list[KnowledgeCard]:
    """Parse organizer output into KnowledgeCards (create actions only for now)."""
    data = extract_json(output)
    cards = []
    for action in data.get("actions", []):
        if action.get("action") != "create":
            continue
        cards.append(KnowledgeCard(
            title=action["title"],
            content=action["content"],
            domain=action.get("domain", "general"),
            applicability=action.get("applicability", ""),
            limitations=action.get("limitations", ""),
            tags=action.get("tags", []),
            related_card_ids=action.get("related_card_ids", []),
        ))
    return cards


def parse_insight_cards(output: str) -> list[InsightCard]:
    """Parse insight finder output into InsightCards."""
    data = extract_json(output)
    cards = []
    for ic in data.get("insight_cards", []):
        cards.append(InsightCard(
            title=ic["title"],
            content=ic["content"],
            hypothesis=ic.get("hypothesis", ""),
            evidence_for=ic.get("evidence_for", []),
            evidence_against=ic.get("evidence_against", []),
            tags=ic.get("tags", []),
        ))
    return cards
