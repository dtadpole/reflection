"""Output parsers for agent responses.

Extracted from pipeline.py so handlers can reuse them independently.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from agenix.storage.models import (
    Card,
    Difficulty,
    Experience,
    Problem,
    TestCase,
    TestResult,
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

    # Try scanning for JSON objects using raw_decode (handles trailing text)
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        brace = text.find("{", idx)
        if brace < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, brace)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        idx = brace + 1

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


def parse_experience(
    output: str, problem_id: str, *, experience_id: str | None = None,
) -> Experience:
    """Parse solver output into an Experience."""
    data = extract_json(output)
    test_results = []
    for tr in data.get("test_results", []):
        if not isinstance(tr, dict) or "passed" not in tr:
            continue
        tc_data = tr.get("test_case")
        test_case = TestCase(**tc_data) if isinstance(tc_data, dict) else TestCase(
            input=str(tr.get("input", "")),
            expected_output=str(tr.get("expected_output", "")),
        )
        test_results.append(TestResult(
            test_case=test_case,
            passed=tr["passed"],
            actual_output=tr.get("actual_output", ""),
            error=tr.get("error", ""),
        ))
    kwargs: dict = dict(
        problem_id=problem_id,
        code_solution=data.get("code_solution", ""),
        final_answer=data.get("final_answer", ""),
        is_correct=data.get("is_correct", False),
        test_results=test_results,
        completed_at=datetime.now(timezone.utc),
    )
    if experience_id is not None:
        kwargs["experience_id"] = experience_id
    return Experience(**kwargs)


def parse_reflection_cards(
    output: str, experience_ids: list[str],
) -> list[Card]:
    """Parse critic output into reflection Cards."""
    data = extract_json(output)
    cards = []
    for rc in data.get("reflection_cards", []):
        cards.append(Card(
            card_type="reflection",
            title=rc["title"],
            content=rc["content"],
            code_snippet=rc.get("code_snippet", ""),
            experience_ids=experience_ids[:3],
            category=rc.get("category", "general"),
            confidence=rc.get("confidence", 0.5),
            tags=rc.get("tags", []),
            supporting_steps=rc.get("supporting_steps", []),
            applies_to=rc.get("applies_to", []),
            not_applies_to=rc.get("not_applies_to", []),
        ))
    return cards


def parse_knowledge_actions(
    output: str, experience_ids: list[str] | None = None,
) -> list[Card]:
    """Parse organizer output into knowledge Cards (create actions only for now)."""
    data = extract_json(output)
    cards = []
    for action in data.get("actions", []):
        if action.get("action") != "create":
            continue
        cards.append(Card(
            card_type="knowledge",
            title=action["title"],
            content=action["content"],
            code_snippet=action.get("code_snippet", ""),
            experience_ids=(experience_ids or [])[:3],
            domain=action.get("domain", "general"),
            applicability=action.get("applicability", ""),
            limitations=action.get("limitations", ""),
            tags=action.get("tags", []),
            related_card_ids=action.get("related_card_ids", []),
            applies_to=action.get("applies_to", []),
            not_applies_to=action.get("not_applies_to", []),
        ))
    return cards


def parse_insight_cards(
    output: str, experience_ids: list[str] | None = None,
) -> list[Card]:
    """Parse insight finder output into insight Cards."""
    data = extract_json(output)
    cards = []
    for ic in data.get("insight_cards", []):
        cards.append(Card(
            card_type="insight",
            title=ic["title"],
            content=ic["content"],
            code_snippet=ic.get("code_snippet", ""),
            experience_ids=(experience_ids or [])[:3],
            hypothesis=ic.get("hypothesis", ""),
            evidence_for=ic.get("evidence_for", []),
            evidence_against=ic.get("evidence_against", []),
            tags=ic.get("tags", []),
            applies_to=ic.get("applies_to", []),
            not_applies_to=ic.get("not_applies_to", []),
        ))
    return cards
