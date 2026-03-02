"""Output parsers for agent responses.

Extracted from pipeline.py so handlers can reuse them independently.
"""

from __future__ import annotations

import json
import re
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


def _parse_reflection_cards_json(
    data: dict, experience_ids: list[str],
) -> list[Card]:
    """Parse JSON-format critic output into reflection Cards."""
    cards = []
    for rc in data.get("reflection_cards", []):
        cards.append(Card(
            card_type="reflection",
            title=rc["title"],
            content=rc["content"],
            code_snippet=rc.get("code_snippet", ""),
            experience_ids=experience_ids[:3],
            tags=rc.get("tags", []),
            applicability=rc.get("applicability", ""),
            limitations=rc.get("limitations", ""),
        ))
    return cards


def _parse_reflection_cards_markdown(
    output: str, experience_ids: list[str],
) -> list[Card]:
    """Parse markdown-format critic output into reflection Cards.

    Handles the natural output format of the Claude agent, which produces
    markdown sections like:
        ### Reflection Card N — Title
        **Observation:** ...
        **Code snippet:**
        ```python
        ...
        ```
        **Confidence:** 0.85
    """
    cards = []
    # Split on "### Reflection Card" or "### Card" headers
    sections = re.split(
        r"###\s+(?:Reflection\s+)?Card\s*\d*\s*[—–-]\s*",
        output,
        flags=re.IGNORECASE,
    )
    for section in sections[1:]:  # Skip preamble before first card
        # Title: first non-empty line
        lines = section.strip().split("\n")
        title = lines[0].strip().rstrip("*").strip()

        # Code snippet: extract from fenced code block
        code_match = re.search(
            r"```(?:python)?\s*\n(.*?)```",
            section,
            re.DOTALL,
        )
        code_snippet = code_match.group(1).strip() if code_match else ""

        # Content: everything except the title line
        content = "\n".join(lines[1:]).strip()

        # Tags: look for **Tags:** or tags: [...]
        tags_match = re.search(
            r"\*\*Tags[:\*]*\s*(.+)",
            section,
        )
        tags: list[str] = []
        if tags_match:
            raw = tags_match.group(1).strip()
            tags = [t.strip().strip("`") for t in raw.split(",")]

        cards.append(Card(
            card_type="reflection",
            title=title,
            content=content,
            code_snippet=code_snippet,
            experience_ids=experience_ids[:3],
            tags=tags,
        ))
    return cards


def parse_reflection_cards(
    output: str, experience_ids: list[str],
) -> list[Card]:
    """Parse critic output into reflection Cards.

    Tries JSON first, falls back to markdown parsing.
    """
    try:
        data = extract_json(output)
        return _parse_reflection_cards_json(data, experience_ids)
    except (ValueError, KeyError):
        pass
    # Fall back to markdown parsing
    return _parse_reflection_cards_markdown(output, experience_ids)


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
            applicability=action.get("applicability", ""),
            limitations=action.get("limitations", ""),
            tags=action.get("tags", []),
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
            tags=ic.get("tags", []),
            applicability=ic.get("applicability", ""),
            limitations=ic.get("limitations", ""),
        ))
    return cards
