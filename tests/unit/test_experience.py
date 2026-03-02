"""Tests for experience helper functions."""

from __future__ import annotations

import json

from agenix.storage.experience import extract_problem_id


class TestExtractProblemId:
    def test_extracts_from_first_user_message(self):
        log = json.dumps({
            "role": "user",
            "content": json.dumps({
                "problem": {"problem_id": "prob_123", "title": "Test"},
                "knowledge": [],
            }),
        })
        assert extract_problem_id(log) == "prob_123"

    def test_skips_non_user_roles(self):
        lines = [
            json.dumps({"role": "system", "content": "init"}),
            json.dumps({
                "role": "user",
                "content": json.dumps({
                    "problem": {"problem_id": "prob_456"},
                }),
            }),
        ]
        log = "\n".join(lines)
        assert extract_problem_id(log) == "prob_456"

    def test_returns_none_for_no_problem(self):
        log = json.dumps({"role": "user", "content": "just text"})
        assert extract_problem_id(log) is None

    def test_returns_none_for_empty(self):
        assert extract_problem_id("") is None

    def test_returns_none_for_malformed(self):
        assert extract_problem_id("not json\nalso not json\n") is None
