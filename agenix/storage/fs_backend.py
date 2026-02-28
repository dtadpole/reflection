"""Filesystem + DuckDB storage backend.

Data is stored as one JSON file per entity in a structured directory layout.
DuckDB is used as an in-process query engine to scan and filter JSON files
without maintaining a persistent database.

Directory layout under <env_path>:
    problems/<problem_id>.json          shared across runs
    cards/<card_id>.json                shared across runs (all card types)
    <run_tag>/solver/<trajectory_id>.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, TypeVar

import duckdb
from pydantic import BaseModel

from agenix.config import StorageConfig
from agenix.storage.models import (
    Card,
    CardStatus,
    CardType,
    InsightCard,
    KnowledgeCard,
    Problem,
    ProblemStatus,
    ReflectionCard,
    Trajectory,
)

T = TypeVar("T", bound=BaseModel)


def _write_json(path: Path, model: BaseModel) -> None:
    """Write a Pydantic model as a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")


def _read_json(path: Path, model_cls: type[T]) -> T:
    """Read a JSON file into a Pydantic model."""
    return model_cls.model_validate_json(path.read_text(encoding="utf-8"))


def _read_card_json(path: Path) -> KnowledgeCard | InsightCard | ReflectionCard:
    """Read a card JSON file, dispatching to the correct card subclass."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    card_type = raw.get("card_type", "knowledge")
    if card_type == CardType.INSIGHT.value:
        return InsightCard.model_validate(raw)
    if card_type == CardType.REFLECTION.value:
        return ReflectionCard.model_validate(raw)
    return KnowledgeCard.model_validate(raw)


class FSBackend:
    """Filesystem-based storage with DuckDB query engine."""

    def __init__(self, config: StorageConfig | None = None) -> None:
        self.config = config or StorageConfig()

    @property
    def env_path(self) -> Path:
        return self.config.env_path

    @property
    def problems_dir(self) -> Path:
        return self.config.problems_path

    @property
    def cards_dir(self) -> Path:
        return self.config.cards_path

    def run_dir(self, run_tag: str) -> Path:
        return self.config.run_path(run_tag)

    def initialize(self) -> None:
        """Create the directory structure."""
        self.problems_dir.mkdir(parents=True, exist_ok=True)
        self.cards_dir.mkdir(parents=True, exist_ok=True)

    # --- Problems ---

    def save_problem(self, problem: Problem) -> Path:
        path = self.problems_dir / f"{problem.problem_id}.json"
        _write_json(path, problem)
        return path

    def get_problem(self, problem_id: str) -> Optional[Problem]:
        path = self.problems_dir / f"{problem_id}.json"
        if not path.exists():
            return None
        return _read_json(path, Problem)

    def list_problems(
        self,
        status: Optional[ProblemStatus] = None,
        domain: Optional[str] = None,
        limit: int = 100,
    ) -> list[Problem]:
        if not self.problems_dir.exists():
            return []
        problems = [
            _read_json(p, Problem)
            for p in sorted(self.problems_dir.glob("*.json"))
        ]
        if status:
            problems = [p for p in problems if p.status == status]
        if domain:
            problems = [p for p in problems if p.domain == domain]
        problems.sort(key=lambda p: p.created_at, reverse=True)
        return problems[:limit]

    def update_problem_status(self, problem_id: str, status: ProblemStatus) -> None:
        problem = self.get_problem(problem_id)
        if problem is None:
            raise FileNotFoundError(f"Problem not found: {problem_id}")
        problem.status = status
        self.save_problem(problem)

    # --- Trajectories ---

    def save_trajectory(
        self, trajectory: Trajectory, run_tag: str, agent: str = "solver"
    ) -> Path:
        path = (
            self.run_dir(run_tag)
            / agent
            / f"{trajectory.trajectory_id}.json"
        )
        _write_json(path, trajectory)
        return path

    def get_trajectory(
        self, trajectory_id: str, run_tag: str, agent: str = "solver"
    ) -> Optional[Trajectory]:
        path = self.run_dir(run_tag) / agent / f"{trajectory_id}.json"
        if not path.exists():
            return None
        return _read_json(path, Trajectory)

    def list_trajectories(
        self,
        run_tag: Optional[str] = None,
        agent: str = "solver",
        is_correct: Optional[bool] = None,
        limit: int = 100,
    ) -> list[Trajectory]:
        """List trajectories. If run_tag is None, scans all runs."""
        if run_tag:
            search_dir = self.run_dir(run_tag) / agent
            patterns = [search_dir]
        else:
            patterns = sorted(self.env_path.glob(f"run_*/{agent}"))

        trajectories: list[Trajectory] = []
        for d in patterns:
            if not d.is_dir():
                continue
            for f in d.glob("*.json"):
                trajectories.append(_read_json(f, Trajectory))

        if is_correct is not None:
            trajectories = [t for t in trajectories if t.is_correct == is_correct]
        trajectories.sort(key=lambda t: t.created_at, reverse=True)
        return trajectories[:limit]

    # --- Cards ---

    def save_card(self, card: Card) -> Path:
        path = self.cards_dir / f"{card.card_id}.json"
        _write_json(path, card)
        return path

    def get_card(self, card_id: str) -> Optional[KnowledgeCard | InsightCard | ReflectionCard]:
        path = self.cards_dir / f"{card_id}.json"
        if not path.exists():
            return None
        return _read_card_json(path)

    def list_cards(
        self,
        card_type: Optional[CardType] = None,
        domain: Optional[str] = None,
        status: Optional[CardStatus] = CardStatus.ACTIVE,
        limit: int = 100,
    ) -> list[KnowledgeCard | InsightCard | ReflectionCard]:
        if not self.cards_dir.exists():
            return []
        cards = [_read_card_json(p) for p in sorted(self.cards_dir.glob("*.json"))]
        if status is not None:
            cards = [c for c in cards if c.status == status]
        if card_type:
            cards = [c for c in cards if c.card_type == card_type]
        if domain:
            cards = [
                c for c in cards
                if isinstance(c, KnowledgeCard) and c.domain == domain
            ]
        cards.sort(key=lambda c: c.updated_at, reverse=True)
        return cards[:limit]

    def list_cards_by_trajectory(
        self,
        trajectory_id: str,
        status: Optional[CardStatus] = CardStatus.ACTIVE,
        limit: int = 100,
    ) -> list[ReflectionCard]:
        """List reflection cards for a given trajectory."""
        cards = self.list_cards(
            card_type=CardType.REFLECTION, status=status, limit=limit
        )
        return [
            c for c in cards
            if isinstance(c, ReflectionCard) and c.trajectory_id == trajectory_id
        ]

    def find_cards_by_source(
        self,
        source_id: str,
        source_type: Optional[str] = None,
    ) -> list[KnowledgeCard | InsightCard | ReflectionCard]:
        """Find cards referencing a given source entity in their source_refs."""
        if not self.cards_dir.exists():
            return []
        results: list[KnowledgeCard | InsightCard | ReflectionCard] = []
        for p in sorted(self.cards_dir.glob("*.json")):
            card = _read_card_json(p)
            for ref in card.source_refs:
                if ref.id == source_id and (
                    source_type is None or ref.type == source_type
                ):
                    results.append(card)
                    break
        return results

    # --- DuckDB Query Engine ---

    def query_problems(self, sql_where: str = "", limit: int = 100) -> list[dict]:
        """Query problems via DuckDB SQL over JSON files."""
        return self._query_json_dir(self.problems_dir, sql_where, limit)

    def query_cards(self, sql_where: str = "", limit: int = 100) -> list[dict]:
        """Query cards via DuckDB SQL over JSON files."""
        return self._query_json_dir(self.cards_dir, sql_where, limit)

    def query_trajectories(
        self,
        sql_where: str = "",
        run_tag: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query trajectories via DuckDB SQL over JSON files."""
        if run_tag:
            search_dir = self.run_dir(run_tag) / "solver"
        else:
            # Scan all runs — use glob pattern
            return self._query_json_glob(
                str(self.env_path / "run_*" / "solver" / "*.json"),
                sql_where,
                limit,
            )
        return self._query_json_dir(search_dir, sql_where, limit)

    def _query_json_dir(
        self, directory: Path, sql_where: str = "", limit: int = 100
    ) -> list[dict]:
        """Run a DuckDB query over all JSON files in a directory."""
        if not directory.exists() or not list(directory.glob("*.json")):
            return []
        glob_pattern = str(directory / "*.json")
        return self._query_json_glob(glob_pattern, sql_where, limit)

    def _query_json_glob(
        self, glob_pattern: str, sql_where: str = "", limit: int = 100
    ) -> list[dict]:
        """Run a DuckDB query over JSON files matching a glob pattern."""
        query = f"SELECT * FROM read_json_auto('{glob_pattern}')"
        if sql_where:
            query += f" WHERE {sql_where}"
        query += f" LIMIT {limit}"
        try:
            con = duckdb.connect()
            result = con.execute(query)
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
            con.close()
            return [dict(zip(columns, row)) for row in rows]
        except duckdb.IOException:
            # No files match the glob
            return []

    # --- Stats ---

    def count_problems(self, status: Optional[ProblemStatus] = None) -> int:
        if not self.problems_dir.exists():
            return 0
        if status:
            return len([
                p for p in self.problems_dir.glob("*.json")
                if _read_json(p, Problem).status == status
            ])
        return len(list(self.problems_dir.glob("*.json")))

    def count_cards(self, card_type: Optional[CardType] = None) -> int:
        if not self.cards_dir.exists():
            return 0
        if card_type:
            return len([
                p for p in self.cards_dir.glob("*.json")
                if _read_card_json(p).card_type == card_type
            ])
        return len(list(self.cards_dir.glob("*.json")))

    def count_trajectories(
        self, is_correct: Optional[bool] = None, run_tag: Optional[str] = None
    ) -> int:
        return len(self.list_trajectories(
            run_tag=run_tag, is_correct=is_correct, limit=999999
        ))
