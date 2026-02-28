"""LanceDB vector index wrapper for knowledge card embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import lancedb
import numpy as np
import pyarrow as pa

from agenix.config import StorageConfig


def _cards_schema(vector_dim: int) -> pa.Schema:
    return pa.schema([
        pa.field("card_id", pa.utf8()),
        pa.field("card_type", pa.utf8()),
        pa.field("title", pa.utf8()),
        pa.field("domain", pa.utf8()),
        pa.field("tags", pa.utf8()),
        pa.field("vector", pa.list_(pa.float32(), vector_dim)),
    ])


class LanceIndex:
    """LanceDB vector index for semantic search over knowledge cards."""

    def __init__(
        self,
        db_path: Path | str | None = None,
        vector_dim: int = 384,
        table_name: str = "cards",
    ) -> None:
        if db_path is None:
            db_path = StorageConfig().lance_path
        self._db_path = Path(db_path)
        self._vector_dim = vector_dim
        self._table_name = table_name
        self._db: Optional[lancedb.DBConnection] = None
        self._table = None

    def _ensure_open(self) -> None:
        """Open database and create table if needed."""
        if self._db is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self._db_path))
            schema = _cards_schema(self._vector_dim)
            self._table = self._db.create_table(
                self._table_name, schema=schema, exist_ok=True
            )

    @property
    def table(self):
        self._ensure_open()
        return self._table

    def add(
        self,
        card_id: str,
        card_type: str,
        title: str,
        domain: str,
        tags: str,
        vector: np.ndarray | list[float],
    ) -> None:
        """Add a card embedding to the index."""
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        self.table.add([{
            "card_id": card_id,
            "card_type": card_type,
            "title": title,
            "domain": domain,
            "tags": tags,
            "vector": vector,
        }])

    def search(
        self,
        query_vector: np.ndarray | list[float],
        limit: int = 5,
        where: Optional[str] = None,
    ) -> list[dict]:
        """Search for similar cards by vector, returning card_id + distance."""
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        q = self.table.search(query_vector).limit(limit)
        if where:
            q = q.where(where)
        return q.to_list()

    def delete(self, card_id: str) -> None:
        """Remove a card from the index."""
        self.table.delete(f"card_id = '{card_id}'")

    def count(self) -> int:
        """Return the number of indexed cards."""
        return self.table.count_rows()
