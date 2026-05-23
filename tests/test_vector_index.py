from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from system.ai.research_brief import build_research_brief
from system.ai.vector_index import build_vector_index, initialize_vector_schema, search_vectors


class VectorIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.db_path = self.root / "test.db"
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        schema = Path(__file__).resolve().parents[1] / "system" / "schema.sql"
        self.conn.executescript(schema.read_text(encoding="utf-8"))
        initialize_vector_schema(self.conn)
        self.conn.execute(
            """
            INSERT INTO ingestion_runs (run_id, run_date, status, started_at, finished_at, notes)
            VALUES (1, '2026-05-21', 'completed', '2026-05-21T00:00:00', '2026-05-21T00:01:00', '')
            """
        )
        self._insert_document(
            1,
            "tickertape",
            "Reliance Industries refinery margins and telecom subscriber growth improved.",
        )
        self._insert_document(
            2,
            "news",
            "A banking sector update covered credit growth and asset quality trends.",
        )
        self.conn.commit()

    def tearDown(self) -> None:
        self.conn.close()
        self.tempdir.cleanup()

    def _insert_document(self, document_id: int, source_name: str, content: str) -> None:
        self.conn.execute(
            """
            INSERT INTO raw_documents (
                document_id, run_id, source_name, file_path, file_type, content,
                content_sha256, record_count, source_timestamp, ingested_at
            )
            VALUES (?, 1, ?, ?, 'json', ?, ?, 1, '2026-05-21T00:00:00', '2026-05-21T00:00:00')
            """,
            (document_id, source_name, f"/tmp/{document_id}.json", content, f"hash-{document_id}"),
        )

    def test_build_and_search_local_vector_index(self) -> None:
        state = build_vector_index(self.conn, self.root / "vector_indexes", run_id=1, embedding_dim=64, limit=10)

        self.assertEqual(state["run_id"], 1)
        self.assertEqual(state["item_count"], 2)
        self.assertIn(state["backend"], {"exact", "turbovec"})

        result = search_vectors(self.conn, "Reliance refinery telecom", self.root / "vector_indexes", run_id=1, k=1)

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["results"][0]["document_id"], 1)

    def test_research_brief_auto_builds_vector_index(self) -> None:
        result = build_research_brief(
            self.conn,
            "RELIANCE",
            self.root / "vector_indexes",
            focus="refinery telecom",
            run_id=1,
            evidence_limit=1,
        )

        self.assertEqual(result["ticker"], "RELIANCE")
        self.assertEqual(result["retrieval"]["count"], 1)
        self.assertEqual(result["evidence"][0]["document_id"], 1)
        self.assertIn("brief_markdown", result)


if __name__ == "__main__":
    unittest.main()
