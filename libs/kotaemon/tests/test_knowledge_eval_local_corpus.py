"""Generated-file tests for the local evaluation corpus inventory."""

from __future__ import annotations

import hashlib

from kotaemon.indices.knowledge.evaluation.local_corpus import scan_sources
from .knowledge_eval_test_fixtures import document, make_generated_corpus


def test_scan_sources_assigns_stable_ids_and_orders_paths(tmp_path):
    root = tmp_path / "sources"
    (root / "nested").mkdir(parents=True)
    files = {
        "zeta.md": b"zeta evidence",
        "alpha.docx": b"alpha evidence",
        "nested/report.pdf": b"report evidence",
        "table.xlsx": b"table evidence",
        "ignored.py": b"unsupported generated code",
    }
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    first = scan_sources(root)
    second = scan_sources(root)

    assert first == second
    assert [row.relative_path for row in first] == [
        "alpha.docx",
        "nested/report.pdf",
        "table.xlsx",
        "zeta.md",
    ]
    assert [row.suffix for row in first] == [".docx", ".pdf", ".xlsx", ".md"]
    assert [row.byte_size for row in first] == [
        len(files["alpha.docx"]),
        len(files["nested/report.pdf"]),
        len(files["table.xlsx"]),
        len(files["zeta.md"]),
    ]
    assert all(len(row.source_id) == 64 for row in first)
    assert all(int(row.source_id, 16) >= 0 for row in first)
    assert first[0].source_id == hashlib.sha256(files["alpha.docx"]).hexdigest()
    assert all(row.duplicate_of is None for row in first)


def test_scan_sources_deduplicates_bytes_and_excludes_lock_files(tmp_path):
    root = tmp_path / "sources"
    root.mkdir()
    (root / "a.md").write_bytes(b"same evidence")
    (root / "b.md").write_bytes(b"same evidence")
    (root / "c.md").write_bytes(b"same evidence\n")
    (root / "~$draft.docx").write_bytes(b"lock")
    (root / ".DS_Store").write_bytes(b"metadata")

    rows = scan_sources(root)

    assert [row.relative_path for row in rows] == ["a.md", "b.md", "c.md"]
    assert rows[0].source_id == rows[1].source_id
    assert rows[0].source_id != rows[2].source_id
    assert [row.duplicate_of for row in rows] == [None, "a.md", None]


def test_scan_sources_excludes_symlinks(tmp_path):
    root = tmp_path / "sources"
    root.mkdir()
    (root / "real.md").write_bytes(b"generated source")
    (root / "linked.md").symlink_to(root / "real.md")

    rows = scan_sources(root)

    assert [row.relative_path for row in rows] == ["real.md"]


def test_generated_corpus_and_document_fixtures_are_deterministic(tmp_path):
    item = document("fixture-doc", "Synthetic fixture text.")
    assert item.doc_id == "fixture-doc"
    assert item.text == "Synthetic fixture text."

    root = make_generated_corpus(tmp_path / "generated")
    rows = scan_sources(root)

    assert [row.relative_path for row in rows] == [
        "legacy/notes.md",
        "li/frontend.md",
        "wang/automation.md",
        "zhang/internship.md",
    ]
    assert all(row.duplicate_of is None for row in rows)
