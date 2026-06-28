"""Tests for the RAG knowledge base module."""

from __future__ import annotations

from pathlib import Path

import pytest

from taiji.tools.rag import CHUNK_OVERLAP, CHUNK_SIZE, RAGKnowledgeBase

SAMPLE_TEXT = (
    "Artificial intelligence is a field of computer science focused on building systems "
    "that can reason, learn, and respond to the world.\n\n"
    "Deep learning is a subset of machine learning. It often relies on layered neural "
    "networks to learn useful representations from data.\n\n"
    "Python is widely used for data processing, machine learning, and automation. "
    "It is also convenient for research tooling and quick experiments."
)


class TestRAGInit:
    def test_init_without_persist(self) -> None:
        kb = RAGKnowledgeBase()
        assert kb.documents == {}
        assert kb.chunks == []
        assert kb.embeddings is None

    def test_init_with_empty_persist_dir(self, tmp_path: Path) -> None:
        kb = RAGKnowledgeBase(persist_dir=str(tmp_path))
        assert kb.documents == {}
        assert kb.persist_dir == str(tmp_path)


class TestRAGChunking:
    def test_chunk_text_empty(self) -> None:
        assert RAGKnowledgeBase._chunk_text("") == []

    def test_chunk_text_whitespace_only(self) -> None:
        assert RAGKnowledgeBase._chunk_text("   \n  \n  ") == []

    def test_chunk_text_short(self) -> None:
        result = RAGKnowledgeBase._chunk_text("short text", chunk_size=200, overlap=50)
        assert result == ["short text"]

    def test_chunk_text_produces_multiple_chunks(self) -> None:
        result = RAGKnowledgeBase._chunk_text(SAMPLE_TEXT, chunk_size=100, overlap=20)
        assert len(result) > 1
        assert all(chunk.strip() for chunk in result)

    def test_chunk_text_uses_overlap(self) -> None:
        result = RAGKnowledgeBase._chunk_text(SAMPLE_TEXT, chunk_size=90, overlap=12)
        assert len(result) > 1
        assert result[1].startswith(result[0][-12:])

    def test_chunk_text_can_disable_overlap(self) -> None:
        result = RAGKnowledgeBase._chunk_text(SAMPLE_TEXT, chunk_size=90, overlap=0)
        assert len(result) > 1
        assert not result[1].startswith(result[0][-min(12, len(result[0])) :])


class TestRAGDocumentManagement:
    def test_add_text_and_get_doc_names(self) -> None:
        kb = RAGKnowledgeBase()
        kb.add_text("test.txt", "This is test content.")
        assert "test.txt" in kb.get_doc_names()
        assert kb.documents["test.txt"] == "This is test content."

    def test_add_empty_text(self) -> None:
        kb = RAGKnowledgeBase()
        kb.add_text("empty.txt", "  ")
        assert "empty.txt" not in kb.documents

    def test_add_duplicate_overwrites(self) -> None:
        kb = RAGKnowledgeBase()
        kb.add_text("doc.txt", "version1")
        kb.add_text("doc.txt", "version2")
        assert kb.documents["doc.txt"] == "version2"

    def test_remove_file(self) -> None:
        kb = RAGKnowledgeBase()
        kb.add_text("doc1.txt", "content1")
        kb.add_text("doc2.txt", "content2")
        kb.remove_file("doc1.txt")
        assert "doc1.txt" not in kb.documents
        assert "doc2.txt" in kb.documents

    def test_remove_nonexistent_no_error(self) -> None:
        kb = RAGKnowledgeBase()
        kb.remove_file("missing.txt")

    def test_clear(self) -> None:
        kb = RAGKnowledgeBase()
        kb.add_text("doc.txt", "content")
        kb.clear()
        assert kb.documents == {}
        assert kb.chunks == []
        assert kb.embeddings is None

    def test_add_file_not_found(self) -> None:
        kb = RAGKnowledgeBase()
        with pytest.raises(FileNotFoundError):
            kb.add_file("missing-file-does-not-exist.txt")


class TestRAGSearchEmpty:
    def test_search_empty_kb(self) -> None:
        kb = RAGKnowledgeBase()
        assert kb.search("artificial intelligence") == []

    def test_search_with_fallback_empty(self) -> None:
        kb = RAGKnowledgeBase()
        assert kb.search_with_fallback("test") == []

    def test_rebuild_index_empty(self) -> None:
        kb = RAGKnowledgeBase()
        message = kb.rebuild_index()
        assert "empty" in message.lower()


class TestRAGPersistence:
    def test_save_and_load_cycle(self, tmp_path: Path) -> None:
        kb1 = RAGKnowledgeBase(persist_dir=str(tmp_path))
        kb1.add_text("doc.txt", SAMPLE_TEXT)
        result = kb1.rebuild_index()

        kb2 = RAGKnowledgeBase(persist_dir=str(tmp_path))
        assert "doc.txt" in kb2.documents
        assert kb2.documents["doc.txt"] == SAMPLE_TEXT
        assert len(kb2.chunks) > 0
        assert isinstance(result, str)

    def test_set_persist_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "kb"
        kb = RAGKnowledgeBase()
        kb.set_persist_dir(str(target))
        assert kb.persist_dir == str(target)
        assert target.exists()


def test_module_constants_are_stable() -> None:
    assert CHUNK_SIZE == 200
    assert CHUNK_OVERLAP == 50
