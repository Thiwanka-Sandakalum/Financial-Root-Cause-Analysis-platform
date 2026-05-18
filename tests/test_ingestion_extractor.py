from src.config import Settings
from src.ingestion.extractor import _iter_batches, _normalize_items
from src.ingestion.models import ChunkRecord


class _Item:
    def __init__(self, type: str, canonical_name: str, raw_text: str, confidence: float) -> None:
        self.type = type
        self.canonical_name = canonical_name
        self.raw_text = raw_text
        self.confidence = confidence


def _settings() -> Settings:
    return Settings(
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_USERNAME="neo4j",
        NEO4J_PASSWORD="password",
        NEO4J_DATABASE="neo4j",
        GEMINI_USE_VERTEXAI="true",
        GOOGLE_CLOUD_PROJECT="demo-project",
    )


def test_normalize_items_clamps_confidence_and_marks_low_confidence() -> None:
    chunk = ChunkRecord(
        chunk_id="chunk_1",
        doc_id="doc_1",
        company_id="AAPL",
        text="Revenue increased",
        position=0,
        token_count=2,
        metadata={},
    )
    settings = _settings()

    items = [
        _Item("Metric", "Revenue", "Revenue increased", 1.5),
        _Item("Event", "Earnings Call", "Earnings call held", -0.5),
    ]

    records = _normalize_items(items, chunk, settings, fallback_type="Other")

    assert len(records) == 2
    assert records[0].confidence == 1.0
    assert records[1].confidence == 0.0
    assert records[1].metadata["is_low_confidence"] is True


def test_normalize_items_skips_empty_values() -> None:
    chunk = ChunkRecord(
        chunk_id="chunk_2",
        doc_id="doc_2",
        company_id="AAPL",
        text="Text",
        position=0,
        token_count=1,
        metadata={},
    )
    settings = _settings()

    items = [
        _Item("Entity", "", "Apple", 0.8),
        _Item("Entity", "Apple", "", 0.8),
    ]

    records = _normalize_items(items, chunk, settings, fallback_type="Entity")

    assert records == []


def test_iter_batches_splits_inputs_by_batch_size() -> None:
    inputs = [
        {"instruction": "x", "chunk_text": "a"},
        {"instruction": "x", "chunk_text": "b"},
        {"instruction": "x", "chunk_text": "c"},
        {"instruction": "x", "chunk_text": "d"},
        {"instruction": "x", "chunk_text": "e"},
    ]

    batches = _iter_batches(inputs, 2)

    assert len(batches) == 3
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 1
