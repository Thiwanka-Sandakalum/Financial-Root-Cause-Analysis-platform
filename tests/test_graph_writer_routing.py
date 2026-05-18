from src.ingestion.graph_writer import (
    _classify_record_type,
    _entity_extraction_params,
    _event_extraction_params,
    _metric_extraction_params,
    _normalize_company_identity,
)
from src.ingestion.models import ExtractionRecord


def _records() -> list[ExtractionRecord]:
    return [
        ExtractionRecord(
            entity_id="ent_1",
            entity_type="Company",
            canonical_name="Apple",
            raw_text="Apple",
            confidence=0.9,
            source_chunk_id="chunk_1",
        ),
        ExtractionRecord(
            entity_id="ent_2",
            entity_type="Event",
            canonical_name="Earnings Call",
            raw_text="earnings call",
            confidence=0.8,
            source_chunk_id="chunk_1",
        ),
        ExtractionRecord(
            entity_id="ent_3",
            entity_type="Revenue",
            canonical_name="Revenue",
            raw_text="revenue",
            confidence=0.85,
            source_chunk_id="chunk_1",
        ),
    ]


def test_record_type_classification() -> None:
    assert _classify_record_type("Revenue") == "metric"
    assert _classify_record_type("Event") == "event"
    assert _classify_record_type("Company") == "entity"


def test_extraction_param_routing() -> None:
    records = _records()

    entities = _entity_extraction_params(records)
    events = _event_extraction_params(records)
    metrics = _metric_extraction_params(records)

    assert len(entities) == 1
    assert len(events) == 1
    assert len(metrics) == 1
    assert entities[0]["entity_id"] == "ent_1"
    assert events[0]["entity_id"] == "ent_2"
    assert metrics[0]["entity_id"] == "ent_3"


def test_normalize_company_identity_collapses_corporate_suffixes() -> None:
    assert _normalize_company_identity("NVIDIA") == "nvidia"
    assert _normalize_company_identity("Nvidia") == "nvidia"
    assert _normalize_company_identity("NVIDIA CORPORATION") == "nvidia"


def test_normalize_company_identity_handles_empty_values() -> None:
    assert _normalize_company_identity("") == ""
    assert _normalize_company_identity("   ") == ""
