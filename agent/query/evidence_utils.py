from __future__ import annotations

import re


def state_dict(value: object) -> dict:
    if hasattr(value, "model_dump"):
        return value.model_dump()  # type: ignore[no-any-return]
    if isinstance(value, dict):
        return value
    return {}


def serialize_docs(docs: list[dict]) -> str:
    lines: list[str] = []
    for doc in docs:
        meta = doc.get("metadata", {})
        lines.append(
            f"[{meta.get('source_type')} | {meta.get('doc_type')} | {meta.get('period')} | "
            f"{meta.get('section_type')} | score={meta.get('score')}]\n{doc.get('page_content', '')}"
        )
    return "\n\n---\n\n".join(lines)


def serialize_graph_paths(paths: list[dict]) -> str:
    lines: list[str] = []
    for path in paths:
        lines.append(str(path))
    return "\n".join(lines)


def dedupe_documents(docs: list[dict]) -> list[dict]:
    seen: set[tuple[str | None, str | None]] = set()
    unique_docs: list[dict] = []

    for doc in docs:
        meta = doc.get("metadata", {})
        key = (meta.get("source_type"), meta.get("source_id"))
        if key in seen:
            continue
        seen.add(key)
        unique_docs.append(doc)

    unique_docs.sort(
        key=lambda doc: float(doc.get("metadata", {}).get("score", 0.0)), reverse=True
    )
    return unique_docs


def evidence_item(doc: dict) -> dict:
    meta = doc.get("metadata", {})
    return {
        "source_type": meta.get("source_type"),
        "source_id": meta.get("source_id"),
        "doc_id": meta.get("doc_id"),
        "doc_type": meta.get("doc_type"),
        "period": meta.get("period"),
        "section_id": meta.get("section_id"),
        "section_type": meta.get("section_type"),
        "page": meta.get("page"),
        "score": meta.get("score"),
        "ticker": meta.get("ticker"),
        "text": doc.get("page_content", ""),
    }


def table_has_multi_period_comparison(evidence: list[dict]) -> bool:
    period_pattern = re.compile(r"\bq[1-4]\s*fy\s*\d{2,4}\b", re.IGNORECASE)

    for item in evidence:
        if item.get("source_type") != "table":
            continue

        text = (item.get("text") or "").lower()
        if not text:
            continue

        period_tokens = {
            token.upper().replace(" ", "") for token in period_pattern.findall(text)
        }
        if len(period_tokens) >= 2:
            return True

        has_qoq = "q/q" in text or "previous quarter" in text
        has_yoy = "y/y" in text or "year ago" in text
        if has_qoq and has_yoy:
            return True

    return False
