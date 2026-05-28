from __future__ import annotations

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
