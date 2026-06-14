from __future__ import annotations

from typing import Any

from langsmith import traceable
from neo4j import Driver
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem


def _normalize_text(value: str | None) -> str:
	return (value or "").strip()


def _retriever_doc(source_type: str, source_id: str, text: str, metadata: dict) -> dict:
	payload = dict(metadata)
	payload["source_type"] = source_type
	payload["source_id"] = source_id
	return {
		"page_content": text,
		"type": "Document",
		"metadata": payload,
	}


def _record_to_result_item(record: Any) -> RetrieverResultItem:
	metadata = record.get("metadata")
	if not isinstance(metadata, dict):
		metadata = {}
	content = record.get("content")
	if not isinstance(content, str):
		content = str(content or "")
	return RetrieverResultItem(content=content, metadata=metadata)


_CHUNK_VECTOR_CYPHER_RETRIEVER: VectorCypherRetriever | None = None
_TABLE_VECTOR_CYPHER_RETRIEVER: VectorCypherRetriever | None = None


_CHUNK_VECTOR_CYPHER_QUERY = """
OPTIONAL MATCH (s:Section)-[:HAS_CHUNK]->(node)
OPTIONAL MATCH (d:Document)-[:CONTAINS]->(s)
OPTIONAL MATCH (c:Company)-[:FILED]->(d)
OPTIONAL MATCH (prev:Chunk)-[:NEXT_CHUNK]->(node)
OPTIONAL MATCH (node)-[:NEXT_CHUNK]->(next:Chunk)
OPTIONAL MATCH (node)-[:MENTIONS]->(entity)
OPTIONAL MATCH (entity)-[rel:CAUSED|IMPACTED|DEPENDS_ON|REPORTED_BY|HAS_EXECUTIVE]->(other)
WITH node, score, s, d, c, prev, next, entity, rel, other
WHERE ($ticker IS NULL OR toUpper(c.ticker) = toUpper($ticker))
  AND ($time_filter IS NULL OR d.period CONTAINS $time_filter)
WITH
	node,
	score,
	s,
	d,
	c,
	collect(DISTINCT prev.id) AS prev_chunk_ids,
	collect(DISTINCT next.id) AS next_chunk_ids,
	collect(DISTINCT entity.name) AS entity_names,
	collect(
		DISTINCT CASE
			WHEN $use_causal_edges THEN type(rel) + ':' + coalesce(other.name, other.id, '')
			ELSE NULL
		END
	) AS causal_link_names
RETURN
	coalesce(node.text, '') AS content,
	{
		source_type: 'chunk',
		source_id: node.id,
		page: node.page,
		section_type: node.section_type,
		section_id: s.id,
		section_title: s.title,
		doc_id: d.id,
		doc_type: d.type,
		period: d.period,
		ticker: c.ticker,
		company: c.name,
		prev_chunks: [x IN prev_chunk_ids WHERE x IS NOT NULL],
		next_chunks: [x IN next_chunk_ids WHERE x IS NOT NULL],
		entities: [x IN entity_names WHERE x IS NOT NULL],
		causal_links: [x IN causal_link_names WHERE x IS NOT NULL],
		score: score,
		time_filter: $time_filter
	} AS metadata
"""


_TABLE_VECTOR_CYPHER_QUERY = """
OPTIONAL MATCH (s:Section)-[:HAS_TABLE]->(node)
OPTIONAL MATCH (d:Document)-[:CONTAINS]->(s)
OPTIONAL MATCH (c:Company)-[:FILED]->(d)
WITH node, score, s, d, c
WHERE ($ticker IS NULL OR toUpper(c.ticker) = toUpper($ticker))
  AND ($time_filter IS NULL OR d.period CONTAINS $time_filter)
RETURN
	coalesce(node.markdown, '') AS content,
	{
		source_type: 'table',
		source_id: node.id,
		sequence: node.sequence,
		section_type: s.section_type,
		section_id: s.id,
		section_title: s.title,
		doc_id: d.id,
		doc_type: d.type,
		period: d.period,
		ticker: c.ticker,
		company: c.name,
		score: score,
		time_filter: $time_filter
	} AS metadata
"""


def _build_chunk_retriever(driver: Driver) -> VectorCypherRetriever:
	global _CHUNK_VECTOR_CYPHER_RETRIEVER
	if _CHUNK_VECTOR_CYPHER_RETRIEVER is None:
		_CHUNK_VECTOR_CYPHER_RETRIEVER = VectorCypherRetriever(
			driver=driver,
			index_name="chunk_embeddings",
			retrieval_query=_CHUNK_VECTOR_CYPHER_QUERY,
			result_formatter=_record_to_result_item,
		)
	return _CHUNK_VECTOR_CYPHER_RETRIEVER


def _build_table_retriever(driver: Driver) -> VectorCypherRetriever:
	global _TABLE_VECTOR_CYPHER_RETRIEVER
	if _TABLE_VECTOR_CYPHER_RETRIEVER is None:
		_TABLE_VECTOR_CYPHER_RETRIEVER = VectorCypherRetriever(
			driver=driver,
			index_name="table_embeddings",
			retrieval_query=_TABLE_VECTOR_CYPHER_QUERY,
			result_formatter=_record_to_result_item,
		)
	return _TABLE_VECTOR_CYPHER_RETRIEVER


@traceable(run_type="retriever", name="retrieve_chunk_documents")
def retrieve_chunk_documents(
	driver: Driver,
	embedding: list[float],
	top_k: int = 6,
	company_ticker: str | None = None,
	time_filter: str | None = None,
	use_causal_edges: bool = True,
) -> list[dict]:
	retriever = _build_chunk_retriever(driver)
	result = retriever.search(
		query_vector=embedding,
		top_k=top_k,
		query_params={
			"ticker": _normalize_text(company_ticker) or None,
			"time_filter": _normalize_text(time_filter) or None,
			"use_causal_edges": use_causal_edges,
		},
	)

	docs: list[dict] = []
	for item in result.items:
		metadata = item.metadata if isinstance(item.metadata, dict) else {}
		source_id = str(metadata.get("source_id") or "").strip()
		if not source_id:
			continue
		docs.append(_retriever_doc("chunk", source_id, str(item.content or ""), metadata))
	return docs


@traceable(run_type="retriever", name="retrieve_table_documents")
def retrieve_table_documents(
	driver: Driver,
	embedding: list[float],
	top_k: int = 4,
	company_ticker: str | None = None,
	time_filter: str | None = None,
) -> list[dict]:
	retriever = _build_table_retriever(driver)
	result = retriever.search(
		query_vector=embedding,
		top_k=top_k,
		query_params={
			"ticker": _normalize_text(company_ticker) or None,
			"time_filter": _normalize_text(time_filter) or None,
		},
	)

	docs: list[dict] = []
	for item in result.items:
		metadata = item.metadata if isinstance(item.metadata, dict) else {}
		source_id = str(metadata.get("source_id") or "").strip()
		if not source_id:
			continue
		docs.append(_retriever_doc("table", source_id, str(item.content or ""), metadata))
	return docs
