"""Hybrid retriever combining vector and fulltext search from Neo4j."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from neo4j import Driver
from pydantic import ConfigDict, PrivateAttr

from src.config import Settings
from src.ingestion.chunker import BM25ChunkIndex
from src.llm.gemini import build_embedding_model


# Lucene reserved characters that must be escaped before passing a raw
# user query to db.index.fulltext.queryNodes.
# See: https://lucene.apache.org/core/9_0_0/queryparser/org/apache/lucene/queryparser/classic/package-summary.html
_LUCENE_SPECIAL_CHARS = re.compile(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)')

VECTOR_WEIGHT = 0.50
FULLTEXT_WEIGHT = 0.25
BM25_WEIGHT = 0.25
RRF_K = 60


def _sanitize_fulltext_query(query: str) -> str:
    """Escape Lucene special characters in a raw user query string."""
    return _LUCENE_SPECIAL_CHARS.sub(r"\\\1", query)


def _rrf_score(rank: int) -> float:
    return 1.0 / (RRF_K + rank)


class HybridRetriever(BaseRetriever):
    """Retriever that combines vector, fulltext, and BM25 search.
    
    Three-channel retrieval merged via weighted Reciprocal Rank Fusion (RRF):
    - Vector search: semantic relevance
    - Fulltext search: keyword/exact match
    - BM25 index: lexical + exact numeric matching
    """


    model_config = ConfigDict(arbitrary_types_allowed=True)
    driver: Driver
    settings: Settings
    _embedding_model: Any = PrivateAttr(default=None)

    def __init__(self, driver: Driver, settings: Settings) -> None:
        super().__init__(driver=driver, settings=settings)
        self._embedding_model = build_embedding_model(settings)

    def _get_relevant_documents(
        self,
        query: str,
        run_manager=None,
        **kwargs: Any,
    ) -> list[Document]:
        """Retrieve documents by blending vector and fulltext search.
        
        Args:
            query: User question or search text
            run_manager: LangChain run manager (unused)
            **kwargs: Additional arguments (unused)
            
        Returns:
            List of Document objects sorted by combined score (highest first)
        """
        top_k = int(kwargs.get("top_k", self.settings.query_retrieval_top_k))
        company_id = str(kwargs.get("company_id") or "").strip() or None
        period_key = str(kwargs.get("period_key") or "").strip() or None
        if period_key is None and kwargs.get("period_end"):
            period_key = str(kwargs["period_end"]).strip()
        
        # Three-channel retrieval.
        vector_results = self._vector_search(query, top_k)
        fulltext_results = self._fulltext_search(query, top_k)
        bm25_results = self._bm25_search(query, top_k, company_id, period_key)
        
        merged = self._rrf_merge(vector_results, fulltext_results, bm25_results)
        
        return merged

    def _bm25_search(
        self,
        query: str,
        top_k: int,
        company_id: str | None,
        period_key: str | None,
    ) -> list[Document]:
        """Retrieve chunks via persisted ingestion-time BM25 index."""
        if not company_id or not period_key:
            return []

        try:
            index = BM25ChunkIndex.load(company_id=company_id, period_key=period_key)
            rows = index.retrieve(query, top_k=top_k)
        except Exception as exc:
            raise RuntimeError(f"BM25 search failed: {exc}") from exc

        results: list[Document] = []
        for row in rows:
            results.append(
                Document(
                    page_content=row.get("text", "") or "",
                    metadata={
                        "chunk_id": row.get("chunk_id"),
                        "doc_id": row.get("doc_id"),
                        "company_id": company_id,
                        "section_title": row.get("section_title"),
                        "page_number": row.get("page_number"),
                        "chunk_type": row.get("chunk_type", "prose"),
                        "bm25_score": float(row.get("bm25_score", 0.0)),
                        "search_type": "bm25",
                    },
                )
            )
        return results

    def _vector_search(self, query: str, top_k: int) -> list[Document]:
        """Retrieve chunks via vector similarity search."""
        try:
            query_embedding = self._embedding_model.embed_query(query)
        except Exception as exc:
            raise RuntimeError(f"Failed to embed query: {exc}") from exc

        cypher = (
            "CALL db.index.vector.queryNodes('chunk_embedding_index', $top_k, $query_embedding) "
            "YIELD node, score "
            "WHERE node:Chunk "
            "RETURN "
            "  node.chunk_id AS chunk_id, "
            "  node.doc_id AS doc_id, "
            "  node.text AS text, "
            "  node.company_id AS company_id, "
            "  node.token_count AS token_count, "
            "  node.position AS position, "
            "  score AS vector_score"
        )

        results: list[Document] = []
        try:
            with self.driver.session(database=self.settings.neo4j_database) as session:
                records = session.run(cypher, top_k=top_k, query_embedding=query_embedding).fetch(top_k)
                
                for record in records:
                    doc = Document(
                        page_content=record["text"] or "",
                        metadata={
                            "chunk_id": record["chunk_id"],
                            "doc_id": record["doc_id"],
                            "company_id": record["company_id"],
                            "position": record["position"],
                            "vector_score": float(record["vector_score"]),
                            "search_type": "vector",
                        },
                    )
                    results.append(doc)
        except Exception as exc:
            raise RuntimeError(f"Vector search failed: {exc}") from exc

        return results

    def _fulltext_search(self, query: str, top_k: int) -> list[Document]:
        """Retrieve chunks via fulltext search."""
        safe_query = _sanitize_fulltext_query(query)
        cypher = (
            "CALL db.index.fulltext.queryNodes('chunk_fulltext_index', $query_text) "
            "YIELD node, score "
            "MATCH (node)-[:PART_OF]->(d:Document) "
            "RETURN "
            "  node.chunk_id AS chunk_id, "
            "  node.doc_id AS doc_id, "
            "  node.text AS text, "
            "  node.company_id AS company_id, "
            "  node.token_count AS token_count, "
            "  node.position AS position, "
            "  score AS fulltext_score "
            "ORDER BY score DESC "
            "LIMIT $top_k"
        )

        results: list[Document] = []
        try:
            with self.driver.session(database=self.settings.neo4j_database) as session:
                records = session.run(cypher, query_text=safe_query, top_k=top_k).fetch(top_k)
                
                for record in records:
                    doc = Document(
                        page_content=record["text"] or "",
                        metadata={
                            "chunk_id": record["chunk_id"],
                            "doc_id": record["doc_id"],
                            "company_id": record["company_id"],
                            "position": record["position"],
                            "fulltext_score": float(record["fulltext_score"]),
                            "search_type": "fulltext",
                        },
                    )
                    results.append(doc)
        except Exception as exc:
            raise RuntimeError(f"Fulltext search failed: {exc}") from exc

        return results

    def _rrf_merge(
        self,
        vector_results: list[Document],
        fulltext_results: list[Document],
        bm25_results: list[Document],
    ) -> list[Document]:
        """Merge three ranked lists using weighted Reciprocal Rank Fusion.
        
        Multi-channel evidence gets a +0.05 bonus when present in 2+ channels.
        """
        chunk_scores: dict[str, dict[str, Any]] = {}

        for rank, doc in enumerate(vector_results, start=1):
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id:
                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = {
                        "doc": doc,
                        "rrf": 0.0,
                        "channels": [],
                    }
                chunk_scores[chunk_id]["rrf"] += _rrf_score(rank) * VECTOR_WEIGHT
                chunk_scores[chunk_id]["channels"].append("vector")

        for rank, doc in enumerate(fulltext_results, start=1):
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id:
                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = {
                        "doc": doc,
                        "rrf": 0.0,
                        "channels": [],
                    }
                chunk_scores[chunk_id]["rrf"] += _rrf_score(rank) * FULLTEXT_WEIGHT
                chunk_scores[chunk_id]["channels"].append("fulltext")

        for rank, doc in enumerate(bm25_results, start=1):
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id:
                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = {
                        "doc": doc,
                        "rrf": 0.0,
                        "channels": [],
                    }
                chunk_scores[chunk_id]["rrf"] += _rrf_score(rank) * BM25_WEIGHT
                chunk_scores[chunk_id]["channels"].append("bm25")

        scored_chunks: list[tuple[str, float, Document, list[str]]] = []
        for chunk_id, scores in chunk_scores.items():
            channels = list(dict.fromkeys(scores["channels"]))
            combined_score = float(scores["rrf"])
            if len(channels) >= 2:
                combined_score = min(1.0, combined_score + 0.05)

            doc = scores["doc"]
            doc.metadata["combined_score"] = combined_score
            doc.metadata["retrieval_channels"] = channels
            scored_chunks.append((chunk_id, combined_score, doc, channels))

        # Sort by combined score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        return [doc for _, _, doc, _ in scored_chunks]

    # Backward-compatible alias used by existing unit tests.
    def _merge_results(
        self,
        vector_results: list[Document],
        fulltext_results: list[Document],
    ) -> list[Document]:
        return self._rrf_merge(vector_results, fulltext_results, [])
