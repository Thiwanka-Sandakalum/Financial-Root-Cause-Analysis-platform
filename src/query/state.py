"""Query pipeline state definitions.

Follows the LangGraph canonical pattern: state lives in its own module,
separate from node logic and graph construction.
"""

from __future__ import annotations

from typing import Any, Annotated, Sequence

from langchain_core.documents import Document
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from neo4j import Driver

from src.config import Settings
from src.db.neo4j_client import Neo4jClient
from src.query.query_classifier import QueryIntent


class RaptorSection(TypedDict, total=False):
    section_id: str
    title: str
    summary: str
    score: float


class QueryWarning(TypedDict):
    stage: str
    message: str


class InputState(TypedDict, total=False):
    """User-facing input schema: minimal and explicit.

    Callers provide only these fields to start a query.  Internal working
    channels are hidden from the caller.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    company_id: str | None
    period_start: str | None
    period_end: str | None


class InternalState(TypedDict, total=False):
    """Internal working state: extends InputState with all intermediate channels.

    Managed by the graph during execution.  Includes InputState fields plus all
    working channels (intent, coverage, retrieval, synthesis, grounding, response).
    """

    # Input fields (mirrored from InputState)
    messages: Annotated[list[BaseMessage], add_messages]
    company_id: str | None
    period_start: str | None
    period_end: str | None

    # Cached query text extracted once from messages by the classify node
    query_text: str

    # Query classification
    intent: QueryIntent

    # Coverage check
    coverage_ok: bool
    missing_data_msg: str
    warnings: list[QueryWarning]

    # Retrieval and expansion
    candidate_chunks: list[Document]
    retrieved_chunks: list[Document]
    expanded_chunks: list[Document]
    raptor_sections: list[RaptorSection]
    causal_evidence: list[Document]
    ranked_evidence: list[Document]

    # Synthesis
    answer: str
    citations: list[dict[str, Any]]
    synthesis_confidence: float
    synthesis_reasoning: str

    # Grounding validation
    is_grounded: bool
    unsupported_claims: list[str]
    grounding_confidence: float

    # Final response (assembled by response node)
    final_response: dict[str, Any]


class OutputState(TypedDict, total=False):
    """Final output schema: minimal and focused on delivery.

    Contains only the final response and updated messages.
    """

    final_response: dict[str, Any]
    messages: Annotated[list[BaseMessage], add_messages]


# Backward-compatibility alias used in node type annotations
QueryState = InternalState


class QueryPipeline:
    """Owns and manages the compiled query graph and Neo4j driver lifecycle.

    Creates a single compiled graph instance on initialization and reuses it
    across all invocations, eliminating per-request compilation overhead.
    Implements context manager protocol for proper resource cleanup.
    """

    def __init__(self, settings: Settings, driver: Driver | Neo4jClient):
        self.settings = settings
        self.driver = driver
        self._owns_client = isinstance(driver, Neo4jClient)
        self._compiled_graph = None

    @property
    def compiled_graph(self):
        """Lazily compile and cache the graph on first access."""
        if self._compiled_graph is None:
            # Import here to avoid circular imports
            from src.query.orchestration import build_query_graph

            self._compiled_graph = build_query_graph(self.settings, self.driver)
        return self._compiled_graph

    def invoke(
        self,
        messages: Sequence[BaseMessage],
        company_id: str | None,
        period_start: str | None,
        period_end: str | None,
    ) -> dict[str, Any]:
        """Run query through the compiled pipeline."""
        result = self.compiled_graph.invoke(
            {
                "messages": list(messages),
                "company_id": company_id,
                "period_start": period_start,
                "period_end": period_end,
            }
        )
        return result.get("final_response", {"status": "failed", "answer": ""})

    def close(self) -> None:
        """Close Neo4j client if owned by this pipeline."""
        if self._owns_client and isinstance(self.driver, Neo4jClient):
            self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
