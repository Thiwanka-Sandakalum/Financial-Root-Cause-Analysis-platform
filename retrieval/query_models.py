from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


IntentType = Literal[
    "fact_lookup",
    "trend_analysis",
    "root_cause",
    "risk_assessment",
    "comparison",
    "follow_up",
]


class QueryIntent(BaseModel):
    intent: IntentType = Field(description="Primary question type")
    needs_tables: bool = Field(description="Whether table evidence is needed")
    needs_graph_traversal: bool = Field(description="Whether graph expansion is needed")
    needs_multi_period: bool = Field(description="Whether multiple periods are required")
    needs_company_filter: bool = Field(description="Whether the query should filter to one company")
    confidence: float = Field(ge=0.0, le=1.0, description="Intent classification confidence")
    rationale: str = Field(description="Short explanation for the classification")


class RetrievalPlan(BaseModel):
    top_k_chunks: int = Field(default=8, ge=1, le=20)
    top_k_tables: int = Field(default=6, ge=0, le=20)
    hop_depth: int = Field(default=2, ge=0, le=4)
    required_section_types: list[str] = Field(default_factory=list)
    required_doc_types: list[str] = Field(default_factory=list)
    use_causal_edges: bool = Field(default=True)
    use_prev_next_chunks: bool = Field(default=True)
    use_company_filter: bool = Field(default=True)


class Citation(BaseModel):
    source_type: Literal["chunk", "table", "graph"]
    source_id: str
    doc_id: str | None = None
    section_id: str | None = None
    page: int | None = None
    score: float | None = None
    title: str | None = None


class FinalAnswer(BaseModel):
    answer: str = Field(
        description=(
            "Comprehensive answer grounded in the evidence. "
            "Include all relevant numbers, percentages, period comparisons, and key drivers "
            "mentioned in the evidence. Write multiple sentences. Do not abbreviate or summarize excessively."
        )
    )
    bullets: list[str] = Field(
        default_factory=list,
        description=(
            "Detailed supporting bullet points. Include one bullet per key data point, "
            "driver, or insight found in the evidence. Aim for 3–8 bullets. "
            "Each bullet should be a complete, standalone fact with numbers where available."
        ),
    )
    citations: list[Citation] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = Field(default="medium")
    open_questions: list[str] = Field(default_factory=list)


class QueryState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    company_ticker: str | None
    time_filter: str | None
    intent: dict
    retrieval_plan: dict
    chunk_hits: list[dict]
    table_hits: list[dict]
    graph_paths: list[dict]
    evidence: list[dict]
    gaps: list[str]
    final_answer: dict
