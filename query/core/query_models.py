from __future__ import annotations

from typing import Annotated, Any, Literal

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

RetrievalToolName = Literal[
	"chunk_search",
	"table_search",
	"graph_traversal",
]

VisualizationChartType = Literal[
	"line",
	"grouped_bar",
]


class QueryIntent(BaseModel):
	intent: IntentType = Field(description="Primary question type")
	needs_tables: bool = Field(description="Whether table evidence is needed")
	needs_graph_traversal: bool = Field(description="Whether graph expansion is needed")
	needs_multi_period: bool = Field(
		description="Whether multiple periods are required"
	)
	needs_company_filter: bool = Field(
		description="Whether the query should filter to one company"
	)
	confidence: float = Field(
		ge=0.0, le=1.0, description="Intent classification confidence"
	)
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


class RetrievalToolPlan(BaseModel):
	selected_tools: list[RetrievalToolName] = Field(default_factory=list)
	chunk_top_k: int = Field(default=8, ge=1, le=20)
	table_top_k: int = Field(default=6, ge=0, le=20)
	hop_depth: int = Field(default=2, ge=0, le=4)
	use_company_filter: bool = Field(default=True)
	use_causal_edges: bool = Field(default=True)
	rationale: str = Field(default="")


class DomainDecision(BaseModel):
	domain_status: Literal["financial_supported", "non_financial", "uncertain"] = Field(
		default="uncertain",
		description="Whether the request is in supported financial domain",
	)
	domain_reason: str = Field(
		default="",
		description="Short reason for the domain decision",
	)
	domain_confidence: Literal["low", "medium", "high"] = Field(
		default="medium",
		description="Confidence in domain classification",
	)
	domain_signals: list[str] = Field(
		default_factory=list,
		description="Optional cues used for the domain decision",
	)
	clarification_question: str | None = Field(
		default=None,
		description="Optional clarifying question for uncertain requests",
	)


class QueryAnalysis(BaseModel):
	question: str = Field(description="Normalized user question")
	company_ticker: str | None = Field(
		default=None,
		description="Selected company ticker from the provided catalog when available",
	)
	intent: QueryIntent
	retrieval_plan: RetrievalPlan


class ReadinessDecision(BaseModel):
	answer_mode: Literal["answer", "clarify", "request_ingestion"] = Field(
		default="answer",
		description="Whether to proceed, ask clarification, or request new ingestion",
	)
	reason: str = Field(default="", description="Short explanation for the readiness decision")
	missing_slots: list[str] = Field(
		default_factory=list,
		description="Missing slots such as year, period, metric, entity, source",
	)
	clarification_questions: list[str] = Field(
		default_factory=list,
		description="Targeted follow-up questions when answer_mode is clarify",
	)
	ingest_recommendations: list[str] = Field(
		default_factory=list,
		description="Specific document/data ingestion recommendations",
	)


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
			"driver, or insight found in the evidence. Aim for 3-8 bullets. "
			"Each bullet should be a complete, standalone fact with numbers where available."
		),
	)
	citations: list[Citation] = Field(default_factory=list)
	confidence: Literal["low", "medium", "high"] = Field(default="medium")
	open_questions: list[str] = Field(default_factory=list)


class VisualizationSpec(BaseModel):
	enabled: bool = Field(
		default=False,
		description="Whether frontend should render a visualization card",
	)
	reason: str = Field(default="", description="Reason for enabling or skipping visualization")
	chart_type: VisualizationChartType | None = Field(default=None)
	title: str | None = Field(default=None)
	x_field: str | None = Field(default=None)
	y_field: str | None = Field(default=None)
	series: list[str] = Field(default_factory=list)
	unit: str | None = Field(default=None)
	data: list[dict[str, Any]] = Field(default_factory=list)
	insight: str | None = Field(default=None)
	citations: list[Citation] = Field(default_factory=list)


class QueryState(TypedDict, total=False):
	messages: Annotated[list[AnyMessage], add_messages]
	question: str
	company_ticker: str | None
	time_filter: str | None
	intent: QueryIntent | dict[str, Any]
	retrieval_plan: RetrievalPlan | dict[str, Any]
	readiness_decision: ReadinessDecision | dict[str, Any]
	tool_plan: RetrievalToolPlan | dict[str, Any]
	chunk_hits: list[dict[str, Any]]
	table_hits: list[dict[str, Any]]
	graph_paths: list[dict[str, Any]]
	evidence: list[dict[str, Any]]
	final_answer: FinalAnswer | dict[str, Any]
	retry_count: int
	error_class: str | None
	blocked_reason: str | None
	domain_status: str | None
	policy_flags: dict[str, Any]
	retrieval_quality: dict[str, Any]
	evaluation_signals: dict[str, Any]
	claim_support_ratio: float
	claim_total_count: int
	unsupported_claim_count: int
	confidence_overridden: bool
	visualization: VisualizationSpec | dict[str, Any]
