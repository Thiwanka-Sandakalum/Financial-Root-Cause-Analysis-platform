from __future__ import annotations

from query.core.query_models import RetrievalToolPlan


GRAPH_MODEL_CONTEXT = (
	"Company FILED Document; Document CONTAINS Section; Section HAS_CHUNK Chunk and HAS_TABLE Table; "
	"Chunk NEXT_CHUNK Chunk; Chunk MENTIONS Company, FinancialMetric, FinancialEvent, Executive, Product, "
	"MacroEvent, RiskFactor; causal edges include CAUSED, IMPACTED, DEPENDS_ON, REPORTED_BY, HAS_EXECUTIVE."
)

ALLOWED_TOOL_NAMES = {"chunk_search", "table_search", "graph_traversal"}


def build_default_tool_plan(intent: dict, retrieval_plan: dict) -> RetrievalToolPlan:
	selected_tools: list[str] = ["chunk_search"]

	if intent.get("needs_tables") or retrieval_plan.get("top_k_tables", 0):
		selected_tools.append("table_search")

	if intent.get("needs_graph_traversal") or intent.get("intent") in {
		"root_cause",
		"comparison",
		"trend_analysis",
		"risk_assessment",
	}:
		selected_tools.append("graph_traversal")

	unique_tools = [tool for tool in selected_tools if tool in ALLOWED_TOOL_NAMES]
	ordered_tools = list(dict.fromkeys(unique_tools))

	return RetrievalToolPlan(
		selected_tools=ordered_tools,
		chunk_top_k=int(retrieval_plan.get("top_k_chunks", 8) or 8),
		table_top_k=int(retrieval_plan.get("top_k_tables", 6) or 6),
		hop_depth=int(retrieval_plan.get("hop_depth", 2) or 2),
		use_company_filter=bool(retrieval_plan.get("use_company_filter", True)),
		use_causal_edges=bool(retrieval_plan.get("use_causal_edges", True)),
		rationale="Heuristic fallback based on the intent and retrieval plan.",
	)


def normalize_tool_plan(
	payload: dict,
	intent: dict,
	retrieval_plan: dict,
) -> RetrievalToolPlan:
	selected_tools = payload.get("selected_tools") or []
	selected_tools = [tool for tool in selected_tools if tool in ALLOWED_TOOL_NAMES]
	if not selected_tools:
		return build_default_tool_plan(intent, retrieval_plan)

	plan_data = {
		"selected_tools": list(dict.fromkeys(selected_tools)),
		"chunk_top_k": payload.get("chunk_top_k", retrieval_plan.get("top_k_chunks", 8)),
		"table_top_k": payload.get("table_top_k", retrieval_plan.get("top_k_tables", 6)),
		"hop_depth": payload.get("hop_depth", retrieval_plan.get("hop_depth", 2)),
		"use_company_filter": payload.get(
			"use_company_filter", retrieval_plan.get("use_company_filter", True)
		),
		"use_causal_edges": payload.get(
			"use_causal_edges", retrieval_plan.get("use_causal_edges", True)
		),
		"rationale": payload.get("rationale") or payload.get("reason") or "",
	}
	return RetrievalToolPlan.model_validate(plan_data)