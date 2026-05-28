from query.agent_query.tool_planner import build_default_tool_plan, normalize_tool_plan


def test_build_default_tool_plan_prefers_multiple_tools_for_complex_questions():
	intent = {
		"intent": "root_cause",
		"needs_tables": True,
		"needs_graph_traversal": True,
	}
	retrieval_plan = {
		"top_k_chunks": 10,
		"top_k_tables": 7,
		"hop_depth": 3,
		"use_company_filter": True,
		"use_causal_edges": True,
	}

	plan = build_default_tool_plan(intent, retrieval_plan)

	assert plan.selected_tools == ["chunk_search", "table_search", "graph_traversal"]
	assert plan.chunk_top_k == 10
	assert plan.table_top_k == 7
	assert plan.hop_depth == 3
	assert plan.use_company_filter is True
	assert plan.use_causal_edges is True


def test_normalize_tool_plan_filters_unknown_tools_and_keeps_order():
	intent = {"intent": "fact_lookup", "needs_tables": False, "needs_graph_traversal": False}
	retrieval_plan = {"top_k_chunks": 8, "top_k_tables": 0, "hop_depth": 2}
	payload = {
		"selected_tools": ["graph_traversal", "chunk_search", "graph_traversal", "bad_tool"],
		"chunk_top_k": 4,
		"table_top_k": 0,
		"hop_depth": 1,
		"use_company_filter": False,
		"use_causal_edges": False,
		"rationale": "Prefer graph traversal first.",
	}

	plan = normalize_tool_plan(payload, intent, retrieval_plan)

	assert plan.selected_tools == ["graph_traversal", "chunk_search"]
	assert plan.chunk_top_k == 4
	assert plan.table_top_k == 0
	assert plan.hop_depth == 1
	assert plan.use_company_filter is False
	assert plan.use_causal_edges is False
	assert plan.rationale == "Prefer graph traversal first."
