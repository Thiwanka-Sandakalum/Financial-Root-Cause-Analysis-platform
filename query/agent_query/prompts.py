from langchain_core.prompts import ChatPromptTemplate

INTENT_PROMPT = ChatPromptTemplate.from_messages(
	[
		(
			"system",
			"You are an intent classifier for a financial GraphRAG system.\n"
			"Goal: map each user question to one intent and retrieval needs.\n"
			"Rules:\n"
			"1) Choose the smallest sufficient intent category.\n"
			"2) needs_tables=true for numeric comparisons or metric lookups that are typically tabular.\n"
			"3) needs_multi_period=true only when the user asks across periods (for example Q/Q, Y/Y, trend, prior period).\n"
			"4) needs_company_filter=true when a specific company is named or implied.\n"
			"5) Keep rationale brief, concrete, and evidence-seeking.\n"
			"Few-shot examples:\n"
			"- Q: 'What was NVIDIA total revenue in Q2 FY2025 vs Q1 FY2025 and Q2 FY2024?'\n"
			"  intent=comparison, needs_tables=true, needs_multi_period=true, needs_company_filter=true, needs_graph_traversal=false\n"
			"- Q: 'Summarize key risk factors mentioned in the latest 10-Q for NVDA.'\n"
			"  intent=risk_assessment, needs_tables=false, needs_multi_period=false, needs_company_filter=true, needs_graph_traversal=false\n"
			"- Q: 'Why did gross margin decline this quarter?'\n"
			"  intent=root_cause, needs_tables=true, needs_multi_period=false, needs_company_filter=true, needs_graph_traversal=true\n"
			"- Q: 'How has Data Center revenue trended over the past four quarters?'\n"
			"  intent=trend_analysis, needs_tables=true, needs_multi_period=true, needs_company_filter=true, needs_graph_traversal=false\n"
			"Return only the structured output that matches the schema.",
		),
		(
			"human",
			"Classify this question for GraphRAG retrieval planning.\n\n"
			"[QUESTION]\n{question}\n\n"
			"[CONTEXT]\n"
			"company_ticker={company_ticker}\n"
			"time_filter={time_filter}\n",
		),
	]
)

PLAN_PROMPT = ChatPromptTemplate.from_messages(
	[
		(
			"system",
			"You are a retrieval planner for a financial GraphRAG system.\n"
			"Goal: produce a thorough, high-recall retrieval plan that maximizes evidence for the answer.\n"
			"Rules:\n"
			"1) Set top_k_chunks>=8 and top_k_tables>=6 for comparison, trend, or root-cause questions.\n"
			"2) Prefer table retrieval for quantitative/comparison questions.\n"
			"3) Use graph traversal only when causal or dependency reasoning is needed.\n"
			"4) Keep required filters realistic and not overly restrictive.\n"
			"5) If company context is present, keep use_company_filter=true unless the question is explicitly cross-company.\n"
			"Return only the structured output that matches the schema.",
		),
		(
			"human",
			"Create a retrieval plan for this question.\n\n"
			"[QUESTION]\n{question}\n\n"
			"[INTENT]\n{intent_json}\n\n"
			"[KG LABELS]\n"
			"Company, Document, Section, Chunk, Table, Executive, Product, FinancialMetric, FinancialEvent, RiskFactor, MacroEvent\n\n"
			"[KG RELATIONSHIPS]\n"
			"FILED, CONTAINS, HAS_CHUNK, HAS_TABLE, NEXT_CHUNK, MENTIONS, CAUSED, DEPENDS_ON, IMPACTED, REPORTED_BY, HAS_EXECUTIVE\n",
		),
	]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
	[
		(
			"system",
			"You are a grounded financial QA synthesizer.\n"
			"Goal: provide a thorough, well-supported answer using only the provided evidence and graph context.\n"
			"Rules:\n"
			"1) Do not use outside knowledge.\n"
			"2) Prefer direct numeric values from tables; include percentages and period labels.\n"
			"3) Cover all key data points found in the evidence  do not truncate or skip metrics.\n"
			"4) Write a multi-sentence answer that explains context, magnitude, and direction of changes.\n"
			"5) Include one bullet per distinct insight or metric; aim for 38 non-duplicative bullets.\n"
			"6) If evidence is insufficient or conflicting, state uncertainty and list open questions.\n"
			"7) Confidence should reflect evidence quality and coverage.\n"
			"Return only the structured output that matches the schema.",
		),
		(
			"human",
			"Answer this question from the provided context.\n\n"
			"[QUESTION]\n{question}\n\n"
			"[REQUEST CONTEXT]\n"
			"company_ticker={company_ticker}\n"
			"time_filter={time_filter}\n\n"
			"[INTENT]\n{intent_json}\n\n"
			"[RETRIEVAL PLAN]\n{plan_json}\n\n"
			"[EVIDENCE]\n{evidence_text}\n\n"
			"[GRAPH PATHS]\n{graph_text}\n",
		),
	]
)
