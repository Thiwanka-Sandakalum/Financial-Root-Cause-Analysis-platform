from langchain_core.prompts import ChatPromptTemplate

DOMAIN_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages(
	[
		(
			"system",
			"You are a strict domain classifier for a financial filings assistant.\n"
			"Task: classify whether a user request is in-domain.\n"
			"In-domain includes:\n"
			"- financial metrics and ratios (revenue, margins, EPS, cash flow)\n"
			"- filing disclosures, outlooks, periods (Q1-Q4, FY, guidance)\n"
			"- shareholder actions (dividends, buybacks)\n"
			"- qualitative management commentary tied to financial filings/results\n"
			"- business/platform disclosures in filings (product/platform introductions, partner collaborations, commercialization initiatives)\n"
			"- balance-sheet and obligation disclosures (purchase obligations, cloud agreements, commitments, per-share adjustments, corporate actions)\n"
			"Out-of-domain includes unrelated lifestyle/general topics even if a company is mentioned.\n"
			"If a question asks about details explicitly disclosed in a company filing or earnings release, classify as financial_supported.\n"
			"If ambiguous, set domain_status=uncertain and provide one clarifying question.\n"
			"Return only the structured output that matches the schema.",
		),
		(
			"human",
			"Classify this request.\n\n"
			"[RAW QUESTION]\n{question}\n\n"
			"[EXPLICIT CONTEXT]\n"
			"company_ticker={company_ticker}\n"
			"time_filter={time_filter}\n\n"
			"[COMPANY CATALOG]\n{company_catalog}\n",
		),
	]
)

DOMAIN_ADJUDICATE_PROMPT = ChatPromptTemplate.from_messages(
	[
		(
			"system",
			"You are a second-pass adjudicator for low-confidence or potentially incorrect domain decisions in a financial filings assistant.\n"
			"Given an initial decision, choose one final domain_status:\n"
			"- financial_supported when the request can be answered from filing or financial evidence\n"
			"- non_financial when clearly unrelated\n"
			"- uncertain only when genuinely underspecified\n"
			"Prefer financial_supported if the request references filing outlook, margins, revenue, dividend, business platform performance, obligations/commitments, per-share adjustments, stock split/corporate actions, or management commentary from results.\n"
			"Only keep non_financial when the request is clearly unrelated to company filings, financial performance, or disclosed corporate events.\n"
			"Return only the structured output that matches the schema.",
		),
		(
			"human",
			"Adjudicate this request.\n\n"
			"[RAW QUESTION]\n{question}\n\n"
			"[INITIAL DECISION]\n"
			"domain_status={initial_domain_status}\n"
			"domain_reason={initial_domain_reason}\n"
			"domain_confidence={initial_domain_confidence}\n\n"
			"[EXPLICIT CONTEXT]\n"
			"company_ticker={company_ticker}\n"
			"time_filter={time_filter}\n\n"
			"[COMPANY CATALOG]\n{company_catalog}\n",
		),
	]
)

ANALYZE_PROMPT = ChatPromptTemplate.from_messages(
	[
		(
			"system",
			"You are a query analysis planner for a financial GraphRAG system.\n"
			"Goal: normalize the question, map company mentions to a ticker, classify intent, and produce a retrieval plan for an in-domain financial request.\n"
			"Rules:\n"
			"1) Normalize the question text into a concise, clean user intent preserving meaning.\n"
			"2) Use the provided company catalog to map company names to ticker symbols.\n"
			"3) If an explicit ticker is already provided, keep it unless it clearly conflicts with the question.\n"
			"4) If no reliable ticker match exists, set company_ticker=null.\n"
			"5) For comparison, trend, or root-cause questions: prefer higher recall with top_k_chunks>=8 and top_k_tables>=6.\n"
			"6) Set needs_tables=true for quantitative comparisons and metric lookups.\n"
			"7) Set needs_multi_period=true only when question spans multiple periods.\n"
			"8) Keep use_company_filter=true unless the user asks cross-company analysis.\n"
			"Return only the structured output that matches the schema.",
		),
		(
			"human",
			"Analyze and plan for this request.\n\n"
			"[RAW QUESTION]\n{question}\n\n"
			"[DOMAIN DECISION]\n"
			"domain_status={domain_status}\n"
			"domain_reason={domain_reason}\n"
			"domain_confidence={domain_confidence}\n"
			"domain_signals={domain_signals}\n\n"
			"[EXPLICIT CONTEXT]\n"
			"company_ticker={company_ticker}\n"
			"time_filter={time_filter}\n\n"
			"[COMPANY CATALOG]\n{company_catalog}\n",
		),
	]
)

READINESS_PROMPT = ChatPromptTemplate.from_messages(
	[
		(
			"system",
			"You are a query readiness controller for a financial GraphRAG assistant.\n"
			"Decide one answer_mode: answer, clarify, or request_ingestion.\n"
			"Choose clarify when the request is underspecified (missing year, period, scope, or metric disambiguation).\n"
			"Choose request_ingestion only when the request is valid but likely requires data not present in filings in scope (for example live market value/current valuation).\n"
			"Choose answer when request is sufficiently specified for retrieval from filings context.\n"
			"For clarify: provide up to 3 targeted clarification questions.\n"
			"For request_ingestion: provide concrete ingestion recommendations and ask user to re-ask after ingestion.\n"
			"Return only structured output that matches the schema.",
		),
		(
			"human",
			"Assess readiness for this request.\n\n"
			"[QUESTION]\n{question}\n\n"
			"[DOMAIN STATUS]\n{domain_status}\n"
			"[DOMAIN REASON]\n{domain_reason}\n"
			"[COMPANY CONTEXT]\ncompany_ticker={company_ticker}\n"
			"time_filter={time_filter}\n\n"
			"[INTENT]\n{intent_json}\n",
		),
	]
)

TOOL_PLAN_PROMPT = ChatPromptTemplate.from_messages(
	[
		(
			"system",
			"You are a tools retriever planner for a Neo4j financial GraphRAG system.\n"
			"Choose the minimal but sufficient set of retrieval tools for the question.\n"
			"Available tools:\n"
			"1) chunk_search: semantic retrieval over Chunk embeddings for narrative context, management discussion, and nearby evidence.\n"
			"2) table_search: semantic retrieval over Table embeddings for numeric values, period comparisons, and structured disclosures.\n"
			"3) graph_traversal: expand from retrieved chunks or tables through Section, Document, MENTIONS, NEXT_CHUNK, and causal edges for relationship reasoning.\n"
			"Selection rules:\n"
			"- Use chunk_search for fact lookup and narrative explanation.\n"
			"- Use table_search when the question involves metrics, ratios, period comparisons, or tabular facts.\n"
			"- Use graph_traversal when the question asks for root cause, entity relationships, chain-of-effects, or cross-document linking.\n"
			"- Prefer multiple tools when a question mixes metrics and narrative or when intent confidence is moderate or low.\n"
			"- Keep company filtering unless the user explicitly asks for cross-company analysis.\n"
			"Return only structured output that matches the schema."
		),
		(
			"human",
			"Plan tool retrieval for this request.\n\n"
			"[RAW QUESTION]\n{question}\n\n"
			"[INTENT]\n{intent_json}\n\n"
			"[RETRIEVAL PLAN]\n{plan_json}\n\n"
			"[GRAPH MODEL]\n{graph_model}\n",
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
			"3) Cover all key data points found in the evidence and do not truncate or skip metrics.\n"
			"4) Write a multi-sentence answer that explains context, magnitude, and direction of changes.\n"
			"5) Include one bullet per distinct insight or metric; aim for 3-8 non-duplicative bullets.\n"
			"6) If evidence is insufficient or conflicting, state uncertainty and list open questions.\n"
			"7) Confidence should reflect evidence quality and coverage.\n"
			"8) Markdown Styling: Use rich Markdown styling to make the answer and bullets scannable. Highlight the most valuable information using **bold** text, specifically focusing on the exact metrics, dates, entities, and root causes the user asked about.\n"
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
