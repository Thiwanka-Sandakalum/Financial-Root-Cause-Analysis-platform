from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import END, START, StateGraph
from neo4j import Driver, GraphDatabase

from query.agent_query.nodes import make_nodes
from query.core.query_models import QueryState


logger = logging.getLogger(__name__)

def build_query_graph(
	driver: Driver,
	llm: ChatGoogleGenerativeAI,
	embedder: GoogleGenerativeAIEmbeddings,
):
	nodes = make_nodes(driver=driver, llm=llm, embedder=embedder)

	def route_after_analysis(state: QueryState) -> str:
		if state.get("blocked_reason"):
			return "synthesize_answer"
		return "assess_readiness"

	def route_after_readiness(state: QueryState) -> str:
		if state.get("blocked_reason"):
			return "synthesize_answer"
		return "plan_retrieval_tools"

	builder = StateGraph(QueryState)
	builder.add_node("analyze_request", nodes["analyze_request"])
	builder.add_node("assess_readiness", nodes["assess_readiness"])
	builder.add_node("plan_retrieval_tools", nodes["plan_retrieval_tools"])
	builder.add_node("retrieve_with_tools", nodes["retrieve_with_tools"])
	builder.add_node("merge_and_rank_evidence", nodes["merge_and_rank_evidence"])
	builder.add_node("quality_gate", nodes["quality_gate"])
	builder.add_node("synthesize_answer", nodes["synthesize_answer"])

	builder.add_edge(START, "analyze_request")
	builder.add_conditional_edges(
		"analyze_request",
		route_after_analysis,
		{
			"assess_readiness": "assess_readiness",
			"synthesize_answer": "synthesize_answer",
		},
	)
	builder.add_conditional_edges(
		"assess_readiness",
		route_after_readiness,
		{
			"plan_retrieval_tools": "plan_retrieval_tools",
			"synthesize_answer": "synthesize_answer",
		},
	)
	builder.add_edge("plan_retrieval_tools", "retrieve_with_tools")
	builder.add_edge("retrieve_with_tools", "merge_and_rank_evidence")
	builder.add_edge("merge_and_rank_evidence", "quality_gate")
	builder.add_edge("quality_gate", "synthesize_answer")
	builder.add_edge("synthesize_answer", END)

	return builder.compile()

def create_query_graph():
	load_dotenv()  # Load environment variables from .env file
	driver = GraphDatabase.driver(
		os.environ["NEO4J_URI"],
		auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
	)
	llm = ChatGoogleGenerativeAI(
		model=os.getenv("GEMINI_FAST_MODEL"),
		temperature=0.0,
	)
	embedder = GoogleGenerativeAIEmbeddings(
		model=os.getenv("GEMINI_EMBEDDING_MODEL") or "gemini-embedding-001"
	)
	logger.debug("Building query graph with configured Neo4j, LLM, and embedding clients.")
	return build_query_graph(driver=driver, llm=llm, embedder=embedder)
