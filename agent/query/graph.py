from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import END, START, StateGraph
from neo4j import Driver, GraphDatabase

from agent.query.nodes import make_nodes
from retrieval.query_models import QueryState


def build_query_graph(
    driver: Driver,
    llm: ChatGoogleGenerativeAI,
    embedder: GoogleGenerativeAIEmbeddings,
):
    nodes = make_nodes(driver=driver, llm=llm, embedder=embedder)

    builder = StateGraph(QueryState)
    builder.add_node("normalize_input", nodes["normalize_input"])
    builder.add_node("classify_intent", nodes["classify_intent"])
    builder.add_node("build_retrieval_plan", nodes["build_retrieval_plan"])
    builder.add_node("retrieve_chunks", nodes["retrieve_chunks"])
    builder.add_node("retrieve_tables", nodes["retrieve_tables"])
    builder.add_node("traverse_graph", nodes["traverse_graph"])
    builder.add_node("merge_and_rank_evidence", nodes["merge_and_rank_evidence"])
    builder.add_node("check_evidence_gaps", nodes["check_evidence_gaps"])
    builder.add_node("synthesize_answer", nodes["synthesize_answer"])

    builder.add_edge(START, "normalize_input")
    builder.add_edge("normalize_input", "classify_intent")
    builder.add_edge("classify_intent", "build_retrieval_plan")
    builder.add_edge("build_retrieval_plan", "retrieve_chunks")
    builder.add_conditional_edges(
        "retrieve_chunks",
        lambda state: (
            "retrieve_tables"
            if (state.get("retrieval_plan") or {}).get("top_k_tables", 0)
            else "traverse_graph"
        ),
        {"retrieve_tables": "retrieve_tables", "traverse_graph": "traverse_graph"},
    )
    builder.add_edge("retrieve_tables", "traverse_graph")
    builder.add_edge("traverse_graph", "merge_and_rank_evidence")
    builder.add_edge("merge_and_rank_evidence", "check_evidence_gaps")
    builder.add_edge("check_evidence_gaps", "synthesize_answer")
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
    print("Building query graph...", driver, llm, embedder)
    return build_query_graph(driver=driver, llm=llm, embedder=embedder)
