from __future__ import annotations

import argparse
import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from neo4j import GraphDatabase

from agent.query import build_query_graph
from graph.schema import create_schema
from runtime import configure_runtime


def main() -> None:
    configure_runtime()

    parser = argparse.ArgumentParser(description="Run a RootAlpha query against Neo4j")
    parser.add_argument("question", help="The question to ask")
    parser.add_argument("--ticker", default=None, help="Optional company ticker filter")
    parser.add_argument("--period", default=None, help="Optional period filter")
    parser.add_argument("--debug", action="store_true", help="Print intent and retrieval plan details")
    args = parser.parse_args()

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_FAST_MODEL") or os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        temperature=0.0,
    )
    embedder = GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_EMBEDDING_MODEL") or os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"),
    )

    create_schema(driver)
    graph = build_query_graph(driver=driver, llm=llm, embedder=embedder)

    result = graph.invoke(
        {
            "question": args.question,
            "company_ticker": args.ticker,
            "time_filter": args.period,
            "messages": [],
            "chunk_hits": [],
            "table_hits": [],
            "graph_paths": [],
            "evidence": [],
            "gaps": [],
        },
        config={"configurable": {"thread_id": "rootalpha-query"}},
    )

    final_answer = result.get("final_answer") or {}

    if args.debug:
        intent = result.get("intent") or {}
        plan = result.get("retrieval_plan") or {}
        print("\nINTENT\n------")
        print(f"  intent:              {intent.get('intent')}")
        print(f"  needs_tables:        {intent.get('needs_tables')}")
        print(f"  needs_multi_period:  {intent.get('needs_multi_period')}")
        print(f"  needs_graph_trav:    {intent.get('needs_graph_traversal')}")
        print(f"  needs_company_filt:  {intent.get('needs_company_filter')}")
        print(f"  confidence:          {intent.get('confidence')}")
        print(f"  rationale:           {intent.get('rationale')}")
        print("\nRETRIEVAL PLAN\n--------------")
        print(f"  top_k_chunks:        {plan.get('top_k_chunks')}")
        print(f"  top_k_tables:        {plan.get('top_k_tables')}")
        print(f"  hop_depth:           {plan.get('hop_depth')}")
        print(f"  use_causal_edges:    {plan.get('use_causal_edges')}")
        print(f"  use_prev_next:       {plan.get('use_prev_next_chunks')}")
        print(f"  use_company_filt:    {plan.get('use_company_filter')}")
        req_sec = plan.get('required_section_types') or []
        req_doc = plan.get('required_doc_types') or []
        if req_sec:
            print(f"  section_types:       {req_sec}")
        if req_doc:
            print(f"  doc_types:           {req_doc}")
        gaps = result.get("gaps") or []
        if gaps:
            print(f"\nGAPS\n----")
            for g in gaps:
                print(f"  - {g}")

    print("\nANSWER\n------")
    print(final_answer.get("answer", ""))

    bullets = final_answer.get("bullets") or []
    if bullets:
        print("\nBULLETS\n-------")
        for bullet in bullets:
            print(f"- {bullet}")

    citations = final_answer.get("citations") or []
    if citations:
        print("\nCITATIONS\n----------")
        for citation in citations:
            print(citation)

    driver.close()


if __name__ == "__main__":
    main()