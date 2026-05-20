from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from neo4j import Driver

from query.agent_query.evidence_utils import (
    dedupe_documents,
    evidence_item,
    serialize_docs,
    serialize_graph_paths,
    state_dict,
    table_has_multi_period_comparison,
)
from query.agent_query.message_utils import messages_from_input_payload, question_from_state
from query.agent_query.prompts import ANSWER_PROMPT, INTENT_PROMPT, PLAN_PROMPT
from query.core.neo4j_query_retriever import (
    expand_graph_context,
    retrieve_chunk_documents,
    retrieve_table_documents,
)
from query.core.query_models import FinalAnswer, QueryIntent, QueryState, RetrievalPlan


def make_nodes(
    driver: Driver,
    llm: ChatGoogleGenerativeAI,
    embedder: GoogleGenerativeAIEmbeddings,
) -> dict[str, callable]:
    intent_chain = INTENT_PROMPT | llm.with_structured_output(QueryIntent)
    plan_chain = PLAN_PROMPT | llm.with_structured_output(RetrievalPlan)
    answer_chain = ANSWER_PROMPT | llm.with_structured_output(FinalAnswer)

    def normalize_input(state: QueryState) -> dict:
        question = question_from_state(state)
        messages = state.get("messages") or messages_from_input_payload(state.get("input"))

        if not messages and question:
            messages = [HumanMessage(content=question)]

        return {
            "question": question,
            "company_ticker": (state.get("company_ticker") or None),
            "time_filter": (state.get("time_filter") or None),
            "messages": messages,
        }

    def classify_intent(state: QueryState) -> dict:
        result = intent_chain.invoke(
            {
                "question": question_from_state(state),
                "company_ticker": state.get("company_ticker"),
                "time_filter": state.get("time_filter"),
            }
        )
        return {"intent": state_dict(result)}

    def build_retrieval_plan(state: QueryState) -> dict:
        result = plan_chain.invoke(
            {
                "question": question_from_state(state),
                "intent_json": state_dict(state.get("intent")),
            }
        )
        return {"retrieval_plan": state_dict(result)}

    def retrieve_chunks(state: QueryState) -> dict:
        question = question_from_state(state)
        if not question:
            raise ValueError("No question text found in the graph input or messages.")
        plan = state.get("retrieval_plan") or {}
        top_k = plan.get("top_k_chunks", 8)
        use_company_filter = plan.get("use_company_filter", True)
        ticker = state.get("company_ticker") if use_company_filter else None
        embedding = embedder.embed_query(question)
        hits = retrieve_chunk_documents(
            driver=driver,
            embedding=embedding,
            top_k=top_k,
            company_ticker=ticker,
            time_filter=state.get("time_filter"),
        )
        return {"chunk_hits": hits}

    def retrieve_tables(state: QueryState) -> dict:
        plan = state.get("retrieval_plan") or {}
        top_k = plan.get("top_k_tables", 6)
        if not top_k:
            return {"table_hits": []}
        question = question_from_state(state)
        if not question:
            raise ValueError("No question text found in the graph input or messages.")
        use_company_filter = plan.get("use_company_filter", True)
        ticker = state.get("company_ticker") if use_company_filter else None
        embedding = embedder.embed_query(question)
        hits = retrieve_table_documents(
            driver=driver,
            embedding=embedding,
            top_k=top_k,
            company_ticker=ticker,
            time_filter=state.get("time_filter"),
        )
        return {"table_hits": hits}

    def traverse_graph(state: QueryState) -> dict:
        chunk_hits = state.get("chunk_hits") or []
        table_hits = state.get("table_hits") or []
        all_docs = dedupe_documents(chunk_hits + table_hits)
        plan = state.get("retrieval_plan") or {}
        paths = expand_graph_context(
            driver=driver,
            source_documents=all_docs,
            hop_depth=plan.get("hop_depth", 2),
            use_causal_edges=plan.get("use_causal_edges", True),
        )
        return {"graph_paths": paths}

    def merge_and_rank_evidence(state: QueryState) -> dict:
        chunk_hits = state.get("chunk_hits") or []
        table_hits = state.get("table_hits") or []
        all_docs = dedupe_documents(chunk_hits + table_hits)
        evidence = [evidence_item(doc) for doc in all_docs]
        return {"evidence": evidence}

    def check_evidence_gaps(state: QueryState) -> dict:
        evidence = state.get("evidence") or []
        plan = state.get("retrieval_plan") or {}
        intent = state.get("intent") or {}
        gaps: list[str] = []

        if not evidence:
            gaps.append("No evidence retrieved for this question.")
            return {"gaps": gaps}

        if intent.get("needs_multi_period") and not table_has_multi_period_comparison(evidence):
            gaps.append("Need evidence from multiple periods for this question.")

        return {"gaps": gaps}

    def synthesize_answer(state: QueryState) -> dict:
        evidence = state.get("evidence") or []
        gaps = state.get("gaps") or []

        if not evidence:
            fallback = FinalAnswer(
                answer="The current evidence is insufficient to give a reliable answer.",
                bullets=[f"Missing: {g}" for g in gaps],
                confidence="low",
            )
            return {
                "final_answer": fallback.model_dump(),
                "messages": [AIMessage(content=fallback.answer)],
            }

        result = answer_chain.invoke(
            {
                "question": question_from_state(state),
                "company_ticker": state.get("company_ticker"),
                "time_filter": state.get("time_filter"),
                "intent_json": state_dict(state.get("intent")),
                "plan_json": state_dict(state.get("retrieval_plan")),
                "evidence_text": serialize_docs(
                    [{"page_content": item.get("text", ""), "metadata": item} for item in evidence]
                ),
                "graph_text": serialize_graph_paths(state.get("graph_paths") or []),
            }
        )
        return {
            "final_answer": state_dict(result),
            "messages": [AIMessage(content=result.answer if hasattr(result, "answer") else "")],
        }

    return {
        "normalize_input": normalize_input,
        "classify_intent": classify_intent,
        "build_retrieval_plan": build_retrieval_plan,
        "retrieve_chunks": retrieve_chunks,
        "retrieve_tables": retrieve_tables,
        "traverse_graph": traverse_graph,
        "merge_and_rank_evidence": merge_and_rank_evidence,
        "check_evidence_gaps": check_evidence_gaps,
        "synthesize_answer": synthesize_answer,
    }
