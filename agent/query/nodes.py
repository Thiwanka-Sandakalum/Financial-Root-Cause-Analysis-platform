from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from neo4j import Driver

from agent.query.evidence_utils import (
    dedupe_documents,
    evidence_item,
    serialize_docs,
    serialize_graph_paths,
    state_dict,
    table_has_multi_period_comparison,
)
from agent.query.message_utils import messages_from_input_payload, question_from_state
from agent.query.prompts import ANSWER_PROMPT, INTENT_PROMPT, PLAN_PROMPT
from retrieval.neo4j_query_retriever import (
    expand_graph_context,
    retrieve_chunk_documents,
    retrieve_table_documents,
)
from retrieval.query_models import FinalAnswer, QueryIntent, QueryState, RetrievalPlan


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
        messages = state.get("messages") or messages_from_input_payload(
            state.get("input")
        )

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
        intent = state.get("intent") or {}
        result = plan_chain.invoke(
            {
                "question": question_from_state(state),
                "intent_json": intent,
            }
        )
        return {"retrieval_plan": state_dict(result)}

    def retrieve_chunks(state: QueryState) -> dict:
        question = question_from_state(state)
        if not question:
            raise ValueError("No question text found in the graph input or messages.")

        query_embedding = embedder.embed_query(question)
        plan = state.get("retrieval_plan") or {}
        docs = retrieve_chunk_documents(
            driver=driver,
            embedding=query_embedding,
            top_k=plan.get("top_k_chunks", 6),
            company_ticker=state.get("company_ticker")
            if plan.get("use_company_filter", True)
            else None,
            time_filter=state.get("time_filter"),
        )
        return {"chunk_hits": docs}

    def retrieve_tables(state: QueryState) -> dict:
        plan = state.get("retrieval_plan") or {}
        if not plan.get("top_k_tables", 0):
            return {"table_hits": []}

        question = question_from_state(state)
        if not question:
            raise ValueError("No question text found in the graph input or messages.")

        query_embedding = embedder.embed_query(question)
        docs = retrieve_table_documents(
            driver=driver,
            embedding=query_embedding,
            top_k=plan.get("top_k_tables", 4),
            company_ticker=state.get("company_ticker")
            if plan.get("use_company_filter", True)
            else None,
            time_filter=state.get("time_filter"),
        )
        return {"table_hits": docs}

    def traverse_graph(state: QueryState) -> dict:
        chunk_docs = state.get("chunk_hits", [])
        table_docs = state.get("table_hits", [])
        source_docs = dedupe_documents(chunk_docs + table_docs)
        plan = state.get("retrieval_plan") or {}
        paths = expand_graph_context(
            driver=driver,
            source_documents=source_docs,
            hop_depth=plan.get("hop_depth", 2),
            use_causal_edges=plan.get("use_causal_edges", True),
        )
        return {"graph_paths": paths}

    def merge_and_rank_evidence(state: QueryState) -> dict:
        chunk_docs = state.get("chunk_hits", [])
        table_docs = state.get("table_hits", [])
        evidence_docs = dedupe_documents(chunk_docs + table_docs)

        evidence = [evidence_item(doc) for doc in evidence_docs]
        return {"evidence": evidence}

    def check_evidence_gaps(state: QueryState) -> dict:
        gaps: list[str] = []
        evidence = state.get("evidence", [])
        plan = state.get("retrieval_plan") or {}
        intent = state.get("intent") or {}

        if not evidence:
            gaps.append("No evidence retrieved for this question.")

        periods = {item.get("period") for item in evidence if item.get("period")}
        doc_types = {item.get("doc_type") for item in evidence if item.get("doc_type")}
        has_table_multi_period = table_has_multi_period_comparison(evidence)

        if (
            intent.get("needs_multi_period")
            and len(periods) < 2
            and not has_table_multi_period
        ):
            gaps.append("Need evidence from multiple periods for this question.")

        if intent.get("needs_tables") and not any(
            item.get("source_type") == "table" for item in evidence
        ):
            gaps.append("Need table evidence for this question.")

        if plan.get("use_causal_edges") and not state.get("graph_paths"):
            gaps.append("Need graph traversal evidence for causal explanation.")

        if state.get("company_ticker") and state.get("company_ticker") not in {
            item.get("ticker") for item in evidence if item.get("ticker")
        }:
            gaps.append("No company-filtered evidence matched the requested ticker.")

        if intent.get("intent") in {"trend_analysis", "root_cause"} and not doc_types:
            gaps.append("Need document-type coverage for trend or causal analysis.")

        return {"gaps": gaps}

    def synthesize_answer(state: QueryState) -> dict:
        if state.get("gaps"):
            final_answer = FinalAnswer(
                answer="The current evidence is insufficient to give a reliable answer.",
                bullets=[f"Missing: {gap}" for gap in (state.get("gaps") or [])],
                citations=[],
                confidence="low",
                open_questions=state.get("gaps") or [],
            )
            return {
                "final_answer": final_answer.model_dump(),
                "messages": [AIMessage(content=final_answer.answer)],
            }

        result = answer_chain.invoke(
            {
                "question": question_from_state(state),
                "company_ticker": state.get("company_ticker"),
                "time_filter": state.get("time_filter"),
                "intent_json": state.get("intent", {}),
                "plan_json": state.get("retrieval_plan", {}),
                "evidence_text": serialize_docs(
                    [
                        {
                            "page_content": item["text"],
                            "type": "Document",
                            "metadata": item,
                        }
                        for item in state.get("evidence", [])
                    ]
                ),
                "graph_text": serialize_graph_paths(state.get("graph_paths", [])),
            }
        )
        final_answer = state_dict(result)
        return {
            "final_answer": final_answer,
            "messages": [AIMessage(content=final_answer.get("answer", ""))],
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
