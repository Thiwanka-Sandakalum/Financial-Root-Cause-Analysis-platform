"""Query intent classification and entity extraction."""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.config import Settings
from src.llm.gemini import build_fast_model


class DateRange(BaseModel):
    """Temporal range extracted from query."""

    start: Optional[date] = Field(None, description="Start date (YYYY-MM-DD)")
    end: Optional[date] = Field(None, description="End date (YYYY-MM-DD)")


class QueryIntent(BaseModel):
    """Structured representation of user query intent."""

    intent_type: Literal[
        "fact_lookup",
        "metric_comparison",
        "temporal_analysis",
        "relationship_inquiry",
        "root_cause_analysis",
        "other",
    ] = Field(
        description=(
            "Intent category. "
            "fact_lookup=single data point, "
            "metric_comparison=compare values, "
            "temporal_analysis=trend over time, "
            "relationship_inquiry=cause/effect or qualitative context, "
            "root_cause_analysis=explicit causal analysis of drivers, "
            "other=not answerable from financial filings."
        )
    )
    companies: list[str] = Field(
        default_factory=list,
        description=(
            "Ticker symbols or company names explicitly mentioned in the question "
            "(e.g. 'AAPL', 'Microsoft'). Empty list if none mentioned."
        ),
    )
    metrics: list[str] = Field(
        default_factory=list,
        description=(
            "Financial metrics referenced in the question "
            "(e.g. 'revenue', 'EPS', 'gross margin', 'EBITDA'). "
            "Empty list if none mentioned."
        ),
    )
    time_range: Optional[DateRange] = Field(
        None,
        description=(
            "Calendar date range the question covers. "
            "Null if the query has no temporal scope. "
            "Convert fiscal quarter labels to calendar dates."
        ),
    )
    question_text: str = Field(description="Normalised, de-jargoned version of the user question")


def classify_query(query: str, settings: Settings) -> QueryIntent:
    """Classify user query and extract structured intent.
    
    Args:
        query: User's natural language question
        settings: Application settings with model configuration
        
    Returns:
        QueryIntent with extracted entities and intent type
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a financial query classifier for an investor data system. "
                    "Determine whether a question can be answered from investor-relevant sources: "
                    "financial reports, earnings releases, investor presentations, SEC filings, "
                    "financial news coverage, analyst reports, or market performance data — "
                    "and if so, extract structured metadata.\n\n"
                    "INTENT TYPES — choose exactly one:\n"
                    "• fact_lookup         — single data point from investor or financial sources\n"
                    "  e.g. 'What was [Company] revenue in Q3?', "
                    "'What is the gross margin for fiscal 2024?', "
                    "'What was [Company] stock price at year end?'\n"
                    "• metric_comparison   — comparing two or more financial or market values\n"
                    "  e.g. 'How did operating income change from Q2 to Q3?', "
                    "'Compare [Company A] vs [Company B] net income'\n"
                    "• temporal_analysis   — trend, growth, or trajectory over time\n"
                    "  e.g. 'How has EPS grown over the past four quarters?', "
                    "'Show me revenue quarter by quarter', "
                    "'How did [Company] stock perform in [year]?', "
                    "'What were the biggest stock gains in [year]?'\n"
                    "• relationship_inquiry — cause/effect, events, or qualitative context\n"
                    "  e.g. 'What caused the Q4 revenue spike?', "
                    "'What drove [Company] dominance in [year]?', "
                    "'What did management say about demand in the earnings call?', "
                    "'What major product did [Company] announce at [event]?'\n"
                    "• root_cause_analysis — identify primary drivers behind metric moves\n"
                    "  e.g. 'Why did revenue fall in Q3?', "
                    "'What were the root causes of margin compression?', "
                    "'What caused EPS miss in FY2025?'\n"
                    "• other               — NOT answerable from any investor or financial source\n"
                    "  e.g. weather, recipes, general trivia, creative writing, code help, "
                    "greetings, medical advice, sports scores, entertainment\n\n"
                    "RULES:\n"
                    "1. Stock performance, market cap, investor returns, and analyst coverage "
                    "are FINANCIAL topics — classify as temporal_analysis or fact_lookup, "
                    "NOT as 'other'.\n"
                    "2. Company announcements, product launches, and executive actions covered "
                    "in financial news are relationship_inquiry, NOT 'other'.\n"
                    "3. Only classify as 'other' when the question has NO connection to a "
                    "company's financial or investor profile whatsoever.\n"
                    "4. Extract only companies and metrics explicitly mentioned in the question.\n"
                    "5. For time periods, convert fiscal quarter labels to calendar dates."
                ),
            ),
            (
                "human",
                "Classify this query and extract structured intent:\n\n{question}",
            ),
        ]
    )

    model = build_fast_model(settings).with_structured_output(QueryIntent)
    chain = prompt | model

    try:
        result: QueryIntent = chain.invoke({"question": query})
        if result.time_range is not None and not (result.time_range.start or result.time_range.end):
            result.time_range = None

        return result
    except Exception as exc:
        raise RuntimeError(f"Query classification failed: {exc}") from exc
