"""
RootAlpha Evaluation Reporter & Insights Portal
=============================================
Generates rich, publication-grade analytical reports from RAGAS evaluation runs.
Produces high-quality markdown summaries with advanced statistical insights,
risk warnings, performance distributions, and correlation maps.

Usage
-----
uv run python evals/report.py --csv evals/results/eval_results.csv --output evals/results/evaluation_report.md
"""
from __future__ import annotations

import argparse
import os
import pandas as pd
import numpy as np


def generate_report(csv_path: str, output_path: str) -> None:
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    total_queries = len(df)

    # Detect metric columns
    exclude_cols = {"user_input", "response", "reference", "retrieved_contexts", "category"}
    metric_cols = [col for col in df.columns if col not in exclude_cols]

    # Calculate overall stats
    stats = {}
    for col in metric_cols:
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        if not series.empty:
            stats[col] = {
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
                "p25": series.quantile(0.25),
                "p50": series.median(),
                "p75": series.quantile(0.75),
                "series": series
            }

    # Generate Markdown Report
    report = []
    report.append("# 📊 RootAlpha GraphRAG Production Evaluation Report")
    report.append(f"**Generated on:** June 20, 2026 | **Total Scenarios Evaluated:** {total_queries} queries  \n")
    report.append("This report presents advanced statistical insight, risk indicators, and performance distribution of our financial GraphRAG agentic pipeline.")

    # 1. Executive Summary Table
    report.append("\n## 🏁 1. Executive Metrics Summary")
    report.append("| Metric | Average | Std Dev | Min | Median (p50) | p75 | Max | Health Status |")
    report.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    warnings_list = []

    for metric, m_stats in stats.items():
        mean = m_stats["mean"]
        std_val = m_stats["std"]
        min_val = m_stats["min"]
        p50 = m_stats["p50"]
        p75 = m_stats["p75"]
        max_val = m_stats["max"]

        # Health Assessment
        if mean >= 0.85:
            status = "🟢 Excellent"
        elif mean >= 0.70:
            status = "🟡 Watch"
        else:
            status = "🔴 Critical Action"

        display_name = metric.replace("_", " ").title()
        report.append(
            f"| **{display_name}** | `{mean:.4f}` | `{std_val:.4f}` | `{min_val:.4f}` | `{p50:.4f}` | `{p75:.4f}` | `{max_val:.4f}` | {status} |"
        )

        # Trigger Advanced Warnings based on statistics
        if mean < 0.70:
            warnings_list.append(
                f"🚨 **Critical Underperformance in {display_name}:** Average score is `{mean:.4f}`. Core retrieval or reasoning mechanisms for this category require immediate calibration."
            )
        if std_val > 0.25:
            warnings_list.append(
                f"⚠️ **High Variance (Instability) in {display_name} (σ={std_val:.4f}):** Pipeline outputs are inconsistent, returning excellent answers on some runs while completely failing on similar ones. Consider investigating prompt ruggedness."
            )
        if min_val == 0.0:
            worst_cases = df[df[metric] == 0.0]["user_input"].head(2).tolist()
            case_str = " | ".join([f"'__{q[:70]}...__'" for q in worst_cases])
            warnings_list.append(
                f"🔥 **Absolute Failure Case (0.0) in {display_name}:** Found queries failing entirely. Examples: {case_str}"
            )

    # 2. Advanced Risk & Alert Notification Panel
    report.append("\n## 🚨 2. Risk Signals & Alert Notification Panel")
    if warnings_list:
        report.append("> [!WARNING]\n> **Critical vulnerabilities identified within the Generation and Retrieval phases:**")
        for warning in warnings_list:
            report.append(f"> - {warning}")
    else:
        report.append("> [!NOTE]\n> **AI Pipeline Health Check:** All metrics are within stable operating boundaries. No critical drifts or anomalies detected.")

    # 3. Micro-level Critical Query Breakdown (Top 5 Weakest Points)
    report.append("\n## ⛓️ 3. Root Cause Diagnosis (Weakest Scenarios)")
    report.append("The following specific scenarios recorded the lowest overall combined metric scores:")
    
    # Calculate a composite score as simple mean of metric values
    df["composite_score"] = df[metric_cols].mean(axis=1)
    weakest_df = df.sort_values(by="composite_score", ascending=True).head(5)

    for idx, row in weakest_df.iterrows():
        report.append(f"### 🔍 Case Study: {row['user_input']}")
        report.append(f"- **Composite Score:** `{row['composite_score']:.4f}`")
        for m in metric_cols:
            report.append(f"  - *{m.replace('_', ' ').title()}:* `{row[m]:.4f}`")
        report.append(f"- **Generated Answer snippet:** *\"{str(row['response'])[:300]}...\"*")
        if "reference" in row and pd.notna(row["reference"]):
            report.append(f"- **Expected Ground Truth Reference:** *\"{str(row['reference'])[:300]}...\"*")
        report.append("")

    # 4. Statistical Correlations
    report.append("\n## 📐 4. Statistical Correlation Matrix")
    report.append("Analyzing relationships between retrieval precision (`context_precision`) and final answer quality (`faithfulness`):")
    if "llm_context_precision_with_reference" in df.columns and "faithfulness" in df.columns:
        corr = df["llm_context_precision_with_reference"].corr(df["faithfulness"])
        report.append(f"- **Retrieval-to-Faithfulness Correlation (Pearson r):** `{corr:.4f}`")
        if corr > 0.5:
            report.append("  - *Insight:* High positive correlation. Improvements in Neo4j vector retrievers will directly improve final answer faithfulness.")
        elif corr < 0.2:
            report.append("  - *Insight:* Low correlation. Retriever noise is high, indicating the generator synthesizes answers despite suboptimal retrieval, or vice-versa.")
    else:
        report.append("- (Insufficient metric columns to compute pairwise correlations)")

    # 5. Production Mitigations Playbook
    report.append("\n## 🛠️ 5. Production Mitigation Playbook")
    report.append("Recommended architectural and design adjustments based on results:")
    report.append("1. **To Fix Low Relevancy / Sub-optimal factual correctness:**")
    report.append("   - Enhance prompt templating inside [agent/query/prompts.py](agent/query/prompts.py) to explicitly enforce numeric grounding rules.")
    report.append("   - Implement few-shot dynamic examples inside the system-instruction layers.")
    report.append("2. **To Counter High Std Dev (Variance):**")
    report.append("   - Lock LLM temperature to `0.0` (validated in [agent/query/graph.py](agent/query/graph.py)).")
    report.append("   - Leverage the dynamic caching layer in development/testing to reduce nondeterministic behavior.")

    # Write output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"Publication-grade Markdown report successfully exported to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate rich evaluation insights reports.")
    parser.add_argument("--csv", default="evals/results/eval_results.csv", help="Source evaluation CSV")
    parser.add_argument("--output", default="evals/results/evaluation_report.md", help="Destination Markdown path")
    args = parser.parse_args()
    generate_report(args.csv, args.output)
