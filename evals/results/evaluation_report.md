# 📊 RootAlpha GraphRAG Production Evaluation Report
**Generated on:** June 20, 2026 | **Total Scenarios Evaluated:** 30 queries  

This report presents advanced statistical insight, risk indicators, and performance distribution of our financial GraphRAG agentic pipeline.

## 🏁 1. Executive Metrics Summary
| Metric | Average | Std Dev | Min | Median (p50) | p75 | Max | Health Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Faithfulness** | `0.9327` | `0.1400` | `0.5000` | `1.0000` | `1.0000` | `1.0000` | 🟢 Excellent |
| **Answer Relevancy** | `0.8260` | `0.1582` | `0.0000` | `0.8617` | `0.8729` | `0.8964` | 🟡 Watch |
| **Context Recall** | `1.0000` | `0.0000` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | 🟢 Excellent |
| **Factual Correctness(Mode=F1)** | `0.6047` | `0.3587` | `0.0000` | `0.6700` | `0.9650` | `1.0000` | 🔴 Critical Action |
| **Llm Context Precision With Reference** | `0.6379` | `0.3804` | `0.0000` | `0.6995` | `1.0000` | `1.0000` | 🔴 Critical Action |

## 🚨 2. Risk Signals & Alert Notification Panel
> [!WARNING]
> **Critical vulnerabilities identified within the Generation and Retrieval phases:**
> - 🔥 **Absolute Failure Case (0.0) in Answer Relevancy:** Found queries failing entirely. Examples: '__What was Apple's revenue in Q2 FY2025?...__'
> - 🚨 **Critical Underperformance in Factual Correctness(Mode=F1):** Average score is `0.6047`. Core retrieval or reasoning mechanisms for this category require immediate calibration.
> - ⚠️ **High Variance (Instability) in Factual Correctness(Mode=F1) (σ=0.3587):** Pipeline outputs are inconsistent, returning excellent answers on some runs while completely failing on similar ones. Consider investigating prompt ruggedness.
> - 🔥 **Absolute Failure Case (0.0) in Factual Correctness(Mode=F1):** Found queries failing entirely. Examples: '__What was NVIDIA total revenue in Q2 FY2025 compared to Q1 FY2025 and Q...__' | '__What was NVIDIA's Data Center compute revenue and networking revenue i...__'
> - 🚨 **Critical Underperformance in Llm Context Precision With Reference:** Average score is `0.6379`. Core retrieval or reasoning mechanisms for this category require immediate calibration.
> - ⚠️ **High Variance (Instability) in Llm Context Precision With Reference (σ=0.3804):** Pipeline outputs are inconsistent, returning excellent answers on some runs while completely failing on similar ones. Consider investigating prompt ruggedness.
> - 🔥 **Absolute Failure Case (0.0) in Llm Context Precision With Reference:** Found queries failing entirely. Examples: '__What was NVIDIA Automotive revenue in Q2 FY2025 and what drove the inc...__' | '__What product caused NVIDIA gross margin to decline sequentially in Q2 ...__'

## ⛓️ 3. Root Cause Diagnosis (Weakest Scenarios)
The following specific scenarios recorded the lowest overall combined metric scores:
### 🔍 Case Study: What specific operational challenges impacted NVIDIA's gross margin in Q2 FY2025, and what is the company's projected gross margin range for the full fiscal year?
- **Composite Score:** `0.5658`
  - *Faithfulness:* `1.0000`
  - *Answer Relevancy:* `0.8290`
  - *Context Recall:* `1.0000`
  - *Factual Correctness(Mode=F1):* `0.0000`
  - *Llm Context Precision With Reference:* `0.0000`
- **Generated Answer snippet:** *"NVIDIA's gross margins in Q2 FY2025 experienced a sequential decrease, primarily due to inventory provisions for low-yielding Blackwell material and a higher mix of new products within the Data Center segment. Specifically, GAAP gross margin decreased by 3.3 percentage points to 75.1% in Q2 FY2025 f..."*
- **Expected Ground Truth Reference:** *"Sequentially, gross margins decreased primarily due to inventory provisions for low-yielding Blackwell material and a higher mix of new products within the Data Center segment. Despite this, the company expects gross margins for the full fiscal year to be in the mid-70% range...."*

### 🔍 Case Study: What are the two specific product categories mentioned as achieving significant scale in the Data Center segment, and how much sequential growth did Networking revenue see in Q2 FY2025?
- **Composite Score:** `0.5759`
  - *Faithfulness:* `1.0000`
  - *Answer Relevancy:* `0.8794`
  - *Context Recall:* `1.0000`
  - *Factual Correctness(Mode=F1):* `0.0000`
  - *Llm Context Precision With Reference:* `0.0000`
- **Generated Answer snippet:** *"In the Data Center segment, two product categories that achieved significant scale in Q2 FY2025 were Spectrum-X Ethernet for AI and NVIDIA AI Enterprise software. Networking revenue demonstrated strong sequential growth in Q2 FY2025, increasing by 16% to reach $3,668 million, up from $3,171 million ..."*
- **Expected Ground Truth Reference:** *"Spectrum-X Ethernet for AI and NVIDIA AI Enterprise software are the two new product categories achieving significant scale. Networking revenue, which includes Spectrum-X, saw sequential growth of 16% in Q2 FY2025...."*

### 🔍 Case Study: What were the exact dollar amounts for GAAP and Non-GAAP operating expenses in Q2 FY2025, and what was the largest adjustment made to reconcile the two?
- **Composite Score:** `0.6355`
  - *Faithfulness:* `1.0000`
  - *Answer Relevancy:* `0.8560`
  - *Context Recall:* `1.0000`
  - *Factual Correctness(Mode=F1):* `0.0000`
  - *Llm Context Precision With Reference:* `0.3216`
- **Generated Answer snippet:** *"In Q2 FY2025, NVIDIA reported GAAP operating expenses of $3,932 million and Non-GAAP operating expenses of $2,792 million for the three months ended July 28, 2024. The largest adjustment made to reconcile these two figures was for stock-based compensation expense, which amounted to $(1,114) million...."*
- **Expected Ground Truth Reference:** *"In Q2 FY2025, GAAP operating expenses were $3,932 million, while Non-GAAP operating expenses were $2,792 million. The largest adjustment to reconcile the two was stock-based compensation expense, which amounted to $1,114 million...."*

### 🔍 Case Study: What was NVIDIA's Data Center compute revenue and networking revenue in Q2 FY2025?
- **Composite Score:** `0.6506`
  - *Faithfulness:* `0.8000`
  - *Answer Relevancy:* `0.8622`
  - *Context Recall:* `1.0000`
  - *Factual Correctness(Mode=F1):* `0.0000`
  - *Llm Context Precision With Reference:* `0.5909`
- **Generated Answer snippet:** *"In the second quarter of fiscal year 2025, NVIDIA's Data Center compute revenue reached $22.6 billion, marking a significant increase of 162% from the prior year and 17% sequentially. Networking revenue for the same period was $3.7 billion, which represents a 114% increase year-over-year and a 16% s..."*
- **Expected Ground Truth Reference:** *"Data Center compute revenue was $22.6 billion in Q2 FY2025, up 162% from Q2 FY2024 and up 17% sequentially. Networking revenue was $3.7 billion, up 114% from Q2 FY2024, including InfiniBand and Ethernet for AI (Spectrum-X). Networking revenue grew 16% sequentially with a doubling of Ethernet for AI ..."*

### 🔍 Case Study: What was NVIDIA Gaming revenue in Q2 FY2025 and how did it change year-over-year?
- **Composite Score:** `0.6549`
  - *Faithfulness:* `0.6667`
  - *Answer Relevancy:* `0.8223`
  - *Context Recall:* `1.0000`
  - *Factual Correctness(Mode=F1):* `0.5000`
  - *Llm Context Precision With Reference:* `0.2854`
- **Generated Answer snippet:** *"NVIDIA's Gaming revenue in the second quarter of fiscal year 2025 (Q2 FY2025) was $2,880 million. This represents a significant increase of 16% compared to the same period last year, when Gaming revenue was $2,486 million in Q2 FY2024. The growth was primarily driven by higher sales of GeForce RTX 4..."*
- **Expected Ground Truth Reference:** *"NVIDIA Gaming revenue in Q2 FY2025 was $2.9 billion, up 9% from Q1 FY2025 and up 16% from Q2 FY2024, reflecting higher sales of GeForce RTX 40 Series GPUs and game console SOCs...."*


## 📐 4. Statistical Correlation Matrix
Analyzing relationships between retrieval precision (`context_precision`) and final answer quality (`faithfulness`):
- **Retrieval-to-Faithfulness Correlation (Pearson r):** `0.1654`
  - *Insight:* Low correlation. Retriever noise is high, indicating the generator synthesizes answers despite suboptimal retrieval, or vice-versa.

## 🛠️ 5. Production Mitigation Playbook
Recommended architectural and design adjustments based on results:
1. **To Fix Low Relevancy / Sub-optimal factual correctness:**
   - Enhance prompt templating inside [agent/query/prompts.py](agent/query/prompts.py) to explicitly enforce numeric grounding rules.
   - Implement few-shot dynamic examples inside the system-instruction layers.
2. **To Counter High Std Dev (Variance):**
   - Lock LLM temperature to `0.0` (validated in [agent/query/graph.py](agent/query/graph.py)).
   - Leverage the dynamic caching layer in development/testing to reduce nondeterministic behavior.