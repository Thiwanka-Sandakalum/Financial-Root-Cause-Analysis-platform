import { useStreamContext } from "@/providers/Stream";
import { useEffect, useState } from "react";

export const STEP_LABELS: Record<string, string> = {
  analyze_request: "Analyzing Request",
  assess_readiness: "Assessing Readiness",
  plan_retrieval_tools: "Planning Document Search",
  retrieve_with_tools: "Retrieving Data",
  merge_and_rank_evidence: "Merging & Ranking Evidence",
  quality_gate: "Quality Assessment",
  synthesize_answer: "Synthesizing Final Answer",
};

export function AgentInlineLoader() {
  const { activeStep } = useStreamContext();
  const [dots, setDots] = useState("");

  useEffect(() => {
    const interval = setInterval(() => {
      setDots((prev) => (prev.length >= 4 ? "" : prev + "."));
    }, 400);

    return () => clearInterval(interval);
  }, []);

  // If there's an active step mapped, show it. Otherwise fallback to "Thinking".
  const stepText = activeStep ? STEP_LABELS[activeStep] || "Processing" : "Thinking";

  return (
    <div className="flex items-start mr-auto gap-2 group">
      <div className="flex items-center rounded-2xl bg-muted px-4 py-2 w-max min-w-[160px]">
        <span className="text-sm font-medium text-slate-600 dark:text-slate-300">
          {stepText}{dots}
        </span>
      </div>
    </div>
  );
}
