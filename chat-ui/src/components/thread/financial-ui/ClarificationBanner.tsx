import React from "react";
import { AlertTriangle, HelpCircle } from "lucide-react";
import { ReadinessDecision } from "@/providers/Stream";

interface ClarificationBannerProps {
  readinessDecision?: ReadinessDecision;
  policyFlags?: string[];
  openQuestions?: string[];
}

export function ClarificationBanner({ readinessDecision, policyFlags, openQuestions }: ClarificationBannerProps) {
  const clarificationQuestions = openQuestions?.length ? openQuestions : (readinessDecision?.clarification_questions || []);
  const hasFlags = policyFlags && policyFlags.length > 0;
  const hasQuestions = clarificationQuestions.length > 0;

  if (!hasFlags && !hasQuestions) return null;

  return (
    <div className="flex flex-col gap-3 mt-4 mb-2">
      {hasFlags && (
        <div className="p-3 border rounded-lg flex items-start gap-3 bg-amber-500/10 border-amber-500/30">
          <AlertTriangle className="h-5 w-5 text-amber-500 mt-0.5 shrink-0" />
          <div className="flex flex-col gap-1">
            <span className="font-semibold text-sm text-amber-600 dark:text-amber-500">
              Policy Warning
            </span>
            <div className="text-xs text-muted-foreground">
              <ul className="list-disc pl-4 mt-1 space-y-0.5">
                {policyFlags.map((flag, idx) => <li key={idx}>{flag}</li>)}
              </ul>
            </div>
          </div>
        </div>
      )}

      {hasQuestions && (
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2 text-muted-foreground">
            <HelpCircle className="h-4 w-4 text-blue-500/70" />
            <span className="text-xs font-medium">To provide the best answer, please clarify:</span>
          </div>
          <div className="flex flex-wrap gap-2 mt-1">
            {clarificationQuestions.map((q, idx) => (
              <button 
                key={idx} 
                className="text-xs bg-muted/40 hover:bg-muted/70 text-muted-foreground border border-border/50 px-3 py-1.5 rounded-full transition-colors text-left"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
