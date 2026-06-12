import React from "react";
import { FinalAnswer } from "@/providers/Stream";
import { MarkdownText } from "../markdown-text";

interface StructuredAnswerProps {
  finalAnswer?: FinalAnswer;
}

export function StructuredAnswer({ finalAnswer }: StructuredAnswerProps) {
  if (!finalAnswer || (!finalAnswer.answer && (!finalAnswer.bullets || finalAnswer.bullets.length === 0))) {
    return null;
  }

  return (
    <div className="mt-4 mb-6 flex flex-col gap-4">
      {/* answer is now rendered as the main message content in ai.tsx */}

      {finalAnswer.bullets && finalAnswer.bullets.length > 0 && (
        <div className="p-4 bg-muted/30 border border-border/50 rounded-lg">
          <h4 className="font-semibold text-sm mb-2 text-foreground/90">Key Takeaways</h4>
          <ul className="list-disc pl-5 space-y-1 text-sm text-muted-foreground">
            {finalAnswer.bullets.map((bullet, idx) => (
              <li key={idx}><MarkdownText>{bullet}</MarkdownText></li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
