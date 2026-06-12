import React from "react";
import { Loader2 } from "lucide-react";
import { IntentData, ToolPlan } from "@/providers/Stream";

interface AgentProgressProps {
  intent?: IntentData;
  toolPlan?: ToolPlan;
}

export function AgentProgress({ intent, toolPlan }: AgentProgressProps) {
  if (!intent && !toolPlan?.selected_tools?.length) return null;

  const getStatusMessage = () => {
    const tools = toolPlan?.selected_tools || [];
    if (tools.includes("Table Search")) return "Analyzing financial tables...";
    if (tools.includes("Graph Traversal")) return "Tracing entity relationships...";
    if (tools.includes("Chunk Search")) return "Scanning annual reports...";
    
    if (intent?.primary_intent) {
      // Capitalize each word and replace underscores
      const intentStr = intent.primary_intent.replace(/_/g, " ");
      return `Processing: ${intentStr.charAt(0).toUpperCase() + intentStr.slice(1)}...`;
    }

    return "Analyzing request...";
  };

  return (
    <div className="flex items-center gap-3 px-4 py-3 my-2 text-sm text-muted-foreground bg-muted/40 rounded-lg border border-border/50 animate-in fade-in-0 slide-in-from-bottom-2">
      <Loader2 className="h-4 w-4 animate-spin text-primary" />
      <span className="font-medium text-foreground/80">{getStatusMessage()}</span>
    </div>
  );
}
