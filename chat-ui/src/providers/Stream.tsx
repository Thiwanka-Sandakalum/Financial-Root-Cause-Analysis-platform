import React, {
  createContext,
  useContext,
  ReactNode,
  useState,
  useEffect,
} from "react";
import { useStream } from "@langchain/langgraph-sdk/react";
import { type Message } from "@langchain/langgraph-sdk";
import {
  uiMessageReducer,
  type UIMessage,
  type RemoveUIMessage,
} from "@langchain/langgraph-sdk/react-ui";
import { useQueryState } from "nuqs";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { LangGraphLogoSVG } from "@/components/icons/langgraph";
import { Label } from "@/components/ui/label";
import { ArrowRight } from "lucide-react";
import { PasswordInput } from "@/components/ui/password-input";
import { getApiKey } from "@/lib/api-key";
import { useThreads } from "./Thread";
import { toast } from "sonner";

export interface ReadinessDecision {
  answer_mode?: string;
  reason?: string;
  missing_slots?: string[];
  clarification_questions?: string[];
  ingest_recommendations?: string[];
}

export interface IntentData {
  primary_intent?: string;
  confidence_score?: number;
}

export interface ToolPlan {
  selected_tools?: string[];
}

export interface ChunkHit {
  page_content?: string;
  metadata?: {
    company?: string;
    doc_type?: string;
    period?: string;
    section_title?: string;
    [key: string]: any;
  };
}

export interface VisualizationData {
  enabled: boolean;
  chart_type?: "BarChart" | "LineChart" | "PieChart" | string;
  data?: any[];
  x_field?: string;
  y_field?: string;
}

export interface FinalAnswer {
  answer?: string;
  bullets?: string[];
  citations?: any[];
  open_questions?: string[];
}

export type StateType = { 
  messages: Message[]; 
  ui?: UIMessage[];
  intent?: IntentData;
  tool_plan?: ToolPlan;
  chunk_hits?: ChunkHit[];
  visualization?: VisualizationData;
  readiness_decision?: ReadinessDecision;
  policy_flags?: string[];
  final_answer?: FinalAnswer;
};

const useTypedStream = useStream<
  StateType,
  {
    UpdateType: {
      messages?: Message[] | Message | string;
      ui?: (UIMessage | RemoveUIMessage)[] | UIMessage | RemoveUIMessage;
    };
    CustomEventType: UIMessage | RemoveUIMessage;
  }
>;

type StreamContextType = ReturnType<typeof useTypedStream> & {
  activeStep: string;
  completedSteps: string[];
};
const StreamContext = createContext<StreamContextType | undefined>(undefined);

async function sleep(ms = 4000) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function checkGraphStatus(
  apiUrl: string,
  apiKey: string | null,
): Promise<boolean> {
  try {
    const res = await fetch(`${apiUrl}/info`, {
      ...(apiKey && {
        headers: {
          "X-Api-Key": apiKey,
        },
      }),
    });

    return res.ok;
  } catch (e) {
    console.error(e);
    return false;
  }
}

const StreamSession = ({
  children,
  apiKey,
  apiUrl,
  assistantId,
}: {
  children: ReactNode;
  apiKey: string | null;
  apiUrl: string;
  assistantId: string;
}) => {
  const [threadId, setThreadId] = useQueryState("threadId");
  const { getThreads, setThreads } = useThreads();
  
  const [activeStep, setActiveStep] = useState<string>("");
  const [completedSteps, setCompletedSteps] = useState<string[]>([]);

  const streamValue = useTypedStream({
    apiUrl,
    apiKey: apiKey ?? undefined,
    assistantId,
    threadId: threadId ?? null,
    callerOptions: {
      fetch: async (input, init) => {
        const response = await fetch(input, init);
        if (response.body && input.toString().includes("/stream")) {
          const [stream1, stream2] = response.body.tee();
          
          (async () => {
            const reader = stream1.getReader();
            const decoder = new TextDecoder();
            let buffer = "";
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              buffer += decoder.decode(value, { stream: true });
              
              const parts = buffer.split(/\r?\n\r?\n/);
              buffer = parts.pop() || "";
              
              for (const part of parts) {
                const lines = part.split(/\r?\n/);
                let eventType = "";
                let dataContent = "";
                
                for (const line of lines) {
                  if (line.startsWith("event:")) {
                    eventType = line.substring(6).trim();
                  } else if (line.startsWith("data:")) {
                    dataContent += line.substring(5).trim();
                  }
                }
                
                if (eventType === "messages" && dataContent) {
                  try {
                    const parsed = JSON.parse(dataContent);
                    if (Array.isArray(parsed) && parsed.length > 1) {
                      const nodeName = parsed[1]?.langgraph_node;
                      if (nodeName) {
                        setActiveStep((current) => {
                          if (current && current !== nodeName) {
                            setCompletedSteps((prev) => 
                              prev.includes(current) ? prev : [...prev, current]
                            );
                          }
                          return nodeName;
                        });
                      }
                    }
                  } catch (e) {}
                }
              }
            }
          })().catch(console.error);

          return new Response(stream2, {
            headers: response.headers,
            status: response.status,
            statusText: response.statusText,
          });
        }
        return response;
      }
    },
    onCustomEvent: (event, options) => {
      options.mutate((prev) => {
        const ui = uiMessageReducer(prev.ui ?? [], event);
        return { ...prev, ui };
      });
    },
    onThreadId: (id) => {
      setThreadId(id);
      // Refetch threads list when thread ID changes.
      // Wait for some seconds before fetching so we're able to get the new thread that was created.
      sleep().then(() => getThreads().then(setThreads).catch(console.error));
    },
  });

  useEffect(() => {
    if (!streamValue.isLoading) {
      // Stream finished, mark active step as completed
      setActiveStep((current) => {
        if (current) {
          setCompletedSteps((prev) => 
            prev.includes(current) ? prev : [...prev, current]
          );
        }
        return "";
      });
    } else {
      // New stream started
      setCompletedSteps([]);
      setActiveStep("");
    }
  }, [streamValue.isLoading]);

  useEffect(() => {
    checkGraphStatus(apiUrl, apiKey).then((ok) => {
      if (!ok) {
        toast.error("Failed to connect to LangGraph server", {
          description: () => (
            <p>
              Please ensure your graph is running at <code>{apiUrl}</code> and
              your API key is correctly set (if connecting to a deployed graph).
            </p>
          ),
          duration: 10000,
          richColors: true,
          closeButton: true,
        });
      }
    });
  }, [apiKey, apiUrl]);

  return (
    <StreamContext.Provider value={{ ...streamValue, activeStep, completedSteps }}>
      {children}
    </StreamContext.Provider>
  );
};

// Default values for the local platform
const DEFAULT_API_URL = "http://localhost:2024";
const DEFAULT_ASSISTANT_ID = "rootalpha";

export const StreamProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  // Hardcode the configuration values for the bespoke financial platform
  // This prevents the routing state from losing connection details.
  const apiUrl = DEFAULT_API_URL;
  const assistantId = DEFAULT_ASSISTANT_ID;
  const apiKey = ""; // Bypassed for local development

  return (
    <StreamSession apiKey={apiKey} apiUrl={apiUrl} assistantId={assistantId}>
      {children}
    </StreamSession>
  );
};

// Create a custom hook to use the context
export const useStreamContext = (): StreamContextType => {
  const context = useContext(StreamContext);
  if (context === undefined) {
    throw new Error("useStreamContext must be used within a StreamProvider");
  }
  return context;
};

export default StreamContext;
