import { v4 as uuidv4 } from "uuid";
import { ReactNode, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { useStreamContext } from "@/providers/Stream";
import { useState, FormEvent } from "react";
import { Button } from "../ui/button";
import { Checkpoint, Message } from "@langchain/langgraph-sdk";
import { AssistantMessage, AssistantMessageLoading } from "./messages/ai";
import { AgentInlineLoader } from "./agent-stepper";
import { HumanMessage } from "./messages/human";
import {
  DO_NOT_RENDER_ID_PREFIX,
  ensureToolCallsHaveResponses,
} from "@/lib/ensure-tool-responses";
import { TooltipIconButton } from "./tooltip-icon-button";
import {
  ArrowDown,
  LoaderCircle,
  PanelRightOpen,
  PanelRightClose,
  SquarePen,
  LineChart,
  Wrench,
  FileSearch,
  BarChart3,
  Calculator,
  ReceiptText,
} from "lucide-react";
import { useQueryState, parseAsBoolean } from "nuqs";
import { StickToBottom, useStickToBottomContext } from "use-stick-to-bottom";
import ThreadHistory from "./history";
import { toast } from "sonner";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { Label } from "../ui/label";
import { Switch } from "../ui/switch";
import { GitHubSVG } from "../icons/github";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../ui/tooltip";

function StickyToBottomContent(props: {
  content: ReactNode;
  footer?: ReactNode;
  className?: string;
  contentClassName?: string;
}) {
  const context = useStickToBottomContext();
  return (
    <div
      ref={context.scrollRef}
      style={{ width: "100%", height: "100%" }}
      className={props.className}
    >
      <div ref={context.contentRef} className={props.contentClassName}>
        {props.content}
      </div>

      {props.footer}
    </div>
  );
}

function ScrollToBottom(props: { className?: string }) {
  const { isAtBottom, scrollToBottom } = useStickToBottomContext();

  if (isAtBottom) return null;
  return (
    <Button
      variant="outline"
      className={props.className}
      onClick={() => scrollToBottom()}
    >
      <ArrowDown className="w-4 h-4" />
      <span>Scroll to bottom</span>
    </Button>
  );
}



const TOOL_PROMPTS: Record<string, string> = {
  document_search: "SYSTEM DIRECTIVE: You are strictly required to invoke the Document Search retrieval tool to gather qualitative evidence from corporate filings, annual reports, or official press releases to accurately answer the user's query. Do not rely solely on your internal knowledge base.",
  financial_metrics: "SYSTEM DIRECTIVE: You are strictly required to invoke the Financial Metrics tool to extract precise quantitative data (e.g., revenue, EBITDA, margins, CAPEX) from the structured financial database. Your answer must be strictly backed by this retrieved quantitative data.",
  valuation: "SYSTEM DIRECTIVE: You are strictly required to invoke the Valuation Analysis tool. You must compute or retrieve valuation multiples, DCF components, or comparative company analysis data before attempting to answer the user's query.",
  earnings: "SYSTEM DIRECTIVE: You are strictly required to invoke the Earnings Transcripts search tool to identify management commentary, Q&A insights, and forward-looking guidance from recent earnings calls before answering the user's query.",
};

type ToolId = keyof typeof TOOL_PROMPTS | null;

function FinancialToolsMenu({
  selectedTool,
  onSelectTool,
}: {
  selectedTool: ToolId;
  onSelectTool: (id: ToolId) => void;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSelect = (id: ToolId) => {
    onSelectTool(id);
    setIsOpen(false);
  };

  const getTriggerIcon = () => {
    switch (selectedTool) {
      case "document_search":
        return <FileSearch className="w-5 h-5 text-blue-500" />;
      case "financial_metrics":
        return <BarChart3 className="w-5 h-5 text-green-500" />;
      case "valuation":
        return <Calculator className="w-5 h-5 text-purple-500" />;
      case "earnings":
        return <ReceiptText className="w-5 h-5 text-orange-500" />;
      default:
        return <Wrench className="w-5 h-5" />;
    }
  };

  return (
    <div className="relative" ref={menuRef}>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className={cn(
          "text-gray-500 hover:text-gray-700 rounded-full transition-colors relative",
          selectedTool && "bg-gray-100 dark:bg-gray-800 ring-2 ring-gray-200 dark:ring-gray-700"
        )}
        onClick={() => setIsOpen(!isOpen)}
        title={selectedTool ? "Tool Selected" : "Select Tool"}
      >
        {getTriggerIcon()}
      </Button>

      {isOpen && (
        <div className="absolute bottom-full left-0 mb-2 w-56 rounded-xl bg-white shadow-lg ring-1 ring-black ring-opacity-5 z-50 overflow-hidden dark:bg-gray-800 dark:ring-gray-700">
          <div className="px-4 py-3 border-b border-gray-100 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50 flex justify-between items-center">
            <p className="text-sm font-medium text-gray-900 dark:text-gray-200">Financial Tools</p>
            {selectedTool && (
              <button 
                onClick={() => handleSelect(null)}
                className="text-xs text-red-500 hover:text-red-700 underline"
              >
                Clear
              </button>
            )}
          </div>
          <div className="py-1">
            <div 
              onClick={() => handleSelect("document_search")}
              className={cn("flex items-center gap-3 px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer transition-colors", selectedTool === "document_search" && "bg-blue-50 dark:bg-blue-900/20")}
            >
              <FileSearch className="w-4 h-4 text-blue-500" />
              Document Search
            </div>
            <div 
              onClick={() => handleSelect("financial_metrics")}
              className={cn("flex items-center gap-3 px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer transition-colors", selectedTool === "financial_metrics" && "bg-green-50 dark:bg-green-900/20")}
            >
              <BarChart3 className="w-4 h-4 text-green-500" />
              Financial Metrics
            </div>
            <div 
              onClick={() => handleSelect("valuation")}
              className={cn("flex items-center gap-3 px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer transition-colors", selectedTool === "valuation" && "bg-purple-50 dark:bg-purple-900/20")}
            >
              <Calculator className="w-4 h-4 text-purple-500" />
              Valuation Analysis
            </div>
            <div 
              onClick={() => handleSelect("earnings")}
              className={cn("flex items-center gap-3 px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer transition-colors", selectedTool === "earnings" && "bg-orange-50 dark:bg-orange-900/20")}
            >
              <ReceiptText className="w-4 h-4 text-orange-500" />
              Earnings Transcripts
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export function Thread() {
  const [selectedTool, setSelectedTool] = useState<ToolId>(null);
  const [threadId, setThreadId] = useQueryState("threadId");
  const [chatHistoryOpen, setChatHistoryOpen] = useQueryState(
    "chatHistoryOpen",
    parseAsBoolean.withDefault(false),
  );
  const [hideToolCalls, setHideToolCalls] = useQueryState(
    "hideToolCalls",
    parseAsBoolean.withDefault(false),
  );
  const [input, setInput] = useState("");
  const [firstTokenReceived, setFirstTokenReceived] = useState(false);
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");

  const stream = useStreamContext();
  const messages = stream.messages;
  const isLoading = stream.isLoading;

  const lastError = useRef<string | undefined>(undefined);

  useEffect(() => {
    if (!stream.error) {
      lastError.current = undefined;
      return;
    }
    try {
      const message = (stream.error as any).message;
      if (!message || lastError.current === message) {
        // Message has already been logged. do not modify ref, return early.
        return;
      }

      // Message is defined, and it has not been logged yet. Save it, and send the error
      lastError.current = message;
      toast.error("An error occurred. Please try again.", {
        description: (
          <p>
            <strong>Error:</strong> <code>{message}</code>
          </p>
        ),
        richColors: true,
        closeButton: true,
      });
    } catch {
      // no-op
    }
  }, [stream.error]);

  // TODO: this should be part of the useStream hook
  const prevMessageLength = useRef(0);
  useEffect(() => {
    if (
      messages.length !== prevMessageLength.current &&
      messages?.length &&
      messages[messages.length - 1].type === "ai"
    ) {
      setFirstTokenReceived(true);
    }

    prevMessageLength.current = messages.length;
  }, [messages]);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    setFirstTokenReceived(false);

    let finalInput = input;
    if (selectedTool && TOOL_PROMPTS[selectedTool]) {
      finalInput = `${input}\n\n[HIDDEN_TOOL_PROMPT_START]\n${TOOL_PROMPTS[selectedTool]}\n[HIDDEN_TOOL_PROMPT_END]`;
    }

    const newHumanMessage: Message = {
      id: uuidv4(),
      type: "human",
      content: finalInput,
    };

    const toolMessages = ensureToolCallsHaveResponses(stream.messages);
    stream.submit(
      { messages: [...toolMessages, newHumanMessage] },
      {
        streamMode: ["values"],
        optimisticValues: (prev) => ({
          ...prev,
          messages: [
            ...(prev.messages ?? []),
            ...toolMessages,
            newHumanMessage,
          ],
        }),
      },
    );

    setInput("");
  };

  const handleRegenerate = (
    parentCheckpoint: Checkpoint | null | undefined,
  ) => {
    // Do this so the loading state is correct
    prevMessageLength.current = prevMessageLength.current - 1;
    setFirstTokenReceived(false);
    stream.submit(undefined, {
      checkpoint: parentCheckpoint,
      streamMode: ["values"],
    });
  };

  const chatStarted = !!threadId || !!messages.length;
  const hasNoAIOrToolMessages = !messages.find(
    (m) => m.type === "ai" || m.type === "tool",
  );

  return (
    <div className="flex w-full h-screen overflow-hidden">
      <div className="relative lg:flex hidden">
        <motion.div
          className="absolute h-full border-r bg-white overflow-hidden z-20"
          style={{ width: 300 }}
          animate={
            isLargeScreen
              ? { x: chatHistoryOpen ? 0 : -300 }
              : { x: chatHistoryOpen ? 0 : -300 }
          }
          initial={{ x: -300 }}
          transition={
            isLargeScreen
              ? { type: "spring", stiffness: 300, damping: 30 }
              : { duration: 0 }
          }
        >
          <div className="relative h-full" style={{ width: 300 }}>
            <ThreadHistory />
          </div>
        </motion.div>
      </div>
      <motion.div
        className={cn(
          "flex-1 flex flex-col min-w-0 overflow-hidden relative",
          !chatStarted && "grid-rows-[1fr]",
        )}
        layout={isLargeScreen}
        animate={{
          marginLeft: chatHistoryOpen ? (isLargeScreen ? 300 : 0) : 0,
          width: chatHistoryOpen
            ? isLargeScreen
              ? "calc(100% - 300px)"
              : "100%"
            : "100%",
        }}
        transition={
          isLargeScreen
            ? { type: "spring", stiffness: 300, damping: 30 }
            : { duration: 0 }
        }
      >
        {!chatStarted && (
          <div className="absolute top-0 left-0 w-full flex items-center justify-between gap-3 p-2 pl-4 z-10">
            <div>
              {(!chatHistoryOpen || !isLargeScreen) && (
                <Button
                  className="hover:bg-gray-100"
                  variant="ghost"
                  onClick={() => setChatHistoryOpen((p) => !p)}
                >
                  {chatHistoryOpen ? (
                    <PanelRightOpen className="size-5" />
                  ) : (
                    <PanelRightClose className="size-5" />
                  )}
                </Button>
              )}
            </div>
            <div className="absolute top-2 right-4 flex items-center">
            </div>
          </div>
        )}
        {chatStarted && (
          <div className="flex items-center justify-between gap-3 p-2 z-10 relative">
            <div className="flex items-center justify-start gap-2 relative">
              <div className="absolute left-0 z-10">
                {(!chatHistoryOpen || !isLargeScreen) && (
                  <Button
                    className="hover:bg-gray-100"
                    variant="ghost"
                    onClick={() => setChatHistoryOpen((p) => !p)}
                  >
                    {chatHistoryOpen ? (
                      <PanelRightOpen className="size-5" />
                    ) : (
                      <PanelRightClose className="size-5" />
                    )}
                  </Button>
                )}
              </div>
              <motion.button
                className="flex gap-2 items-center cursor-pointer"
                onClick={() => setThreadId(null)}
                animate={{
                  marginLeft: !chatHistoryOpen ? 48 : 0,
                }}
                transition={{
                  type: "spring",
                  stiffness: 300,
                  damping: 30,
                }}
              >
                <LineChart className="w-8 h-8 text-blue-600 dark:text-blue-500" />
                <span className="text-xl font-semibold tracking-tight">
                  RootAlpha Analyst
                </span>
              </motion.button>
            </div>

            <div className="flex items-center gap-4">
              <TooltipIconButton
                size="lg"
                className="p-4"
                tooltip="New thread"
                variant="ghost"
                onClick={() => setThreadId(null)}
              >
                <SquarePen className="size-5" />
              </TooltipIconButton>
            </div>

            <div className="absolute inset-x-0 top-full h-5 bg-gradient-to-b from-background to-background/0" />
          </div>
        )}

        <StickToBottom className="relative flex-1 overflow-hidden">
          <StickyToBottomContent
            className={cn(
              "absolute px-4 inset-0 overflow-y-scroll [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent",
              !chatStarted && "flex flex-col items-stretch mt-[25vh]",
              chatStarted && "grid grid-rows-[1fr_auto]",
            )}
            contentClassName="pt-8 pb-16  max-w-3xl mx-auto flex flex-col gap-4 w-full"
            content={
              <>
                {(() => {
                  const visibleMessages = messages.filter(
                    (m) => !m.id?.startsWith(DO_NOT_RENDER_ID_PREFIX)
                  );
                  const lastHumanMessageIndex = visibleMessages
                    .map((m) => m.type)
                    .lastIndexOf("human");

                  return visibleMessages.map((message, index) => {
                    const isCurrentRun = index > lastHumanMessageIndex;
                    if (isLoading && isCurrentRun) {
                      return null;
                    }

                    return message.type === "human" ? (
                      <HumanMessage
                        key={message.id || `${message.type}-${index}`}
                        message={message}
                        isLoading={isLoading}
                      />
                    ) : (
                      <AssistantMessage
                        key={message.id || `${message.type}-${index}`}
                        message={message}
                        isLoading={isLoading}
                        handleRegenerate={handleRegenerate}
                      />
                    );
                  });
                })()}
                {/* Special rendering case where there are no AI/tool messages, but there is an interrupt.
                    We need to render it outside of the messages list, since there are no messages to render */}
                {hasNoAIOrToolMessages && !!stream.interrupt && (
                  <AssistantMessage
                    key="interrupt-msg"
                    message={undefined}
                    isLoading={isLoading}
                    handleRegenerate={handleRegenerate}
                  />
                )}
                {isLoading && (
                  <AgentInlineLoader />
                )}
              </>
            }
            footer={
              <div className="sticky flex flex-col items-center gap-8 bottom-0 bg-white">
                {!chatStarted && (
                  <div className="flex flex-col items-center justify-center gap-4 mb-8 mt-4">
                    <div className="flex gap-3 items-center">
                      <LineChart className="flex-shrink-0 h-8 w-8 text-blue-600 dark:text-blue-500" />
                      <h1 className="text-2xl font-semibold tracking-tight">
                        RootAlpha Analyst
                      </h1>
                    </div>
                    <p className="text-sm text-muted-foreground max-w-sm text-center">
                      Try asking: "What was the primary driver for the revenue decline in Q3?"
                    </p>
                  </div>
                )}

                <ScrollToBottom className="absolute bottom-full left-1/2 -translate-x-1/2 mb-4 animate-in fade-in-0 zoom-in-95" />

                <div className="bg-muted rounded-2xl border shadow-xs mx-auto mb-8 w-full max-w-3xl relative z-10">
                  <form
                    onSubmit={handleSubmit}
                    className="grid grid-rows-[1fr_auto] gap-2 max-w-3xl mx-auto"
                  >
                    <textarea
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (
                          e.key === "Enter" &&
                          !e.shiftKey &&
                          !e.metaKey &&
                          !e.nativeEvent.isComposing
                        ) {
                          e.preventDefault();
                          const el = e.target as HTMLElement | undefined;
                          const form = el?.closest("form");
                          form?.requestSubmit();
                        }
                      }}
                      placeholder="Ask about financial performance, metrics, or root causes..."
                      className="p-3.5 pb-0 border-none bg-transparent field-sizing-content shadow-none ring-0 outline-none focus:outline-none focus:ring-0 resize-none"
                    />

                    <div className="flex items-center justify-between p-2 pt-4">
                      <div>
                        <div className="flex items-center space-x-2">
                          <FinancialToolsMenu selectedTool={selectedTool} onSelectTool={setSelectedTool} />
                        </div>
                      </div>
                      {stream.isLoading ? (
                        <Button key="stop" onClick={() => stream.stop()}>
                          <LoaderCircle className="w-4 h-4 animate-spin" />
                          Cancel
                        </Button>
                      ) : (
                        <Button
                          type="submit"
                          className="transition-all shadow-md"
                          disabled={isLoading || !input.trim()}
                        >
                          Send
                        </Button>
                      )}
                    </div>
                  </form>
                </div>
              </div>
            }
          />
        </StickToBottom>
      </motion.div>
    </div>
  );
}
