import React, { useState } from "react";
import { ChevronDown, ChevronUp, FileText, Maximize2 } from "lucide-react";
import { ChunkHit } from "@/providers/Stream";
import { useContextPanel } from "@/providers/ContextPanel";

interface SourceReferencesProps {
  chunkHits?: ChunkHit[];
  citations?: any[];
}

export function SourceReferences({ chunkHits, citations }: SourceReferencesProps) {
  const [expanded, setExpanded] = useState(false);
  const { openPanel } = useContextPanel();

  // Use citations if available, otherwise fallback to chunkHits
  const hasCitations = citations && citations.length > 0;
  const hasChunkHits = chunkHits && chunkHits.length > 0;
  
  if (!hasCitations && !hasChunkHits) return null;

  const count = hasCitations ? citations.length : chunkHits!.length;

  return (
    <div className="mt-4 mb-2 border border-border/60 rounded-lg overflow-hidden bg-card/50">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-3 py-2 bg-muted/30 hover:bg-muted/50 transition-colors text-xs font-medium text-muted-foreground"
      >
        <div className="flex items-center gap-2">
          <FileText className="h-3.5 w-3.5" />
          <span>{count} {count === 1 ? 'Source' : 'Sources'} Referenced</span>
        </div>
        {expanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
      </button>

      {expanded && (
        <div className="p-3 border-t border-border/50 flex flex-col gap-3 max-h-64 overflow-y-auto">
          {hasCitations
            ? citations.map((citation, idx) => {
                // Handle raw string citations just in case
                if (typeof citation === "string") {
                  return (
                    <div key={idx} className="text-sm border-b border-border/30 last:border-0 pb-2 last:pb-0">
                      {citation}
                    </div>
                  );
                }
                
                // Handle object citations
                return (
                  <div key={idx} className="flex flex-col gap-1 text-sm border-b border-border/30 last:border-0 pb-2 last:pb-0 relative group pr-6">
                    <button 
                      onClick={() => openPanel('source', citation)}
                      className="absolute top-0 right-0 p-1 rounded-md bg-muted/50 hover:bg-muted text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity hover:text-foreground"
                      title="Read Document"
                    >
                      <Maximize2 className="w-3.5 h-3.5" />
                    </button>
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-semibold text-foreground/80 text-xs">
                        {citation.title || "Document"}
                      </span>
                      {citation.source_type && (
                        <span className="text-muted-foreground text-[10px] bg-muted/60 border border-border/50 px-1.5 py-0.5 rounded-sm uppercase">
                          {citation.source_type}
                        </span>
                      )}
                      {citation.page !== undefined && (
                        <span className="text-muted-foreground text-[10px] bg-muted/60 border border-border/50 px-1.5 py-0.5 rounded-sm">
                          Page {citation.page}
                        </span>
                      )}
                      {citation.score !== undefined && (
                        <span className="text-muted-foreground text-[10px] opacity-70">
                          (Score: {Number(citation.score).toFixed(2)})
                        </span>
                      )}
                    </div>
                    {citation.source_id && (
                      <div className="text-muted-foreground text-[10px] break-all opacity-80 mt-0.5 font-mono">
                        {citation.source_id}
                      </div>
                    )}
                  </div>
                );
              })
            : chunkHits!.map((hit, idx) => (
                // Existing chunk_hits rendering
                <div key={idx} className="flex flex-col gap-1 text-sm border-b border-border/30 last:border-0 pb-2 last:pb-0 relative group pr-6">
                  <button 
                    onClick={() => openPanel('source', hit)}
                    className="absolute top-0 right-0 p-1 rounded-md bg-muted/50 hover:bg-muted text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity hover:text-foreground"
                    title="Read Document"
                  >
                    <Maximize2 className="w-3.5 h-3.5" />
                  </button>
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-semibold text-foreground/80 text-xs">
                      {hit.metadata?.company || "Source Document"}
                    </span>
                    {hit.metadata?.doc_type && (
                      <span className="text-muted-foreground text-[10px] bg-muted/60 border border-border/50 px-1.5 py-0.5 rounded-sm">
                        {hit.metadata.doc_type}
                      </span>
                    )}
                    {hit.metadata?.period && (
                      <span className="text-muted-foreground text-xs">
                        {hit.metadata.period}
                      </span>
                    )}
                  </div>
                  {hit.metadata?.section_title && (
                    <div className="text-muted-foreground text-[11px] line-clamp-1 opacity-80">
                      Section: {hit.metadata.section_title}
                    </div>
                  )}
                </div>
              ))}
        </div>
      )}
    </div>
  );
}
