import React from "react";
import { X, BarChart2, FileText } from "lucide-react";
import { useContextPanel } from "@/providers/ContextPanel";
import { DynamicChart } from "@/components/thread/financial-ui/DynamicChart";
import { MarkdownText } from "@/components/thread/markdown-text";
import { Button } from "@/components/ui/button";

export function ContextPanel() {
  const { isOpen, type, data, closePanel } = useContextPanel();

  return (
    <div 
      className={`h-full bg-card border-l border-border/50 shadow-2xl transition-all duration-300 ease-in-out flex flex-col shrink-0 ${isOpen ? "w-[400px] xl:w-[500px] opacity-100" : "w-0 opacity-0 overflow-hidden border-none"}`}
    >
      <div className="flex items-center justify-between p-4 border-b border-border/50 shrink-0">
        <div className="flex items-center gap-2">
          {type === 'chart' ? <BarChart2 className="w-5 h-5 text-blue-600" /> : <FileText className="w-5 h-5 text-blue-600" />}
          <h2 className="font-semibold text-foreground">
            {type === 'chart' ? 'Data Visualization' : 'Source Document'}
          </h2>
        </div>
        <Button variant="ghost" size="icon" onClick={closePanel} className="h-8 w-8 text-muted-foreground hover:bg-muted shrink-0">
          <X className="w-4 h-4" />
        </Button>
      </div>
      
      <div className="flex-1 overflow-y-auto p-6 min-w-[400px] xl:min-w-[500px]">
        {type === 'chart' && data && (
          <div className="w-full flex flex-col gap-4">
             {data.title && <h3 className="text-xl font-bold">{data.title}</h3>}
             <DynamicChart visualization={data} className="h-[400px] border-none shadow-none bg-transparent p-0 mt-0" hideExpand={true} />
             <div className="text-sm text-muted-foreground mt-4">
               <p><strong>X-Axis:</strong> {data.x_field}</p>
               <p><strong>Y-Axis:</strong> {data.y_field}</p>
               <p className="mt-2 text-xs opacity-70">This chart is dynamically generated from verified knowledge base metrics.</p>
             </div>
          </div>
        )}
        
        {type === 'source' && data && (
          <div className="space-y-4 h-full flex flex-col pb-4">
            <div className="space-y-2 shrink-0">
              <h3 className="text-xl font-bold">{data.title || data.metadata?.company || "Source Document"}</h3>
              <div className="flex flex-wrap gap-2">
                {data.source_type && <span className="text-xs bg-muted/60 border border-border/50 px-2 py-1 rounded uppercase font-medium">{data.source_type}</span>}
                {data.metadata?.doc_type && <span className="text-xs bg-muted/60 border border-border/50 px-2 py-1 rounded uppercase font-medium">{data.metadata.doc_type}</span>}
                {data.page !== undefined && <span className="text-xs bg-muted/60 border border-border/50 px-2 py-1 rounded">Page {data.page}</span>}
                {data.metadata?.period && <span className="text-xs bg-muted/60 border border-border/50 px-2 py-1 rounded">{data.metadata.period}</span>}
              </div>
            </div>
            
            {data.source_id ? (
              <div className="flex-1 w-full mt-4 min-h-[500px] border border-border/50 rounded-lg overflow-hidden bg-background">
                <iframe 
                  src={`http://localhost:8000/api/v1/documents/${data.source_id.split('_')[0]}/content${data.page ? `#page=${data.page}` : ''}`}
                  className="w-full h-full"
                  title="Document Viewer"
                />
              </div>
            ) : (
              <div className="mt-6 text-foreground bg-muted/10 p-5 rounded-lg border border-border/50 leading-relaxed font-serif text-[15px]">
                 <MarkdownText>
                   {data.content || data.page_content || "Full document text would be rendered here in a production environment. Currently showing citation metadata."}
                 </MarkdownText>
              </div>
            )}
            
            {data.source_id && (
              <div className="text-xs text-muted-foreground opacity-60 font-mono mt-2 shrink-0 flex justify-between items-center">
                <span>ID: {data.source_id}</span>
                <a href={`http://localhost:8000/api/v1/documents/${data.source_id.split('_')[0]}/content${data.page ? `#page=${data.page}` : ''}`} target="_blank" rel="noopener noreferrer" className="hover:underline text-blue-500">
                  Open in New Tab
                </a>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
