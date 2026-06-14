import React, { useState, useRef } from "react";
import { Database, UploadCloud, FileText, Loader2, PlayCircle, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useDocuments } from "../../hooks/useDocuments";
import { useIngestionJobs } from "../../hooks/useIngestionJobs";

export function KnowledgeBaseView() {
  // Use React Query hooks
  const { query: docsQuery, uploadMutation, deleteMutation } = useDocuments();
  const { query: jobsQuery, startIngestionMutation } = useIngestionJobs();
  
  const documents = docsQuery.data?.documents || [];
  const jobs = jobsQuery.data?.jobs || [];
  const loadingDocs = docsQuery.isLoading;
  
  // Upload State
  const [file, setFile] = useState<File | null>(null);
  const [ticker, setTicker] = useState("DIAL");
  const [docType, setDocType] = useState("Annual Report");
  const [period, setPeriod] = useState("2023");
  const [autoIngest, setAutoIngest] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUpload = async () => {
    if (!file) return;
    
    uploadMutation.mutate({
      file,
      meta: { ticker, docType, period, autoIngest }
    }, {
      onSuccess: () => {
        setFile(null);
      }
    });
  };

  const startIngestion = (docId: string) => {
    startIngestionMutation.mutate(docId);
  };

  const deleteDoc = (docId: string) => {
    deleteMutation.mutate(docId);
  };

  return (
    <div className="w-full h-full p-8 flex flex-col bg-background/50 overflow-y-auto">
      <div className="flex items-center gap-3 mb-8">
        <Database className="w-8 h-8 text-blue-600 dark:text-blue-500" />
        <h1 className="text-3xl font-bold tracking-tight">Knowledge Base</h1>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 flex-1">
        {/* Document List Panel */}
        <div className="xl:col-span-2 border rounded-xl bg-card shadow-sm p-6 flex flex-col">
          <h2 className="text-lg font-semibold mb-4">Ingested Documents</h2>
          <div className="flex-1 overflow-auto min-h-[400px]">
            {loadingDocs ? (
              <div className="flex h-full items-center justify-center">
                <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
              </div>
            ) : documents.length === 0 ? (
              <div className="flex h-full items-center justify-center border border-dashed rounded-lg bg-muted/10">
                <p className="text-muted-foreground">No documents found. Upload one to get started.</p>
              </div>
            ) : (
              <div className="space-y-3 pr-2">
                {documents.map((doc) => (
                  <div key={doc.id} className="flex items-center justify-between p-4 border rounded-lg hover:bg-muted/30 transition-colors bg-background">
                    <div className="flex items-center gap-4">
                      <div className="w-10 h-10 rounded bg-blue-100 dark:bg-blue-900/40 flex items-center justify-center text-blue-600">
                        <FileText className="w-5 h-5" />
                      </div>
                      <div>
                        <p className="font-medium text-sm truncate max-w-[200px]">{doc.filename}</p>
                        <p className="text-xs text-muted-foreground flex items-center gap-2 mt-1">
                          <span className="font-semibold">{doc.company_ticker}</span>
                          <span>•</span>
                          <span>{doc.doc_type}</span>
                          <span>•</span>
                          <span>{doc.fiscal_period}</span>
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${doc.status === 'COMPLETED' ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'}`}>
                        {doc.status}
                      </span>
                      {doc.status !== 'COMPLETED' && (
                        <Button 
                          variant="outline" 
                          size="sm" 
                          onClick={() => startIngestion(doc.id)} 
                          disabled={startIngestionMutation.isPending}
                          className="h-7 text-xs"
                        >
                          {startIngestionMutation.isPending ? (
                            <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                          ) : (
                            <PlayCircle className="w-3 h-3 mr-1" />
                          )}
                          Ingest
                        </Button>
                      )}
                      <Button 
                        variant="ghost" 
                        size="icon" 
                        onClick={() => deleteDoc(doc.id)} 
                        disabled={deleteMutation.isPending}
                        className="h-7 w-7 text-muted-foreground hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-950"
                      >
                        <XCircle className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Upload and Jobs Panel */}
        <div className="border rounded-xl bg-card shadow-sm p-6 flex flex-col gap-6 overflow-y-auto">
          <div>
            <h2 className="text-lg font-semibold mb-4">Upload New Data</h2>
            <div 
              className={`border-2 border-dashed rounded-xl p-6 flex flex-col items-center justify-center gap-2 transition-colors cursor-pointer text-center ${file ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/10' : 'border-border hover:bg-muted/30'}`}
              onClick={() => fileInputRef.current?.click()}
            >
              <input type="file" ref={fileInputRef} className="hidden" accept=".pdf,.txt" onChange={(e) => setFile(e.target.files?.[0] || null)} />
              {file ? (
                <>
                  <FileText className="w-8 h-8 text-blue-500 mb-2" />
                  <p className="font-medium text-sm truncate max-w-[200px]">{file.name}</p>
                  <p className="text-xs text-blue-600 hover:underline" onClick={(e) => { e.stopPropagation(); setFile(null); }}>Remove file</p>
                </>
              ) : (
                <>
                  <UploadCloud className="w-8 h-8 text-muted-foreground mb-2" />
                  <p className="font-medium text-sm">Click to select file</p>
                  <p className="text-xs text-muted-foreground">PDF or TXT up to 50MB</p>
                </>
              )}
            </div>

            {file && (
              <div className="mt-4 space-y-3 animate-in fade-in slide-in-from-top-2">
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <label className="text-xs font-medium text-muted-foreground">Ticker</label>
                    <input className="w-full text-sm p-2 bg-background border rounded-md" value={ticker} onChange={(e) => setTicker(e.target.value)} />
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs font-medium text-muted-foreground">Period</label>
                    <input className="w-full text-sm p-2 bg-background border rounded-md" value={period} onChange={(e) => setPeriod(e.target.value)} />
                  </div>
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-medium text-muted-foreground">Document Type</label>
                  <input className="w-full text-sm p-2 bg-background border rounded-md" value={docType} onChange={(e) => setDocType(e.target.value)} />
                </div>
                <div className="flex items-center space-x-2 pt-2 pb-1">
                  <input 
                    type="checkbox" 
                    id="auto-ingest" 
                    className="rounded border-gray-300"
                    checked={autoIngest}
                    onChange={(e) => setAutoIngest(e.target.checked)} 
                  />
                  <label htmlFor="auto-ingest" className="text-xs font-medium text-muted-foreground">
                    Auto-start ingestion after upload
                  </label>
                </div>
                <Button className="w-full mt-2" onClick={handleUpload} disabled={uploadMutation.isPending}>
                  {uploadMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : <UploadCloud className="w-4 h-4 mr-2" />}
                  {uploadMutation.isPending ? "Uploading..." : "Upload Document"}
                </Button>
              </div>
            )}
          </div>
          
          <div className="flex-1 mt-4">
            <h2 className="text-lg font-semibold mb-4">Active Jobs</h2>
            {jobs.length === 0 ? (
              <div className="border border-dashed rounded-lg p-6 bg-muted/10 text-sm text-muted-foreground text-center flex flex-col items-center justify-center">
                <p>No active processing jobs</p>
              </div>
            ) : (
              <div className="space-y-3">
                {jobs.map((job: any) => (
                  <div key={job.id} className="border rounded-lg p-3 bg-blue-50/50 dark:bg-blue-900/10 shadow-sm animate-in fade-in">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-xs font-semibold text-blue-700 dark:text-blue-400 tracking-wider uppercase">{job.stage}</span>
                      <span className="text-xs font-medium text-muted-foreground">{job.progress}%</span>
                    </div>
                    <div className="w-full bg-blue-200 dark:bg-blue-950 rounded-full h-1.5 mb-2 overflow-hidden">
                      <div className="bg-blue-600 h-1.5 rounded-full transition-all duration-500 ease-out" style={{ width: `${job.progress}%` }}></div>
                    </div>
                    <p className="text-[11px] text-muted-foreground truncate">{job.message}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
