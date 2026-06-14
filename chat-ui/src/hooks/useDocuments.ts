import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api-client";
import { toast } from "sonner";

export function useDocuments(filters?: { ticker?: string; doc_type?: string; period?: string; limit?: number; offset?: number }) {
  const queryClient = useQueryClient();

  // Query to fetch documents
  const query = useQuery({
    queryKey: ["documents", filters],
    queryFn: () => api.documents.list(filters),
  });

  // Mutation to upload a document
  const uploadMutation = useMutation({
    mutationFn: ({ file, meta }: { file: File; meta: { ticker: string; docType: string; period: string; autoIngest?: boolean } }) => 
      api.documents.upload(file, meta),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      // We also invalidate jobs in case autoIngest triggered a new job
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      toast.success("Document uploaded successfully");
    },
    onError: (error: any) => {
      toast.error(`Upload failed: ${error.message}`);
    }
  });

  // Mutation to delete a document
  const deleteMutation = useMutation({
    mutationFn: (id: string) => api.documents.delete(id, true),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      toast.success("Document deleted");
    },
    onError: () => {
      toast.error("Failed to delete document");
    }
  });

  return {
    query,
    uploadMutation,
    deleteMutation
  };
}
