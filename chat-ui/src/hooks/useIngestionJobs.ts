import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api-client";
import { toast } from "sonner";
import { useState, useEffect } from "react";

export function useIngestionJobs() {
  const queryClient = useQueryClient();
  const [reportedErrors, setReportedErrors] = useState<Set<string>>(new Set());

  // Query to fetch jobs. We poll every 3 seconds if there are active jobs.
  const query = useQuery({
    queryKey: ["jobs"],
    queryFn: () => api.ingestion.listJobs("RUNNING"), // The backend lists RUNNING and QUEUED automatically
    refetchInterval: (query) => {
      // Polling strategy: poll every 3 seconds if there are jobs, otherwise every 15 seconds to catch new ones.
      // Or we can just stop polling and rely on invalidation from upload/start. Let's poll slowly if empty.
      const jobs = query.state.data?.jobs || [];
      return jobs.length > 0 ? 3000 : 10000;
    },
  });

  // Effect to catch FAILED jobs or error messages
  // Since the API listJobs("RUNNING") might not return FAILED jobs, we might also want to fetch FAILED periodically, 
  // or rely on the jobs themselves containing errors if they fail. The job model has an 'errors' array.
  useEffect(() => {
    if (query.data?.jobs) {
      query.data.jobs.forEach(job => {
        if (job.status === "FAILED" || job.errors?.length > 0) {
          job.errors.forEach(err => {
            const errKey = `${job.id}-${err.stage}`;
            if (!reportedErrors.has(errKey)) {
              toast.error(`Job Failed (${job.stage}): ${err.error_message}`);
              setReportedErrors(prev => new Set(prev).add(errKey));
            }
          });
        }
      });
    }
  }, [query.data, reportedErrors]);

  // Mutation to start ingestion
  const startIngestionMutation = useMutation({
    mutationFn: (documentId: string) => api.ingestion.start([documentId], true),
    onSuccess: () => {
      toast.success("Ingestion queued!");
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      queryClient.invalidateQueries({ queryKey: ["documents"] });
    },
    onError: () => {
      toast.error("Failed to start ingestion");
    }
  });

  return {
    query,
    startIngestionMutation
  };
}
