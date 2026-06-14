import "./index.css";
import App from "./App.tsx";
import { createRoot } from "react-dom/client";
import { StreamProvider } from "./providers/Stream.tsx";
import { ThreadProvider } from "./providers/Thread.tsx";
import { Toaster } from "@/components/ui/sonner";
import { NuqsAdapter } from "nuqs/adapters/react-router/v6";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      staleTime: 1000 * 60, // 1 minute
    },
  },
});

createRoot(document.getElementById("root")!).render(
  <QueryClientProvider client={queryClient}>
    <BrowserRouter>
      <NuqsAdapter>
        <ThreadProvider>
          <StreamProvider>
            <App />
          </StreamProvider>
        </ThreadProvider>
        <Toaster />
      </NuqsAdapter>
    </BrowserRouter>
  </QueryClientProvider>,
);
