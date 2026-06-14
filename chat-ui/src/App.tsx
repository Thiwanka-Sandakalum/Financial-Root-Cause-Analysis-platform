import { Routes, Route } from "react-router-dom";
import "./App.css";
import { Thread } from "@/components/thread";
import { WorkspaceLayout } from "@/components/layout/WorkspaceLayout";
import { KnowledgeBaseView } from "@/components/knowledge-base/KnowledgeBaseView";

function App() {
  return (
    <Routes>
      <Route element={<WorkspaceLayout />}>
        <Route path="/" element={<Thread />} />
        <Route path="/knowledge-base" element={<KnowledgeBaseView />} />
      </Route>
    </Routes>
  );
}

export default App;
