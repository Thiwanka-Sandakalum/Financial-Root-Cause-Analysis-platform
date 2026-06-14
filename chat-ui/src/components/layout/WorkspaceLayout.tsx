import { NavLink, Outlet } from "react-router-dom";
import { LineChart, Database, Settings } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { ContextPanelProvider } from "@/providers/ContextPanel";
import { ContextPanel } from "@/components/layout/ContextPanel";

export function WorkspaceLayout() {
  return (
    <ContextPanelProvider>
      <div className="flex w-full h-screen overflow-hidden bg-background">
        {/* Global Sidebar */}
        <nav className="w-16 flex flex-col items-center py-4 border-r bg-muted/20 gap-8 z-50 shadow-sm shrink-0">
          <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center shadow-sm">
            <span className="text-white font-bold text-xl">RA</span>
          </div>
          
          <div className="flex flex-col gap-4 flex-1">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <NavLink 
                    to="/" 
                    className={({ isActive }) => cn(
                      "p-3 rounded-xl transition-colors", 
                      isActive ? "bg-blue-100 text-blue-600 dark:bg-blue-900/40 dark:text-blue-400" : "text-muted-foreground hover:bg-muted"
                    )}
                  >
                    <LineChart className="w-6 h-6" />
                  </NavLink>
                </TooltipTrigger>
                <TooltipContent side="right">Analysis</TooltipContent>
              </Tooltip>
            </TooltipProvider>

            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <NavLink 
                    to="/knowledge-base" 
                    className={({ isActive }) => cn(
                      "p-3 rounded-xl transition-colors", 
                      isActive ? "bg-blue-100 text-blue-600 dark:bg-blue-900/40 dark:text-blue-400" : "text-muted-foreground hover:bg-muted"
                    )}
                  >
                    <Database className="w-6 h-6" />
                  </NavLink>
                </TooltipTrigger>
                <TooltipContent side="right">Knowledge Base</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>

          <div className="mt-auto">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button className="p-3 rounded-xl text-muted-foreground hover:bg-muted transition-colors">
                    <Settings className="w-6 h-6" />
                  </button>
                </TooltipTrigger>
                <TooltipContent side="right">Settings</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        </nav>

        {/* Main Content Area */}
        <main className="flex-1 overflow-hidden relative flex">
          <div className="flex-1 overflow-hidden relative">
            <Outlet />
          </div>
          <ContextPanel />
        </main>
      </div>
    </ContextPanelProvider>
  );
}
