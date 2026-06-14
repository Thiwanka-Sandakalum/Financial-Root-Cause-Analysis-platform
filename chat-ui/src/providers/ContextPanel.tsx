import React, { createContext, useContext, useState, ReactNode } from "react";

export type PanelType = "chart" | "source" | null;

interface ContextPanelState {
  isOpen: boolean;
  type: PanelType;
  data: any;
  openPanel: (type: PanelType, data: any) => void;
  closePanel: () => void;
}

const ContextPanelContext = createContext<ContextPanelState | undefined>(undefined);

export function ContextPanelProvider({ children }: { children: ReactNode }) {
  const [isOpen, setIsOpen] = useState(false);
  const [type, setType] = useState<PanelType>(null);
  const [data, setData] = useState<any>(null);

  const openPanel = (newType: PanelType, newData: any) => {
    setType(newType);
    setData(newData);
    setIsOpen(true);
  };

  const closePanel = () => {
    setIsOpen(false);
    // Don't clear data immediately to allow exit animation to run smoothly
    setTimeout(() => {
      setType(null);
      setData(null);
    }, 300);
  };

  return (
    <ContextPanelContext.Provider value={{ isOpen, type, data, openPanel, closePanel }}>
      {children}
    </ContextPanelContext.Provider>
  );
}

export function useContextPanel() {
  const context = useContext(ContextPanelContext);
  if (context === undefined) {
    throw new Error("useContextPanel must be used within a ContextPanelProvider");
  }
  return context;
}
