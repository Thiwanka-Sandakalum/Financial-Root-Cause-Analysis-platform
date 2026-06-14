import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell
} from "recharts";
import { VisualizationData } from "@/providers/Stream";
import { Maximize2 } from "lucide-react";
import { useContextPanel } from "@/providers/ContextPanel";

interface DynamicChartProps {
  visualization?: VisualizationData;
  className?: string;
  hideExpand?: boolean;
}

const COLORS = ['#2563eb', '#16a34a', '#dc2626', '#ca8a04', '#9333ea', '#0891b2'];

export function DynamicChart({ visualization, className, hideExpand }: DynamicChartProps) {
  const { openPanel } = useContextPanel();
  if (!visualization?.enabled || !visualization.data || visualization.data.length === 0) {
    return null;
  }

  const { chart_type, data, x_field, y_field } = visualization;

  const renderChart = () => {
    switch (chart_type) {
      case "LineChart":
        return (
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
            <XAxis dataKey={x_field} fontSize={12} tickMargin={8} />
            <YAxis fontSize={12} tickMargin={8} />
            <RechartsTooltip 
              contentStyle={{ borderRadius: '8px', border: '1px solid hsl(var(--border))', backgroundColor: 'hsl(var(--background))' }} 
            />
            <Line type="monotone" dataKey={y_field!} stroke="#2563eb" strokeWidth={2} activeDot={{ r: 8 }} />
          </LineChart>
        );
      case "PieChart":
        return (
          <PieChart>
            <RechartsTooltip 
              contentStyle={{ borderRadius: '8px', border: '1px solid hsl(var(--border))', backgroundColor: 'hsl(var(--background))' }} 
            />
            <Pie
              data={data}
              dataKey={y_field!}
              nameKey={x_field}
              cx="50%"
              cy="50%"
              outerRadius={80}
              fill="#8884d8"
              label
            >
              {data.map((_, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
          </PieChart>
        );
      case "BarChart":
      default:
        return (
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
            <XAxis dataKey={x_field} fontSize={12} tickMargin={8} />
            <YAxis fontSize={12} tickMargin={8} />
            <RechartsTooltip 
              contentStyle={{ borderRadius: '8px', border: '1px solid hsl(var(--border))', backgroundColor: 'hsl(var(--background))' }}
              cursor={{ fill: 'hsl(var(--muted))', opacity: 0.4 }}
            />
            <Bar dataKey={y_field!} fill="#2563eb" radius={[4, 4, 0, 0]} />
          </BarChart>
        );
    }
  };

  return (
    <div className={`relative w-full h-64 mt-4 p-4 border border-border/60 rounded-lg bg-card/50 group ${className || ""}`}>
      {!hideExpand && (
        <button 
          onClick={() => openPanel('chart', visualization)}
          className="absolute top-2 right-2 p-1.5 rounded-md bg-background/80 border border-border/50 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity hover:text-foreground z-10 shadow-sm"
          title="Expand Chart"
        >
          <Maximize2 className="w-4 h-4" />
        </button>
      )}
      <ResponsiveContainer width="100%" height="100%">
        {renderChart()}
      </ResponsiveContainer>
    </div>
  );
}
