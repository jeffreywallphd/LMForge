import { ArrowRight } from "lucide-react";

interface FlowIndicatorProps {
  label?: string;
  flowText?: string;
  backgroundColor?: string;
  textColor?: string;
  arrowColor?: string;
}

export function FlowIndicator({ 
  label = "Data Flow:",
  flowText = "User → LLM → MCR/API → Custom Tools",
  backgroundColor = "bg-white",
  textColor = "text-gray-600",
  arrowColor = "text-blue-500"
}: FlowIndicatorProps) {
  return (
    <div className="text-center">
      <div className={`inline-flex items-center space-x-2 ${backgroundColor} px-4 py-2 rounded-lg shadow`}>
        <span className={`text-sm ${textColor}`}>{label}</span>
        <ArrowRight className={`w-4 h-4 ${arrowColor}`} />
        <span className="text-sm">{flowText}</span>
      </div>
    </div>
  );
}