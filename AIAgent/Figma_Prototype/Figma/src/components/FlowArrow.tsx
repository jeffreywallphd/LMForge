import { ArrowRight } from "lucide-react";

interface FlowArrowProps {
  color?: string;
  size?: string;
  direction?: "right" | "down" | "left" | "up";
}

export function FlowArrow({ 
  color = "text-gray-400",
  size = "w-6 h-6",
  direction = "right"
}: FlowArrowProps) {
  const getRotation = () => {
    switch (direction) {
      case "down": return "rotate-90 lg:rotate-0";
      case "left": return "rotate-180";
      case "up": return "-rotate-90";
      default: return "";
    }
  };

  return (
    <ArrowRight className={`${size} ${color} ${getRotation()}`} />
  );
}