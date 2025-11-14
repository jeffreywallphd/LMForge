import { Card, CardHeader, CardTitle } from "./ui/card";
import { Brain } from "lucide-react";

interface LLMSectionProps {
  title?: string;
  iconColor?: string;
  backgroundColor?: string;
}

export function LLMSection({ 
  title = "LLM", 
  iconColor = "text-purple-600",
  backgroundColor = "bg-white"
}: LLMSectionProps) {
  return (
    <Card className={`w-full ${backgroundColor}`}>
      <CardHeader className="text-center">
        <Brain className={`w-12 h-12 mx-auto mb-2 ${iconColor}`} />
        <CardTitle>{title}</CardTitle>
      </CardHeader>
    </Card>
  );
}