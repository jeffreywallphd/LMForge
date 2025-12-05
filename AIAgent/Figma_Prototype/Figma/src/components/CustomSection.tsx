import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { FileText, LucideIcon } from "lucide-react";

interface CustomItem {
  icon: LucideIcon;
  label: string;
  iconColor: string;
}

interface CustomSectionProps {
  title?: string;
  items?: CustomItem[];
  backgroundColor?: string;
}

export function CustomSection({ 
  title = "Custom",
  items = [
    { icon: FileText, label: "Python/React", iconColor: "text-yellow-600" }
  ],
  backgroundColor = "bg-white"
}: CustomSectionProps) {
  return (
    <Card className={`w-full ${backgroundColor}`}>
      <CardHeader className="text-center">
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {items.map((item, index) => (
          <div key={index} className="flex items-center space-x-2">
            <item.icon className={`w-4 h-4 ${item.iconColor}`} />
            <span className="text-sm">{item.label}</span>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}