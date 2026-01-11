import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Code, Calculator, LucideIcon } from "lucide-react";

interface MCRItem {
  icon: LucideIcon;
  label: string;
  iconColor: string;
}

interface MCRSectionProps {
  title?: string;
  items?: MCRItem[];
  badgeText?: string;
  badgeVariant?: "default" | "secondary" | "destructive" | "outline";
  backgroundColor?: string;
}

export function MCRSection({ 
  title = "MCR + API",
  items = [
    { icon: Code, label: "Custom Script", iconColor: "text-green-600" },
    { icon: Calculator, label: "Calculus", iconColor: "text-blue-600" }
  ],
  badgeText = "Existing",
  badgeVariant = "secondary",
  backgroundColor = "bg-white"
}: MCRSectionProps) {
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
        <Badge variant={badgeVariant} className="w-full justify-center">
          {badgeText}
        </Badge>
      </CardContent>
    </Card>
  );
}