import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Wrench, Brain, Star, Search, Save, LucideIcon } from "lucide-react";

interface ToolButton {
  icon: LucideIcon;
  label: string;
  variant?: "default" | "destructive" | "outline" | "secondary" | "ghost" | "link";
}

interface ToolBadge {
  label: string;
  variant?: "default" | "secondary" | "destructive" | "outline";
}

interface ToolColumn {
  title: string;
  buttons?: ToolButton[];
  badges?: ToolBadge[];
}

interface ToolsSectionProps {
  title?: string;
  titleIcon?: LucideIcon;
  columns?: ToolColumn[];
  backgroundColor?: string;
}

const defaultColumns: ToolColumn[] = [
  {
    title: "Model Selection",
    buttons: [
      { icon: Brain, label: "Select a Model", variant: "outline" }
    ],
    badges: [
      { label: "Tools Frontend", variant: "secondary" },
      { label: "Local Dev", variant: "secondary" }
    ]
  },
  {
    title: "Tool Management", 
    buttons: [
      { icon: Wrench, label: "List of Tools", variant: "outline" }
    ],
    badges: [
      { label: "Configs", variant: "secondary" }
    ]
  },
  {
    title: "Actions",
    buttons: [
      { icon: Star, label: "Pop Star", variant: "outline" },
      { icon: Search, label: "Lookup", variant: "outline" },
      { icon: Save, label: "Save", variant: "outline" }
    ]
  }
];

export function ToolsSection({ 
  title = "Tools & Actions",
  titleIcon = Wrench,
  columns = defaultColumns,
  backgroundColor = "bg-white"
}: ToolsSectionProps) {
  const TitleIcon = titleIcon;
  
  return (
    <Card className={backgroundColor}>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <TitleIcon className="w-5 h-5" />
          <span>{title}</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {columns.map((column, columnIndex) => (
            <div key={columnIndex} className="space-y-4">
              <h3 className="border-b pb-2">{column.title}</h3>
              {column.buttons && column.buttons.map((button, buttonIndex) => (
                <Button 
                  key={buttonIndex}
                  variant={button.variant || "outline"} 
                  className="w-full justify-start"
                >
                  <button.icon className="w-4 h-4 mr-2" />
                  {button.label}
                </Button>
              ))}
              {column.badges && (
                <div className="space-y-2">
                  {column.badges.map((badge, badgeIndex) => (
                    <Badge 
                      key={badgeIndex}
                      variant={badge.variant || "secondary"}
                    >
                      {badge.label}
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}