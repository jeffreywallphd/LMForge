import { Card, CardHeader, CardTitle } from "./ui/card";
import { User } from "lucide-react";

interface UserSectionProps {
  title?: string;
  iconColor?: string;
  backgroundColor?: string;
}

export function UserSection({ 
  title = "User", 
  iconColor = "text-blue-600",
  backgroundColor = "bg-white"
}: UserSectionProps) {
  return (
    <Card className={`w-full ${backgroundColor}`}>
      <CardHeader className="text-center">
        <User className={`w-12 h-12 mx-auto mb-2 ${iconColor}`} />
        <CardTitle>{title}</CardTitle>
      </CardHeader>
    </Card>
  );
}