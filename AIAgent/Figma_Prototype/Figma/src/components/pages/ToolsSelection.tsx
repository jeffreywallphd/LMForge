import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { Checkbox } from "../ui/checkbox";
import { Input } from "../ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";
import { Page, AgentConfig } from "../../App";
import { 
  ArrowLeft, 
  ArrowRight, 
  Search, 
  Database, 
  Globe, 
  Calculator, 
  Code, 
  FileText, 
  Image, 
  Mail, 
  Calendar, 
  MessageSquare,
  BarChart,
  Camera,
  Music,
  Video,
  Map,
  ShoppingCart,
  CreditCard,
  Wrench,
  CheckCircle
} from "lucide-react";
import { motion } from "motion/react";

interface ToolsSelectionProps {
  navigateTo: (page: Page) => void;
  agentConfig: AgentConfig;
  updateAgentConfig: (updates: Partial<AgentConfig>) => void;
}

const toolCategories = {
  "Data & Analytics": [
    {
      id: "database-query",
      name: "Database Query",
      description: "Execute SQL queries and retrieve data from databases",
      icon: Database,
      provider: "MCP Core",
      popular: true,
      config: { maxRows: 1000, timeout: 30 }
    },
    {
      id: "data-visualization",
      name: "Data Visualization", 
      description: "Create charts, graphs, and visual representations of data",
      icon: BarChart,
      provider: "MCP Analytics",
      popular: true,
      config: { chartTypes: ["bar", "line", "pie"], export: true }
    },
    {
      id: "excel-processor",
      name: "Excel Processor",
      description: "Read, write, and manipulate Excel spreadsheets",
      icon: FileText,
      provider: "MCP Office",
      popular: false,
      config: { maxRows: 10000, formulas: true }
    }
  ],
  "Web & API": [
    {
      id: "web-scraper",
      name: "Web Scraper",
      description: "Extract data from websites and web pages",
      icon: Globe,
      provider: "MCP Web",
      popular: true,
      config: { respectRobots: true, maxPages: 50 }
    },
    {
      id: "api-client",
      name: "API Client",
      description: "Make HTTP requests to REST APIs and webhooks",
      icon: Code,
      provider: "MCP Core",
      popular: true,
      config: { timeout: 30, retries: 3 }
    },
    {
      id: "web-search",
      name: "Web Search",
      description: "Search the internet for information and results",
      icon: Search,
      provider: "MCP Search",
      popular: true,
      config: { maxResults: 10, safeSearch: true }
    }
  ],
  "Communication": [
    {
      id: "email-sender",
      name: "Email Sender",
      description: "Send emails with attachments and templates",
      icon: Mail,
      provider: "MCP Email",
      popular: true,
      config: { templates: true, attachments: true }
    },
    {
      id: "slack-messenger",
      name: "Slack Messenger",
      description: "Send messages and notifications to Slack channels",
      icon: MessageSquare,
      provider: "MCP Slack",
      popular: false,
      config: { channels: [], mentions: true }
    },
    {
      id: "calendar-scheduler",
      name: "Calendar Scheduler",
      description: "Create and manage calendar events and meetings",
      icon: Calendar,
      provider: "MCP Calendar",
      popular: false,
      config: { providers: ["google", "outlook"], reminders: true }
    }
  ],
  "Content & Media": [
    {
      id: "image-generator",
      name: "Image Generator",
      description: "Generate images using AI models like DALL-E",
      icon: Image,
      provider: "MCP AI",
      popular: true,
      config: { styles: ["realistic", "artistic"], sizes: ["512x512", "1024x1024"] }
    },
    {
      id: "text-to-speech",
      name: "Text to Speech",
      description: "Convert text to natural-sounding speech audio",
      icon: Music,
      provider: "MCP Audio",
      popular: false,
      config: { voices: ["male", "female"], languages: ["en", "es", "fr"] }
    },
    {
      id: "video-processor",
      name: "Video Processor", 
      description: "Edit, convert, and process video files",
      icon: Video,
      provider: "MCP Media",
      popular: false,
      config: { formats: ["mp4", "avi", "mov"], maxSize: "100MB" }
    }
  ],
  "Utilities": [
    {
      id: "calculator",
      name: "Calculator",
      description: "Perform mathematical calculations and expressions",
      icon: Calculator,
      provider: "MCP Core",
      popular: true,
      config: { precision: 10, scientificNotation: true }
    },
    {
      id: "file-manager",
      name: "File Manager",
      description: "Read, write, and organize files and folders",
      icon: FileText,
      provider: "MCP Core",
      popular: true,
      config: { maxFileSize: "10MB", allowedTypes: ["txt", "json", "csv"] }
    },
    {
      id: "location-services",
      name: "Location Services",
      description: "Get location data, maps, and geographical information",
      icon: Map,
      provider: "MCP Location",
      popular: false,
      config: { geocoding: true, radius: 50 }
    }
  ]
};

export function ToolsSelection({ navigateTo, agentConfig, updateAgentConfig }: ToolsSelectionProps) {
  const [selectedTools, setSelectedTools] = useState<string[]>(
    agentConfig.tools.map(tool => tool.id)
  );
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("Data & Analytics");

  const allTools = Object.values(toolCategories).flat();
  
  const filteredTools = allTools.filter(tool =>
    tool.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    tool.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const toggleTool = (toolId: string) => {
    const isSelected = selectedTools.includes(toolId);
    if (isSelected) {
      setSelectedTools(prev => prev.filter(id => id !== toolId));
    } else {
      setSelectedTools(prev => [...prev, toolId]);
    }
  };

  const handleNext = () => {
    const selectedToolObjects = allTools.filter(tool => selectedTools.includes(tool.id));
    updateAgentConfig({
      tools: selectedToolObjects.map(tool => ({
        id: tool.id,
        name: tool.name,
        description: tool.description,
        type: tool.provider,
        config: tool.config
      }))
    });
    navigateTo("actions-configuration");
  };

  const ToolCard = ({ tool }: { tool: any }) => {
    const isSelected = selectedTools.includes(tool.id);
    const IconComponent = tool.icon;

    return (
      <motion.div
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        className={`border rounded-lg p-4 cursor-pointer transition-all ${
          isSelected
            ? "border-indigo-500 bg-indigo-50"
            : "border-slate-200 hover:border-slate-300 hover:shadow-md"
        }`}
        onClick={() => toggleTool(tool.id)}
      >
        <div className="flex items-start space-x-3">
          <Checkbox
            checked={isSelected}
            onChange={() => {}}
            className="mt-1"
          />
          <div className="flex-1 space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <IconComponent className="h-5 w-5 text-indigo-600" />
                <h4>{tool.name}</h4>
                {tool.popular && (
                  <Badge className="bg-gradient-to-r from-orange-500 to-red-600 text-xs">
                    Popular
                  </Badge>
                )}
              </div>
              <Badge variant="outline" className="text-xs">
                {tool.provider}
              </Badge>
            </div>
            <p className="text-sm text-muted-foreground">
              {tool.description}
            </p>
          </div>
        </div>
      </motion.div>
    );
  };

  return (
    <div className="min-h-screen p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex items-center gap-4 mb-6">
            <Button variant="outline" onClick={() => navigateTo("model-selection")}>
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back
            </Button>
            <div className="flex-1">
              <h1 className="mb-1 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                Select Tools (MCP)
              </h1>
              <p className="text-muted-foreground">Step 3 of 4: Choose capabilities for your agent</p>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="bg-slate-200 rounded-full h-2 mb-6">
            <motion.div 
              initial={{ width: "50%" }}
              animate={{ width: "75%" }}
              className="bg-gradient-to-r from-indigo-500 to-purple-600 h-2 rounded-full"
              transition={{ duration: 0.5 }}
            />
          </div>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Tools Selection */}
          <div className="lg:col-span-2 space-y-6">
            {/* Search */}
            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <Card className="bg-white/70 backdrop-blur-sm border-white/50">
                <CardContent className="p-4">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
                    <Input
                      placeholder="Search tools by name or capability..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-10 bg-white/50"
                    />
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            {/* Tools by Category */}
            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
            >
              <Card className="bg-white/70 backdrop-blur-sm border-white/50">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Wrench className="h-5 w-5" />
                    Available Tools
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {searchQuery ? (
                    <div className="space-y-4">
                      <h3>Search Results ({filteredTools.length})</h3>
                      {filteredTools.map((tool) => (
                        <ToolCard key={tool.id} tool={tool} />
                      ))}
                    </div>
                  ) : (
                    <Tabs value={selectedCategory} onValueChange={setSelectedCategory}>
                      <TabsList className="grid w-full grid-cols-3 lg:grid-cols-5">
                        {Object.keys(toolCategories).map((category) => (
                          <TabsTrigger key={category} value={category} className="text-xs">
                            {category.split(" ")[0]}
                          </TabsTrigger>
                        ))}
                      </TabsList>
                      
                      {Object.entries(toolCategories).map(([category, tools]) => (
                        <TabsContent key={category} value={category} className="space-y-4 mt-6">
                          {tools.map((tool) => (
                            <ToolCard key={tool.id} tool={tool} />
                          ))}
                        </TabsContent>
                      ))}
                    </Tabs>
                  )}
                </CardContent>
              </Card>
            </motion.div>
          </div>

          {/* Selected Tools Summary */}
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.6 }}
            className="space-y-6"
          >
            <Card className="bg-white/70 backdrop-blur-sm border-white/50 sticky top-6">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle className="h-5 w-5" />
                  Selected Tools ({selectedTools.length})
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {selectedTools.length === 0 ? (
                  <div className="text-center py-6">
                    <Wrench className="mx-auto h-12 w-12 text-muted-foreground mb-3" />
                    <p className="text-sm text-muted-foreground">
                      No tools selected yet. Choose tools to give your agent capabilities.
                    </p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {selectedTools.map((toolId) => {
                      const tool = allTools.find(t => t.id === toolId);
                      if (!tool) return null;
                      const IconComponent = tool.icon;
                      
                      return (
                        <div key={toolId} className="flex items-center gap-3 p-2 bg-slate-50 rounded">
                          <IconComponent className="h-4 w-4 text-indigo-600" />
                          <div className="flex-1">
                            <div className="text-sm">{tool.name}</div>
                            <div className="text-xs text-muted-foreground">{tool.provider}</div>
                          </div>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => toggleTool(toolId)}
                            className="text-red-500 hover:text-red-700"
                          >
                            ×
                          </Button>
                        </div>
                      );
                    })}
                  </div>
                )}

                {selectedTools.length > 0 && (
                  <div className="pt-4 border-t">
                    <div className="text-sm text-muted-foreground">
                      Selected {selectedTools.length} of {allTools.length} available tools
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
              <Button 
                onClick={handleNext}
                className="w-full bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700"
                size="lg"
              >
                Next: Configure Actions
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </motion.div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}