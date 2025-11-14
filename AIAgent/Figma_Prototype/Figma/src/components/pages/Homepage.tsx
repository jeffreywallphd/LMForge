import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { Avatar, AvatarFallback, AvatarImage } from "../ui/avatar";
import { Input } from "../ui/input";
import { Page } from "../../App";
import { 
  Plus, 
  Search, 
  Bot, 
  Brain, 
  Zap, 
  Star, 
  Users, 
  Clock,
  Filter,
  MoreVertical,
  Play,
  Settings,
  Globe
} from "lucide-react";
import { motion } from "motion/react";

interface HomepageProps {
  navigateTo: (page: Page) => void;
  resetAgentConfig: () => void;
}

const mockAgents = [
  {
    id: "1",
    name: "Crawler Agent",
    description: "Fetches raw HTML content from any URL using HTTP requests",
    avatar: "🕷️",
    tags: ["Web", "HTTP", "Fetching"],
    model: "MCP",
    tools: 1,
    actions: 1,
    isPublic: true,
    lastUsed: "Ready",
    performance: 100,
    runs: 0,
    agentType: "crawler"
  },
  {
    id: "2", 
    name: "Parser Agent",
    description: "Extracts clean text from HTML using BeautifulSoup4 parsing",
    avatar: "📄",
    tags: ["HTML", "Parsing", "Text Extraction"],
    model: "MCP",
    tools: 1,
    actions: 1,
    isPublic: true,
    lastUsed: "Ready",
    performance: 100,
    runs: 0,
    agentType: "parser"
  },
  {
    id: "3",
    name: "Combined Flow",
    description: "Complete workflow: Crawl URL → Parse HTML → Extract clean text",
    avatar: "🔄",
    tags: ["Workflow", "Pipeline", "Combined"],
    model: "MCP",
    tools: 2,
    actions: 3,
    isPublic: true,
    lastUsed: "Ready",
    performance: 100,
    runs: 0,
    agentType: "combined"
  }
];

export function Homepage({ navigateTo, resetAgentConfig }: HomepageProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedFilter, setSelectedFilter] = useState("all");

  const handleCreateAgent = () => {
    resetAgentConfig();
    navigateTo("agent-creator");
  };

  const handleManageAgents = () => {
    navigateTo("agent-management");
  };

  const filteredAgents = mockAgents.filter(agent => {
    const matchesSearch = agent.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         agent.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         agent.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    
    const matchesFilter = selectedFilter === "all" || 
                         (selectedFilter === "public" && agent.isPublic) ||
                         (selectedFilter === "private" && !agent.isPublic);
    
    return matchesSearch && matchesFilter;
  });

  return (
    <div className="min-h-screen p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="mb-2 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                AI Agent Hub
              </h1>
              <p className="text-muted-foreground">
                Build, deploy, and manage intelligent AI agents for any task
              </p>
            </div>
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Button 
                onClick={() => navigateTo("url-extractor")} 
                size="lg" 
                className="bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700"
              >
                <Globe className="mr-2 h-5 w-5" />
                Extract URL
              </Button>
            </motion.div>
          </div>

          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <motion.div whileHover={{ scale: 1.02 }}>
              <Card className="bg-gradient-to-br from-blue-50 to-indigo-100 border-blue-200">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-blue-600">Total Agents</p>
                      <p className="font-semibold text-blue-800">3</p>
                    </div>
                    <Bot className="h-8 w-8 text-blue-500" />
                  </div>
                </CardContent>
              </Card>
            </motion.div>
            
            <motion.div whileHover={{ scale: 1.02 }}>
              <Card className="bg-gradient-to-br from-green-50 to-emerald-100 border-green-200">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-green-600">Active Agents</p>
                      <p className="font-semibold text-green-800">3</p>
                    </div>
                    <Zap className="h-8 w-8 text-green-500" />
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div whileHover={{ scale: 1.02 }}>
              <Card className="bg-gradient-to-br from-purple-50 to-violet-100 border-purple-200">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-purple-600">MCP Protocol</p>
                      <p className="font-semibold text-purple-800">Ready</p>
                    </div>
                    <Brain className="h-8 w-8 text-purple-500" />
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div whileHover={{ scale: 1.02 }}>
              <Card className="bg-gradient-to-br from-orange-50 to-red-100 border-orange-200">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-orange-600">Success Rate</p>
                      <p className="font-semibold text-orange-800">100%</p>
                    </div>
                    <Star className="h-8 w-8 text-orange-500" />
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </div>
        </motion.div>

        {/* Search and Filters */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mb-6"
        >
          <Card className="p-4">
            <div className="flex flex-col md:flex-row gap-4">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
                <Input
                  placeholder="Search agents by name, description, or tags..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10"
                />
              </div>
              <div className="flex gap-2">
                <Button
                  variant={selectedFilter === "all" ? "default" : "outline"}
                  onClick={() => setSelectedFilter("all")}
                  size="sm"
                >
                  All
                </Button>
                <Button
                  variant={selectedFilter === "public" ? "default" : "outline"}
                  onClick={() => setSelectedFilter("public")}
                  size="sm"
                >
                  <Users className="mr-1 h-3 w-3" />
                  Public
                </Button>
                <Button
                  variant={selectedFilter === "private" ? "default" : "outline"}
                  onClick={() => setSelectedFilter("private")}
                  size="sm"
                >
                  Private
                </Button>
              </div>
            </div>
          </Card>
        </motion.div>

        {/* Agents Grid */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6"
        >
          {filteredAgents.map((agent, index) => (
            <motion.div
              key={agent.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 * index }}
              whileHover={{ scale: 1.02 }}
              className="group"
            >
              <Card className="h-full hover:shadow-lg transition-all duration-300 bg-white/70 backdrop-blur-sm border-white/50">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <Avatar className="h-12 w-12 bg-gradient-to-br from-indigo-500 to-purple-600">
                      <AvatarImage src="" />
                      <AvatarFallback className="bg-gradient-to-br from-indigo-500 to-purple-600 text-white">
                        {agent.avatar}
                      </AvatarFallback>
                    </Avatar>
                    <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                      <Button variant="ghost" size="sm">
                        <MoreVertical className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                  <div>
                    <CardTitle className="mb-1">{agent.name}</CardTitle>
                    <p className="text-sm text-muted-foreground line-clamp-2">
                      {agent.description}
                    </p>
                  </div>
                </CardHeader>
                
                <CardContent className="space-y-4">
                  <div className="flex flex-wrap gap-1">
                    {agent.tags.map((tag) => (
                      <Badge key={tag} variant="secondary" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                  </div>

                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="flex items-center">
                      <Brain className="mr-1 h-3 w-3 text-muted-foreground" />
                      <span className="text-muted-foreground">{agent.model}</span>
                    </div>
                    <div className="flex items-center">
                      <Clock className="mr-1 h-3 w-3 text-muted-foreground" />
                      <span className="text-muted-foreground">{agent.lastUsed}</span>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-2 text-xs text-center">
                    <div className="bg-slate-50 rounded p-2">
                      <div className="font-medium">{agent.tools}</div>
                      <div className="text-muted-foreground">Tools</div>
                    </div>
                    <div className="bg-slate-50 rounded p-2">
                      <div className="font-medium">{agent.actions}</div>
                      <div className="text-muted-foreground">Actions</div>
                    </div>
                    <div className="bg-slate-50 rounded p-2">
                      <div className="font-medium">{agent.performance}%</div>
                      <div className="text-muted-foreground">Success</div>
                    </div>
                  </div>

                  <div className="flex gap-2 pt-2">
                    <Button 
                      size="sm" 
                      className="flex-1" 
                      variant="outline"
                      onClick={() => navigateTo("url-extractor")}
                    >
                      <Play className="mr-1 h-3 w-3" />
                      Run
                    </Button>
                    <Button 
                      size="sm" 
                      className="flex-1" 
                      variant="outline"
                      onClick={() => navigateTo("url-extractor")}
                    >
                      <Settings className="mr-1 h-3 w-3" />
                      Configure
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </motion.div>

        {/* Empty State */}
        {filteredAgents.length === 0 && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-12"
          >
            <Bot className="mx-auto h-16 w-16 text-muted-foreground mb-4" />
            <h3 className="mb-2">No agents found</h3>
            <p className="text-muted-foreground mb-6">
              {searchQuery ? "Try adjusting your search terms" : "Get started by creating your first AI agent"}
            </p>
            <Button onClick={handleCreateAgent} className="bg-gradient-to-r from-indigo-500 to-purple-600">
              <Plus className="mr-2 h-4 w-4" />
              Create Your First Agent
            </Button>
          </motion.div>
        )}
      </div>
    </div>
  );
}