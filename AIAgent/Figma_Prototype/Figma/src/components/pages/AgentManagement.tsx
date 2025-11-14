import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { Page } from "../../App";
import { 
  ArrowLeft, 
  Plus,
  Edit,
  Trash2,
  Play,
  Code,
  Globe,
  Lock,
  Loader2,
  RefreshCw
} from "lucide-react";
import { motion } from "motion/react";

interface AgentManagementProps {
  navigateTo: (page: Page) => void;
}

interface Agent {
  id: string;
  name: string;
  description: string;
  tools: any[];
  avatar: string;
  tags: string[];
  isPublic: boolean;
  createdAt: string;
  updatedAt: string;
}

// Use proxy instead of direct connection to avoid VPN/proxy interception
// The vite.config.ts proxy will handle routing /api to backend
const API_URL = "/api";

export function AgentManagement({ navigateTo }: AgentManagementProps) {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const fetchAgents = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${API_URL}/agents`);
      if (!response.ok) {
        throw new Error("Failed to fetch agents");
      }
      const data = await response.json();
      setAgents(data.agents || []);
    } catch (err: any) {
      setError(err.message || "Failed to load agents");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAgents();
  }, []);

  const handleDelete = async (agentId: string, agentName: string) => {
    if (!confirm(`Are you sure you want to delete "${agentName}"?`)) {
      return;
    }

    setDeletingId(agentId);
    try {
      const response = await fetch(`${API_URL}/agents/${agentId}`, {
        method: "DELETE"
      });

      if (!response.ok) {
        throw new Error("Failed to delete agent");
      }

      // Remove from list
      setAgents(prev => prev.filter(a => a.id !== agentId));
    } catch (err: any) {
      alert(err.message || "Failed to delete agent");
    } finally {
      setDeletingId(null);
    }
  };

  const handleViewCode = async (agentId: string) => {
    try {
      const response = await fetch(`${API_URL}/agents/${agentId}/code`);
      if (!response.ok) {
        throw new Error("Failed to fetch code");
      }
      const code = await response.text();
      
      // Open code in new window
      const newWindow = window.open();
      if (newWindow) {
        newWindow.document.write(`
          <html>
            <head>
              <title>Agent Code - ${agentId}</title>
              <style>
                body { font-family: monospace; padding: 20px; background: #1e1e1e; color: #d4d4d4; }
                pre { background: #252526; padding: 15px; border-radius: 5px; overflow-x: auto; }
              </style>
            </head>
            <body>
              <h1>Agent Code</h1>
              <pre><code>${code.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</code></pre>
            </body>
          </html>
        `);
      }
    } catch (err: any) {
      alert(err.message || "Failed to view code");
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen p-6 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-indigo-600" />
          <p className="text-muted-foreground">Loading agents...</p>
        </div>
      </div>
    );
  }

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
            <div className="flex items-center gap-4">
              <Button variant="outline" onClick={() => navigateTo("homepage")}>
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Hub
              </Button>
              <div>
                <h1 className="text-3xl font-bold mb-1 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                  Agent Management
                </h1>
                <p className="text-muted-foreground">Manage your custom agents</p>
              </div>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" onClick={fetchAgents}>
                <RefreshCw className="mr-2 h-4 w-4" />
                Refresh
              </Button>
              <Button onClick={() => navigateTo("agent-creator")} className="bg-gradient-to-r from-indigo-500 to-purple-600">
                <Plus className="mr-2 h-4 w-4" />
                Create Agent
              </Button>
            </div>
          </div>
        </motion.div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-800">
            {error}
          </div>
        )}

        {/* Agents List */}
        {agents.length === 0 ? (
          <Card>
            <CardContent className="py-12 text-center">
              <div className="text-6xl mb-4">🤖</div>
              <h3 className="text-xl font-semibold mb-2">No Agents Yet</h3>
              <p className="text-muted-foreground mb-6">Create your first custom agent to get started</p>
              <Button onClick={() => navigateTo("agent-creator")} className="bg-gradient-to-r from-indigo-500 to-purple-600">
                <Plus className="mr-2 h-4 w-4" />
                Create Your First Agent
              </Button>
            </CardContent>
          </Card>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {agents.map((agent) => (
              <motion.div
                key={agent.id}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                whileHover={{ scale: 1.02 }}
                transition={{ duration: 0.2 }}
              >
                <Card className="h-full flex flex-col">
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-3">
                        <div className="text-3xl">{agent.avatar}</div>
                        <div className="flex-1">
                          <CardTitle className="text-lg">{agent.name}</CardTitle>
                          <div className="flex items-center gap-2 mt-1">
                            {agent.isPublic ? (
                              <Badge variant="outline" className="text-xs">
                                <Globe className="mr-1 h-3 w-3" />
                                Public
                              </Badge>
                            ) : (
                              <Badge variant="outline" className="text-xs">
                                <Lock className="mr-1 h-3 w-3" />
                                Private
                              </Badge>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="flex-1 flex flex-col">
                    <p className="text-sm text-muted-foreground mb-4 line-clamp-2">
                      {agent.description || "No description"}
                    </p>

                    {agent.tags.length > 0 && (
                      <div className="flex flex-wrap gap-1 mb-4">
                        {agent.tags.slice(0, 3).map((tag) => (
                          <Badge key={tag} variant="secondary" className="text-xs">
                            {tag}
                          </Badge>
                        ))}
                        {agent.tags.length > 3 && (
                          <Badge variant="secondary" className="text-xs">
                            +{agent.tags.length - 3}
                          </Badge>
                        )}
                      </div>
                    )}

                    <div className="grid grid-cols-2 gap-2 mb-4 text-sm">
                      <div className="bg-slate-50 rounded p-2 text-center">
                        <div className="font-semibold">{agent.tools?.length || 0}</div>
                        <div className="text-muted-foreground text-xs">Tools</div>
                      </div>
                      <div className="bg-slate-50 rounded p-2 text-center">
                        <div className="font-semibold">{agent.tags?.length || 0}</div>
                        <div className="text-muted-foreground text-xs">Tags</div>
                      </div>
                    </div>

                    <div className="mt-auto space-y-2">
                      <div className="flex gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          className="flex-1"
                          onClick={() => handleViewCode(agent.id)}
                        >
                          <Code className="mr-1 h-3 w-3" />
                          Code
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          className="flex-1"
                          onClick={() => navigateTo("agent-creator")}
                        >
                          <Edit className="mr-1 h-3 w-3" />
                          Edit
                        </Button>
                      </div>
                      <Button
                        variant="destructive"
                        size="sm"
                        className="w-full"
                        onClick={() => handleDelete(agent.id, agent.name)}
                        disabled={deletingId === agent.id}
                      >
                        {deletingId === agent.id ? (
                          <>
                            <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                            Deleting...
                          </>
                        ) : (
                          <>
                            <Trash2 className="mr-1 h-3 w-3" />
                            Delete
                          </>
                        )}
                      </Button>
                    </div>

                    <div className="mt-2 text-xs text-muted-foreground">
                      Created: {new Date(agent.createdAt).toLocaleDateString()}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

