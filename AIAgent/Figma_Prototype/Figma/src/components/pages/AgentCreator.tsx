import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Textarea } from "../ui/textarea";
import { Badge } from "../ui/badge";
import { Switch } from "../ui/switch";
import { Label } from "../ui/label";
import { Page } from "../../App";
import { 
  ArrowLeft, 
  Save,
  Plus,
  X,
  Code,
  Play,
  Trash2,
  Edit,
  Check,
  Loader2
} from "lucide-react";
import { motion } from "motion/react";

interface AgentCreatorProps {
  navigateTo: (page: Page) => void;
}

interface Tool {
  id: string;
  name: string;
  description: string;
  type: string;
  config: any;
}

interface Agent {
  id?: string;
  name: string;
  description: string;
  tools: Tool[];
  code: string;
  config: any;
  avatar: string;
  tags: string[];
  isPublic: boolean;
}

// Use proxy instead of direct connection to avoid VPN/proxy interception
// The vite.config.ts proxy will handle routing /api to backend
const API_URL = "/api";

export function AgentCreator({ navigateTo }: AgentCreatorProps) {
  const [agent, setAgent] = useState<Agent>({
    name: "",
    description: "",
    tools: [],
    code: "",
    config: {},
    avatar: "🤖",
    tags: [],
    isPublic: false
  });

  const [newTool, setNewTool] = useState<Partial<Tool>>({
    name: "",
    description: "",
    type: "function",
    config: {}
  });

  const [newTag, setNewTag] = useState("");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const avatarOptions = ["🤖", "🧠", "⚡", "🔥", "💎", "🚀", "🎯", "💡", "🌟", "🎨", "📊", "🔬"];

  const toolTypes = [
    { value: "function", label: "Function" },
    { value: "http_request", label: "HTTP Request" },
    { value: "data_transform", label: "Data Transform" },
    { value: "custom", label: "Custom Code" }
  ];

  const addTool = () => {
    if (!newTool.name || !newTool.description) {
      setError("Tool name and description are required");
      return;
    }

    const tool: Tool = {
      id: `tool_${Date.now()}`,
      name: newTool.name || "",
      description: newTool.description || "",
      type: newTool.type || "function",
      config: newTool.config || {}
    };

    setAgent(prev => ({
      ...prev,
      tools: [...prev.tools, tool]
    }));

    setNewTool({
      name: "",
      description: "",
      type: "function",
      config: {}
    });
    setError("");
  };

  const removeTool = (toolId: string) => {
    setAgent(prev => ({
      ...prev,
      tools: prev.tools.filter(t => t.id !== toolId)
    }));
  };

  const addTag = (tag: string) => {
    if (tag && !agent.tags.includes(tag)) {
      setAgent(prev => ({
        ...prev,
        tags: [...prev.tags, tag]
      }));
    }
    setNewTag("");
  };

  const removeTag = (tag: string) => {
    setAgent(prev => ({
      ...prev,
      tags: prev.tags.filter(t => t !== tag)
    }));
  };

  const handleSave = async () => {
    if (!agent.name || !agent.description) {
      setError("Name and description are required");
      return;
    }

    setSaving(true);
    setError("");
    setSuccess("");

    try {
      console.log("Creating agent:", { ...agent, createdBy: "user" });
      console.log("API URL:", `${API_URL}/agents`);
      
      const response = await fetch(`${API_URL}/agents`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          ...agent,
          createdBy: "user"
        })
      });

      console.log("Response status:", response.status);
      console.log("Response ok:", response.ok);

      if (!response.ok) {
        let errorMessage = "Failed to create agent";
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorData.message || errorMessage;
        } catch (e) {
          const text = await response.text();
          errorMessage = `Server error (${response.status}): ${text.substring(0, 100)}`;
        }
        throw new Error(errorMessage);
      }

      const createdAgent = await response.json();
      setSuccess(`Agent "${createdAgent.name}" created successfully!`);
      setAgent({
        name: "",
        description: "",
        tools: [],
        code: "",
        config: {},
        avatar: "🤖",
        tags: [],
        isPublic: false
      });

      // Clear success message after 3 seconds
      setTimeout(() => setSuccess(""), 3000);
    } catch (err: any) {
      setError(err.message || "Failed to create agent");
    } finally {
      setSaving(false);
    }
  };

  const isFormValid = agent.name.trim().length > 0 && agent.description.trim().length > 0;

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
            <Button variant="outline" onClick={() => navigateTo("homepage")}>
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Hub
            </Button>
            <div className="flex-1">
              <h1 className="text-3xl font-bold mb-1 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                Create Custom Agent
              </h1>
              <p className="text-muted-foreground">Build your own MCP agent with custom tools and capabilities</p>
            </div>
          </div>
        </motion.div>

        {/* Success/Error Messages */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-800"
          >
            {error}
          </motion.div>
        )}

        {success && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-4 p-4 bg-green-50 border border-green-200 rounded-lg text-green-800 flex items-center gap-2"
          >
            <Check className="h-5 w-5" />
            {success}
          </motion.div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Form */}
          <div className="lg:col-span-2 space-y-6">
            {/* Basic Info */}
            <Card>
              <CardHeader>
                <CardTitle>Agent Information</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Agent Name *</Label>
                  <Input
                    placeholder="e.g., Data Analyzer Agent"
                    value={agent.name}
                    onChange={(e) => setAgent(prev => ({ ...prev, name: e.target.value }))}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Description *</Label>
                  <Textarea
                    placeholder="Describe what your agent does..."
                    value={agent.description}
                    onChange={(e) => setAgent(prev => ({ ...prev, description: e.target.value }))}
                    rows={3}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Avatar</Label>
                  <div className="grid grid-cols-6 gap-2">
                    {avatarOptions.map((avatar) => (
                      <button
                        key={avatar}
                        onClick={() => setAgent(prev => ({ ...prev, avatar }))}
                        className={`w-12 h-12 rounded-lg border-2 flex items-center justify-center text-xl transition-all ${
                          agent.avatar === avatar
                            ? "border-indigo-500 bg-indigo-50"
                            : "border-slate-200 hover:border-slate-300"
                        }`}
                      >
                        {avatar}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Tags</Label>
                  <div className="flex flex-wrap gap-2 mb-2">
                    {agent.tags.map((tag) => (
                      <Badge key={tag} variant="secondary" className="cursor-pointer" onClick={() => removeTag(tag)}>
                        {tag} <X className="ml-1 h-3 w-3 inline" />
                      </Badge>
                    ))}
                  </div>
                  <div className="flex gap-2">
                    <Input
                      placeholder="Add a tag..."
                      value={newTag}
                      onChange={(e) => setNewTag(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && addTag(newTag)}
                    />
                    <Button onClick={() => addTag(newTag)} variant="outline">
                      Add
                    </Button>
                  </div>
                </div>

                <div className="flex items-center justify-between p-4 bg-slate-50 rounded-lg">
                  <Label>Make agent public</Label>
                  <Switch
                    checked={agent.isPublic}
                    onCheckedChange={(checked) => setAgent(prev => ({ ...prev, isPublic: checked }))}
                  />
                </div>
              </CardContent>
            </Card>

            {/* Tools */}
            <Card>
              <CardHeader>
                <CardTitle>Tools</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Add New Tool */}
                <div className="p-4 border-2 border-dashed border-slate-200 rounded-lg space-y-4">
                  <h3 className="font-semibold">Add New Tool</h3>
                  <div className="space-y-2">
                    <Label>Tool Name *</Label>
                    <Input
                      placeholder="e.g., fetch_data"
                      value={newTool.name || ""}
                      onChange={(e) => setNewTool(prev => ({ ...prev, name: e.target.value }))}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Description *</Label>
                    <Input
                      placeholder="What does this tool do?"
                      value={newTool.description || ""}
                      onChange={(e) => setNewTool(prev => ({ ...prev, description: e.target.value }))}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Tool Type</Label>
                    <select
                      className="w-full p-2 border rounded-lg"
                      value={newTool.type || "function"}
                      onChange={(e) => setNewTool(prev => ({ ...prev, type: e.target.value }))}
                    >
                      {toolTypes.map(type => (
                        <option key={type.value} value={type.value}>{type.label}</option>
                      ))}
                    </select>
                  </div>
                  <Button onClick={addTool} className="w-full" variant="outline">
                    <Plus className="mr-2 h-4 w-4" />
                    Add Tool
                  </Button>
                </div>

                {/* Existing Tools */}
                {agent.tools.length > 0 && (
                  <div className="space-y-2">
                    <Label>Your Tools ({agent.tools.length})</Label>
                    {agent.tools.map((tool) => (
                      <div key={tool.id} className="p-4 border rounded-lg flex items-start justify-between">
                        <div className="flex-1">
                          <div className="font-semibold">{tool.name}</div>
                          <div className="text-sm text-muted-foreground">{tool.description}</div>
                          <Badge variant="outline" className="mt-2">{tool.type}</Badge>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => removeTool(tool.id)}
                          className="text-red-600 hover:text-red-700"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Preview & Actions */}
          <div className="space-y-6">
            <Card className="sticky top-6">
              <CardHeader>
                <CardTitle>Preview</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="text-center">
                  <div className="text-4xl mb-2">{agent.avatar}</div>
                  <h3 className="font-semibold">{agent.name || "Untitled Agent"}</h3>
                  <p className="text-sm text-muted-foreground">{agent.description || "No description"}</p>
                </div>

                {agent.tags.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {agent.tags.map(tag => (
                      <Badge key={tag} variant="secondary" className="text-xs">{tag}</Badge>
                    ))}
                  </div>
                )}

                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div className="bg-slate-50 rounded p-2 text-center">
                    <div className="font-semibold">{agent.tools.length}</div>
                    <div className="text-muted-foreground">Tools</div>
                  </div>
                  <div className="bg-slate-50 rounded p-2 text-center">
                    <div className="font-semibold">{agent.tags.length}</div>
                    <div className="text-muted-foreground">Tags</div>
                  </div>
                </div>

                <Button
                  onClick={handleSave}
                  disabled={!isFormValid || saving}
                  className="w-full bg-gradient-to-r from-indigo-500 to-purple-600"
                  size="lg"
                >
                  {saving ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Saving...
                    </>
                  ) : (
                    <>
                      <Save className="mr-2 h-4 w-4" />
                      Create Agent
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}

