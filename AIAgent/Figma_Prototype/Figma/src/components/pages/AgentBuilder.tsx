import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Textarea } from "../ui/textarea";
import { Badge } from "../ui/badge";
import { Switch } from "../ui/switch";
import { Label } from "../ui/label";
import { Avatar, AvatarFallback } from "../ui/avatar";
import { Page, AgentConfig } from "../../App";
import { 
  ArrowLeft, 
  ArrowRight, 
  Bot, 
  Sparkles, 
  Globe, 
  Lock,
  Upload,
  Wand2
} from "lucide-react";
import { motion } from "motion/react";

interface AgentBuilderProps {
  navigateTo: (page: Page) => void;
  agentConfig: AgentConfig;
  updateAgentConfig: (updates: Partial<AgentConfig>) => void;
}

const avatarOptions = ["🤖", "🧠", "⚡", "🔥", "💎", "🚀", "🎯", "💡", "🌟", "🎨", "📊", "🔬"];
const suggestedTags = ["Analytics", "Creative", "Development", "Support", "Research", "Automation", "Writing", "Data", "Marketing", "Finance"];

export function AgentBuilder({ navigateTo, agentConfig, updateAgentConfig }: AgentBuilderProps) {
  const [selectedAvatar, setSelectedAvatar] = useState(agentConfig.avatar || "🤖");
  const [newTag, setNewTag] = useState("");

  const handleNext = () => {
    if (agentConfig.name.trim()) {
      updateAgentConfig({ avatar: selectedAvatar });
      navigateTo("model-selection");
    }
  };

  const addTag = (tag: string) => {
    if (tag && !agentConfig.tags.includes(tag)) {
      updateAgentConfig({ tags: [...agentConfig.tags, tag] });
    }
    setNewTag("");
  };

  const removeTag = (tagToRemove: string) => {
    updateAgentConfig({ 
      tags: agentConfig.tags.filter(tag => tag !== tagToRemove) 
    });
  };

  const generateRandomName = () => {
    const adjectives = ["Smart", "Advanced", "Pro", "Elite", "Expert", "Dynamic", "Intelligent", "Powerful"];
    const nouns = ["Assistant", "Agent", "Bot", "Helper", "Companion", "Specialist", "Advisor", "Expert"];
    const randomAdj = adjectives[Math.floor(Math.random() * adjectives.length)];
    const randomNoun = nouns[Math.floor(Math.random() * nouns.length)];
    updateAgentConfig({ name: `${randomAdj} ${randomNoun}` });
  };

  const isFormValid = agentConfig.name.trim().length > 0 && agentConfig.description.trim().length > 0;

  return (
    <div className="min-h-screen p-6">
      <div className="max-w-4xl mx-auto">
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
              <h1 className="mb-1 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                Create New Agent
              </h1>
              <p className="text-muted-foreground">Step 1 of 4: Basic Information</p>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="bg-slate-200 rounded-full h-2 mb-6">
            <motion.div 
              initial={{ width: 0 }}
              animate={{ width: "25%" }}
              className="bg-gradient-to-r from-indigo-500 to-purple-600 h-2 rounded-full"
              transition={{ duration: 0.5 }}
            />
          </div>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Form */}
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="lg:col-span-2 space-y-6"
          >
            {/* Basic Info Card */}
            <Card className="bg-white/70 backdrop-blur-sm border-white/50">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Bot className="h-5 w-5" />
                  Agent Details
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Label htmlFor="name">Agent Name</Label>
                    <Button 
                      variant="ghost" 
                      size="sm" 
                      onClick={generateRandomName}
                      className="text-xs"
                    >
                      <Wand2 className="mr-1 h-3 w-3" />
                      Generate
                    </Button>
                  </div>
                  <Input
                    id="name"
                    placeholder="e.g., Data Analysis Assistant"
                    value={agentConfig.name}
                    onChange={(e) => updateAgentConfig({ name: e.target.value })}
                    className="bg-white/50"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="description">Description</Label>
                  <Textarea
                    id="description"
                    placeholder="Describe what your agent does and its key capabilities..."
                    value={agentConfig.description}
                    onChange={(e) => updateAgentConfig({ description: e.target.value })}
                    rows={4}
                    className="bg-white/50"
                  />
                </div>

                <div className="space-y-3">
                  <Label>Avatar</Label>
                  <div className="grid grid-cols-6 gap-3">
                    {avatarOptions.map((avatar) => (
                      <motion.button
                        key={avatar}
                        whileHover={{ scale: 1.1 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={() => setSelectedAvatar(avatar)}
                        className={`w-12 h-12 rounded-lg border-2 flex items-center justify-center text-xl transition-all ${
                          selectedAvatar === avatar
                            ? "border-indigo-500 bg-indigo-50"
                            : "border-slate-200 hover:border-slate-300"
                        }`}
                      >
                        {avatar}
                      </motion.button>
                    ))}
                  </div>
                </div>

                <div className="space-y-3">
                  <Label>Tags</Label>
                  <div className="flex flex-wrap gap-2 mb-3">
                    {agentConfig.tags.map((tag) => (
                      <Badge 
                        key={tag} 
                        variant="secondary" 
                        className="cursor-pointer hover:bg-red-100 hover:text-red-800"
                        onClick={() => removeTag(tag)}
                      >
                        {tag} ×
                      </Badge>
                    ))}
                  </div>
                  <div className="flex gap-2">
                    <Input
                      placeholder="Add a tag..."
                      value={newTag}
                      onChange={(e) => setNewTag(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && addTag(newTag)}
                      className="bg-white/50"
                    />
                    <Button 
                      onClick={() => addTag(newTag)}
                      disabled={!newTag.trim()}
                      variant="outline"
                    >
                      Add
                    </Button>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {suggestedTags
                      .filter(tag => !agentConfig.tags.includes(tag))
                      .slice(0, 5)
                      .map((tag) => (
                        <Button
                          key={tag}
                          variant="ghost"
                          size="sm"
                          onClick={() => addTag(tag)}
                          className="text-xs"
                        >
                          + {tag}
                        </Button>
                      ))}
                  </div>
                </div>

                <div className="flex items-center justify-between p-4 bg-slate-50 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <div className="flex items-center gap-2">
                      {agentConfig.isPublic ? <Globe className="h-4 w-4" /> : <Lock className="h-4 w-4" />}
                      <Label htmlFor="public">Make agent public</Label>
                    </div>
                  </div>
                  <Switch
                    id="public"
                    checked={agentConfig.isPublic}
                    onCheckedChange={(checked) => updateAgentConfig({ isPublic: checked })}
                  />
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Preview Card */}
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
            className="space-y-6"
          >
            <Card className="bg-white/70 backdrop-blur-sm border-white/50 sticky top-6">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Sparkles className="h-5 w-5" />
                  Preview
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="text-center">
                  <Avatar className="h-16 w-16 mx-auto mb-3 bg-gradient-to-br from-indigo-500 to-purple-600">
                    <AvatarFallback className="bg-gradient-to-br from-indigo-500 to-purple-600 text-white text-2xl">
                      {selectedAvatar}
                    </AvatarFallback>
                  </Avatar>
                  <h3 className="mb-1">
                    {agentConfig.name || "Untitled Agent"}
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    {agentConfig.description || "No description provided"}
                  </p>
                </div>

                {agentConfig.tags.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {agentConfig.tags.map((tag) => (
                      <Badge key={tag} variant="secondary" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                )}

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className="bg-slate-50 rounded p-3 text-center">
                    <div className="font-medium">0</div>
                    <div className="text-muted-foreground">Tools</div>
                  </div>
                  <div className="bg-slate-50 rounded p-3 text-center">
                    <div className="font-medium">0</div>
                    <div className="text-muted-foreground">Actions</div>
                  </div>
                </div>

                <div className="flex items-center gap-2 text-sm">
                  {agentConfig.isPublic ? (
                    <>
                      <Globe className="h-4 w-4 text-green-600" />
                      <span className="text-green-600">Public</span>
                    </>
                  ) : (
                    <>
                      <Lock className="h-4 w-4 text-slate-600" />
                      <span className="text-slate-600">Private</span>
                    </>
                  )}
                </div>
              </CardContent>
            </Card>

            <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
              <Button 
                onClick={handleNext}
                disabled={!isFormValid}
                className="w-full bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700"
                size="lg"
              >
                Next: Select Model
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </motion.div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}