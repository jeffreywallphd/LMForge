import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Textarea } from "../ui/textarea";
import { Badge } from "../ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "../ui/dialog";
import { Label } from "../ui/label";
import { Page, AgentConfig } from "../../App";
import { 
  ArrowLeft, 
  Plus, 
  Edit, 
  Trash2, 
  Zap, 
  MessageSquare, 
  Clock, 
  Globe,
  CheckCircle,
  Save,
  Play,
  AlertTriangle
} from "lucide-react";
import { motion } from "motion/react";
import { toast } from "sonner@2.0.3";

interface ActionsConfigurationProps {
  navigateTo: (page: Page) => void;
  agentConfig: AgentConfig;
  updateAgentConfig: (updates: Partial<AgentConfig>) => void;
}

const triggerTypes = [
  { value: "keyword", label: "Keyword/Phrase", description: "Trigger when specific words are mentioned" },
  { value: "schedule", label: "Schedule", description: "Run on a time-based schedule" },
  { value: "webhook", label: "Webhook", description: "Trigger via HTTP webhook" },
  { value: "email", label: "Email", description: "Trigger when receiving emails" },
  { value: "manual", label: "Manual", description: "Trigger manually by user" },
  { value: "conditional", label: "Conditional", description: "Trigger when conditions are met" }
];

const responseTemplates = [
  "Analyze the data and provide insights",
  "Generate a summary report", 
  "Send notification to team",
  "Update database records",
  "Create visualization",
  "Send email response",
  "Log activity to file",
  "Custom response..."
];

export function ActionsConfiguration({ navigateTo, agentConfig, updateAgentConfig }: ActionsConfigurationProps) {
  const [actions, setActions] = useState(agentConfig.actions);
  const [editingAction, setEditingAction] = useState<any>(null);
  const [isDialogOpen, setIsDialogOpen] = useState(false);

  const [actionForm, setActionForm] = useState({
    name: "",
    description: "",
    trigger: "",
    triggerType: "",
    response: ""
  });

  const handleAddAction = () => {
    setActionForm({
      name: "",
      description: "",
      trigger: "",
      triggerType: "",
      response: ""
    });
    setEditingAction(null);
    setIsDialogOpen(true);
  };

  const handleEditAction = (action: any) => {
    setActionForm({
      name: action.name,
      description: action.description,
      trigger: action.trigger,
      triggerType: action.triggerType || "manual",
      response: action.response
    });
    setEditingAction(action);
    setIsDialogOpen(true);
  };

  const handleSaveAction = () => {
    if (!actionForm.name || !actionForm.trigger || !actionForm.response) {
      toast.error("Please fill in all required fields");
      return;
    }

    const newAction = {
      id: editingAction?.id || `action-${Date.now()}`,
      name: actionForm.name,
      description: actionForm.description,
      trigger: actionForm.trigger,
      triggerType: actionForm.triggerType,
      response: actionForm.response
    };

    if (editingAction) {
      setActions(prev => prev.map(a => a.id === editingAction.id ? newAction : a));
      toast.success("Action updated successfully");
    } else {
      setActions(prev => [...prev, newAction]);
      toast.success("Action added successfully");
    }

    setIsDialogOpen(false);
    setEditingAction(null);
  };

  const handleDeleteAction = (actionId: string) => {
    setActions(prev => prev.filter(a => a.id !== actionId));
    toast.success("Action deleted");
  };

  const handleFinish = () => {
    updateAgentConfig({ actions });
    
    // Generate agent ID and save
    const finalAgent = {
      ...agentConfig,
      actions,
      id: `agent-${Date.now()}`,
      createdAt: new Date()
    };
    
    updateAgentConfig(finalAgent);
    toast.success("Agent created successfully!");
    navigateTo("homepage");
  };

  const getTriggerIcon = (triggerType: string) => {
    switch (triggerType) {
      case "keyword": return MessageSquare;
      case "schedule": return Clock;
      case "webhook": return Globe;
      case "email": return MessageSquare;
      case "manual": return Play;
      case "conditional": return AlertTriangle;
      default: return Zap;
    }
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
            <Button variant="outline" onClick={() => navigateTo("tools-selection")}>
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back
            </Button>
            <div className="flex-1">
              <h1 className="mb-1 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                Configure Actions
              </h1>
              <p className="text-muted-foreground">Step 4 of 4: Define what your agent does</p>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="bg-slate-200 rounded-full h-2 mb-6">
            <motion.div 
              initial={{ width: "75%" }}
              animate={{ width: "100%" }}
              className="bg-gradient-to-r from-indigo-500 to-purple-600 h-2 rounded-full"
              transition={{ duration: 0.5 }}
            />
          </div>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Actions List */}
          <div className="lg:col-span-2 space-y-6">
            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <Card className="bg-white/70 backdrop-blur-sm border-white/50">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2">
                      <Zap className="h-5 w-5" />
                      Agent Actions ({actions.length})
                    </CardTitle>
                    <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
                      <DialogTrigger asChild>
                        <Button onClick={handleAddAction} className="bg-gradient-to-r from-indigo-500 to-purple-600">
                          <Plus className="mr-2 h-4 w-4" />
                          Add Action
                        </Button>
                      </DialogTrigger>
                      <DialogContent className="max-w-2xl">
                        <DialogHeader>
                          <DialogTitle>
                            {editingAction ? "Edit Action" : "Add New Action"}
                          </DialogTitle>
                        </DialogHeader>
                        <div className="space-y-6 py-4">
                          <div className="grid grid-cols-2 gap-4">
                            <div className="space-y-2">
                              <Label htmlFor="action-name">Action Name *</Label>
                              <Input
                                id="action-name"
                                placeholder="e.g., Generate Report"
                                value={actionForm.name}
                                onChange={(e) => setActionForm(prev => ({ ...prev, name: e.target.value }))}
                              />
                            </div>
                            <div className="space-y-2">
                              <Label htmlFor="trigger-type">Trigger Type *</Label>
                              <Select 
                                value={actionForm.triggerType} 
                                onValueChange={(value) => setActionForm(prev => ({ ...prev, triggerType: value }))}
                              >
                                <SelectTrigger>
                                  <SelectValue placeholder="Select trigger type" />
                                </SelectTrigger>
                                <SelectContent>
                                  {triggerTypes.map((type) => (
                                    <SelectItem key={type.value} value={type.value}>
                                      {type.label}
                                    </SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                            </div>
                          </div>

                          <div className="space-y-2">
                            <Label htmlFor="action-description">Description</Label>
                            <Textarea
                              id="action-description"
                              placeholder="Describe what this action does..."
                              value={actionForm.description}
                              onChange={(e) => setActionForm(prev => ({ ...prev, description: e.target.value }))}
                              rows={2}
                            />
                          </div>

                          <div className="space-y-2">
                            <Label htmlFor="trigger-value">Trigger Value *</Label>
                            <Input
                              id="trigger-value"
                              placeholder={
                                actionForm.triggerType === "keyword" ? "Keywords to watch for..." :
                                actionForm.triggerType === "schedule" ? "0 9 * * MON (every Monday at 9am)" :
                                actionForm.triggerType === "webhook" ? "webhook-endpoint-url" :
                                "Trigger condition..."
                              }
                              value={actionForm.trigger}
                              onChange={(e) => setActionForm(prev => ({ ...prev, trigger: e.target.value }))}
                            />
                          </div>

                          <div className="space-y-2">
                            <Label htmlFor="response">Agent Response *</Label>
                            <Select 
                              value={actionForm.response} 
                              onValueChange={(value) => {
                                if (value === "Custom response...") {
                                  setActionForm(prev => ({ ...prev, response: "" }));
                                } else {
                                  setActionForm(prev => ({ ...prev, response: value }));
                                }
                              }}
                            >
                              <SelectTrigger>
                                <SelectValue placeholder="Select or type custom response" />
                              </SelectTrigger>
                              <SelectContent>
                                {responseTemplates.map((template) => (
                                  <SelectItem key={template} value={template}>
                                    {template}
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                            {!responseTemplates.includes(actionForm.response) && (
                              <Textarea
                                placeholder="Custom response instructions..."
                                value={actionForm.response}
                                onChange={(e) => setActionForm(prev => ({ ...prev, response: e.target.value }))}
                                rows={3}
                              />
                            )}
                          </div>

                          <div className="flex justify-end gap-2">
                            <Button variant="outline" onClick={() => setIsDialogOpen(false)}>
                              Cancel
                            </Button>
                            <Button onClick={handleSaveAction}>
                              <Save className="mr-2 h-4 w-4" />
                              {editingAction ? "Update" : "Add"} Action
                            </Button>
                          </div>
                        </div>
                      </DialogContent>
                    </Dialog>
                  </div>
                </CardHeader>
                <CardContent>
                  {actions.length === 0 ? (
                    <div className="text-center py-12">
                      <Zap className="mx-auto h-16 w-16 text-muted-foreground mb-4" />
                      <h3 className="mb-2">No actions configured</h3>
                      <p className="text-muted-foreground mb-6">
                        Add actions to define what your agent does when triggered
                      </p>
                      <Button onClick={handleAddAction} className="bg-gradient-to-r from-indigo-500 to-purple-600">
                        <Plus className="mr-2 h-4 w-4" />
                        Add Your First Action
                      </Button>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      {actions.map((action, index) => {
                        const TriggerIcon = getTriggerIcon(action.triggerType);
                        
                        return (
                          <motion.div
                            key={action.id}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.1 * index }}
                            className="border rounded-lg p-4 hover:shadow-md transition-all"
                          >
                            <div className="flex items-start justify-between">
                              <div className="flex-1 space-y-2">
                                <div className="flex items-center gap-2">
                                  <TriggerIcon className="h-4 w-4 text-indigo-600" />
                                  <h4>{action.name}</h4>
                                  <Badge variant="outline" className="text-xs">
                                    {triggerTypes.find(t => t.value === action.triggerType)?.label || action.triggerType}
                                  </Badge>
                                </div>
                                {action.description && (
                                  <p className="text-sm text-muted-foreground">{action.description}</p>
                                )}
                                <div className="text-sm">
                                  <span className="text-muted-foreground">Trigger: </span>
                                  <code className="bg-slate-100 px-2 py-1 rounded text-xs">{action.trigger}</code>
                                </div>
                                <div className="text-sm">
                                  <span className="text-muted-foreground">Response: </span>
                                  <span className="text-slate-700">{action.response}</span>
                                </div>
                              </div>
                              <div className="flex gap-2">
                                <Button 
                                  variant="outline" 
                                  size="sm"
                                  onClick={() => handleEditAction(action)}
                                >
                                  <Edit className="h-3 w-3" />
                                </Button>
                                <Button 
                                  variant="outline" 
                                  size="sm"
                                  onClick={() => handleDeleteAction(action.id)}
                                  className="text-red-500 hover:text-red-700"
                                >
                                  <Trash2 className="h-3 w-3" />
                                </Button>
                              </div>
                            </div>
                          </motion.div>
                        );
                      })}
                    </div>
                  )}
                </CardContent>
              </Card>
            </motion.div>
          </div>

          {/* Summary & Finish */}
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
            className="space-y-6"
          >
            <Card className="bg-white/70 backdrop-blur-sm border-white/50 sticky top-6">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle className="h-5 w-5" />
                  Agent Summary
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="text-center">
                  <div className="text-2xl mb-2">{agentConfig.avatar}</div>
                  <h3 className="mb-1">{agentConfig.name}</h3>
                  <p className="text-sm text-muted-foreground">{agentConfig.description}</p>
                </div>

                <div className="space-y-3 pt-4 border-t">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Model:</span>
                    <span>{agentConfig.model?.name || "Not selected"}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Tools:</span>
                    <span>{agentConfig.tools.length}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Actions:</span>
                    <span>{actions.length}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Visibility:</span>
                    <span>{agentConfig.isPublic ? "Public" : "Private"}</span>
                  </div>
                </div>

                {agentConfig.tags.length > 0 && (
                  <div className="pt-3 border-t">
                    <div className="text-sm text-muted-foreground mb-2">Tags:</div>
                    <div className="flex flex-wrap gap-1">
                      {agentConfig.tags.map((tag) => (
                        <Badge key={tag} variant="secondary" className="text-xs">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
              <Button 
                onClick={handleFinish}
                className="w-full bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700"
                size="lg"
              >
                <CheckCircle className="mr-2 h-4 w-4" />
                Create Agent
              </Button>
            </motion.div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}