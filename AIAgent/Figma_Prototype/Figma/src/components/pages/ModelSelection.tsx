import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { RadioGroup, RadioGroupItem } from "../ui/radio-group";
import { Label } from "../ui/label";
import { Slider } from "../ui/slider";
import { Page, AgentConfig } from "../../App";
import { 
  ArrowLeft, 
  ArrowRight, 
  Brain, 
  Zap, 
  DollarSign, 
  Clock,
  CheckCircle,
  Info,
  Sparkles,
  Target,
  Gauge
} from "lucide-react";
import { motion } from "motion/react";

interface ModelSelectionProps {
  navigateTo: (page: Page) => void;
  agentConfig: AgentConfig;
  updateAgentConfig: (updates: Partial<AgentConfig>) => void;
}

const models = [
  {
    id: "gpt-4",
    provider: "OpenAI",
    name: "GPT-4",
    description: "Most capable model, excellent for complex reasoning and analysis",
    pricing: "$$$",
    speed: "Medium",
    capabilities: ["Reasoning", "Analysis", "Coding", "Writing"],
    contextWindow: "128K tokens",
    recommended: true,
    specs: {
      reasoning: 95,
      creativity: 90,
      speed: 70,
      cost: 30
    }
  },
  {
    id: "gpt-3.5-turbo",
    provider: "OpenAI", 
    name: "GPT-3.5 Turbo",
    description: "Fast and cost-effective, great for general purpose tasks",
    pricing: "$",
    speed: "Fast",
    capabilities: ["General", "Coding", "Writing", "Support"],
    contextWindow: "16K tokens",
    recommended: false,
    specs: {
      reasoning: 75,
      creativity: 80,
      speed: 95,
      cost: 90
    }
  },
  {
    id: "claude-3-opus",
    provider: "Anthropic",
    name: "Claude-3 Opus",
    description: "Exceptional performance on complex tasks, strong reasoning",
    pricing: "$$$",
    speed: "Medium",
    capabilities: ["Reasoning", "Analysis", "Research", "Writing"],
    contextWindow: "200K tokens",
    recommended: false,
    specs: {
      reasoning: 98,
      creativity: 85,
      speed: 65,
      cost: 25
    }
  },
  {
    id: "claude-3-sonnet",
    provider: "Anthropic",
    name: "Claude-3 Sonnet",
    description: "Balanced performance and speed, ideal for most applications",
    pricing: "$$",
    speed: "Fast",
    capabilities: ["General", "Analysis", "Writing", "Support"],
    contextWindow: "200K tokens",
    recommended: true,
    specs: {
      reasoning: 85,
      creativity: 88,
      speed: 85,
      cost: 70
    }
  },
  {
    id: "gemini-pro",
    provider: "Google",
    name: "Gemini Pro", 
    description: "Strong multimodal capabilities, good for diverse tasks",
    pricing: "$$",
    speed: "Fast",
    capabilities: ["Multimodal", "Analysis", "Coding", "General"],
    contextWindow: "32K tokens",
    recommended: false,
    specs: {
      reasoning: 80,
      creativity: 82,
      speed: 88,
      cost: 75
    }
  },
  {
    id: "llama-2",
    provider: "Meta",
    name: "Llama 2",
    description: "Open source model, customizable and cost-effective",
    pricing: "Free",
    speed: "Medium",
    capabilities: ["General", "Coding", "Open Source"],
    contextWindow: "4K tokens",
    recommended: false,
    specs: {
      reasoning: 70,
      creativity: 75,
      speed: 75,
      cost: 100
    }
  }
];

export function ModelSelection({ navigateTo, agentConfig, updateAgentConfig }: ModelSelectionProps) {
  const [selectedModelId, setSelectedModelId] = useState(agentConfig.model?.name || "");
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(2048);

  const selectedModel = models.find(m => m.id === selectedModelId);

  const handleNext = () => {
    if (selectedModel) {
      updateAgentConfig({
        model: {
          provider: selectedModel.provider,
          name: selectedModel.name,
          config: {
            temperature,
            maxTokens,
            modelId: selectedModel.id
          }
        }
      });
      navigateTo("tools-selection");
    }
  };

  const SpecBar = ({ label, value, color }: { label: string; value: number; color: string }) => (
    <div className="space-y-1">
      <div className="flex justify-between text-sm">
        <span className="text-muted-foreground">{label}</span>
        <span>{value}%</span>
      </div>
      <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value}%` }}
          className={`h-full ${color}`}
          transition={{ duration: 0.5, delay: 0.2 }}
        />
      </div>
    </div>
  );

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
            <Button variant="outline" onClick={() => navigateTo("agent-builder")}>
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back
            </Button>
            <div className="flex-1">
              <h1 className="mb-1 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                Select AI Model
              </h1>
              <p className="text-muted-foreground">Step 2 of 4: Choose the brain for your agent</p>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="bg-slate-200 rounded-full h-2 mb-6">
            <motion.div 
              initial={{ width: "25%" }}
              animate={{ width: "50%" }}
              className="bg-gradient-to-r from-indigo-500 to-purple-600 h-2 rounded-full"
              transition={{ duration: 0.5 }}
            />
          </div>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Model Selection */}
          <div className="lg:col-span-2 space-y-6">
            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <Card className="bg-white/70 backdrop-blur-sm border-white/50">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Brain className="h-5 w-5" />
                    Available Models
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <RadioGroup value={selectedModelId} onValueChange={setSelectedModelId}>
                    <div className="space-y-4">
                      {models.map((model, index) => (
                        <motion.div
                          key={model.id}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.1 * index }}
                          className={`border rounded-lg p-4 cursor-pointer transition-all hover:shadow-md ${
                            selectedModelId === model.id
                              ? "border-indigo-500 bg-indigo-50"
                              : "border-slate-200 hover:border-slate-300"
                          }`}
                          onClick={() => setSelectedModelId(model.id)}
                        >
                          <div className="flex items-start space-x-3">
                            <RadioGroupItem value={model.id} className="mt-1" />
                            <div className="flex-1 space-y-2">
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                  <Label className="cursor-pointer">{model.name}</Label>
                                  <Badge variant="outline" className="text-xs">
                                    {model.provider}
                                  </Badge>
                                  {model.recommended && (
                                    <Badge className="bg-gradient-to-r from-indigo-500 to-purple-600 text-xs">
                                      <Sparkles className="mr-1 h-3 w-3" />
                                      Recommended
                                    </Badge>
                                  )}
                                </div>
                                <div className="flex items-center gap-4 text-sm text-muted-foreground">
                                  <div className="flex items-center gap-1">
                                    <DollarSign className="h-3 w-3" />
                                    {model.pricing}
                                  </div>
                                  <div className="flex items-center gap-1">
                                    <Clock className="h-3 w-3" />
                                    {model.speed}
                                  </div>
                                </div>
                              </div>
                              <p className="text-sm text-muted-foreground">
                                {model.description}
                              </p>
                              <div className="flex flex-wrap gap-1">
                                {model.capabilities.map((cap) => (
                                  <Badge key={cap} variant="secondary" className="text-xs">
                                    {cap}
                                  </Badge>
                                ))}
                              </div>
                              <div className="text-xs text-muted-foreground">
                                Context: {model.contextWindow}
                              </div>
                            </div>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </RadioGroup>
                </CardContent>
              </Card>
            </motion.div>

            {/* Configuration */}
            {selectedModel && (
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
              >
                <Card className="bg-white/70 backdrop-blur-sm border-white/50">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Target className="h-5 w-5" />
                      Model Configuration
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <Label>Temperature</Label>
                        <span className="text-sm text-muted-foreground">{temperature}</span>
                      </div>
                      <Slider
                        value={[temperature]}
                        onValueChange={([value]) => setTemperature(value)}
                        max={1}
                        min={0}
                        step={0.1}
                        className="w-full"
                      />
                      <p className="text-sm text-muted-foreground">
                        Higher values make output more random, lower values more focused
                      </p>
                    </div>

                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <Label>Max Output Tokens</Label>
                        <span className="text-sm text-muted-foreground">{maxTokens}</span>
                      </div>
                      <Slider
                        value={[maxTokens]}
                        onValueChange={([value]) => setMaxTokens(value)}
                        max={4096}
                        min={256}
                        step={128}
                        className="w-full"
                      />
                      <p className="text-sm text-muted-foreground">
                        Maximum number of tokens the model can generate
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )}
          </div>

          {/* Preview/Stats */}
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.6 }}
            className="space-y-6"
          >
            {selectedModel ? (
              <Card className="bg-white/70 backdrop-blur-sm border-white/50 sticky top-6">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Gauge className="h-5 w-5" />
                    Model Performance
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="text-center">
                    <h3 className="mb-1">{selectedModel.name}</h3>
                    <p className="text-sm text-muted-foreground">{selectedModel.provider}</p>
                  </div>

                  <div className="space-y-3">
                    <SpecBar label="Reasoning" value={selectedModel.specs.reasoning} color="bg-blue-500" />
                    <SpecBar label="Creativity" value={selectedModel.specs.creativity} color="bg-purple-500" />
                    <SpecBar label="Speed" value={selectedModel.specs.speed} color="bg-green-500" />
                    <SpecBar label="Cost Efficiency" value={selectedModel.specs.cost} color="bg-orange-500" />
                  </div>

                  <div className="pt-4 border-t">
                    <div className="text-sm space-y-2">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Context Window:</span>
                        <span>{selectedModel.contextWindow}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Pricing:</span>
                        <span>{selectedModel.pricing}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Speed:</span>
                        <span>{selectedModel.speed}</span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ) : (
              <Card className="bg-white/70 backdrop-blur-sm border-white/50 sticky top-6">
                <CardContent className="p-6 text-center">
                  <Brain className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
                  <h3 className="mb-2">Select a Model</h3>
                  <p className="text-sm text-muted-foreground">
                    Choose an AI model to see its performance characteristics
                  </p>
                </CardContent>
              </Card>
            )}

            <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
              <Button 
                onClick={handleNext}
                disabled={!selectedModel}
                className="w-full bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700"
                size="lg"
              >
                Next: Select Tools
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </motion.div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}