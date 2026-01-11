import { useState } from "react";
import { Homepage } from "./components/pages/Homepage";
import { AgentBuilder } from "./components/pages/AgentBuilder";
import { ModelSelection } from "./components/pages/ModelSelection";
import { ToolsSelection } from "./components/pages/ToolsSelection";
import { ActionsConfiguration } from "./components/pages/ActionsConfiguration";
import { URLExtractor } from "./components/pages/URLExtractor";
import { AgentCreator } from "./components/pages/AgentCreator";
import { AgentManagement } from "./components/pages/AgentManagement";

export type Page = "homepage" | "agent-builder" | "model-selection" | "tools-selection" | "actions-configuration" | "url-extractor" | "agent-creator" | "agent-management";

export interface AgentConfig {
  id?: string;
  name: string;
  description: string;
  model: {
    provider: string;
    name: string;
    config: any;
  } | null;
  tools: Array<{
    id: string;
    name: string;
    description: string;
    type: string;
    config: any;
  }>;
  actions: Array<{
    id: string;
    name: string;
    description: string;
    trigger: string;
    response: string;
  }>;
  avatar: string;
  tags: string[];
  isPublic: boolean;
  createdAt: Date;
}

export default function App() {
  const [currentPage, setCurrentPage] = useState<Page>("homepage");
  const [agentConfig, setAgentConfig] = useState<AgentConfig>({
    name: "",
    description: "",
    model: null,
    tools: [],
    actions: [],
    avatar: "",
    tags: [],
    isPublic: false,
    createdAt: new Date()
  });

  const navigateTo = (page: Page) => {
    setCurrentPage(page);
  };

  const updateAgentConfig = (updates: Partial<AgentConfig>) => {
    setAgentConfig(prev => ({ ...prev, ...updates }));
  };

  const resetAgentConfig = () => {
    setAgentConfig({
      name: "",
      description: "",
      model: null,
      tools: [],
      actions: [],
      avatar: "",
      tags: [],
      isPublic: false,
      createdAt: new Date()
    });
  };

  const renderPage = () => {
    switch (currentPage) {
      case "homepage":
        return <Homepage navigateTo={navigateTo} resetAgentConfig={resetAgentConfig} />;
      case "url-extractor":
        return <URLExtractor navigateTo={navigateTo} />;
      case "agent-creator":
        return <AgentCreator navigateTo={navigateTo} />;
      case "agent-management":
        return <AgentManagement navigateTo={navigateTo} />;
      case "agent-builder":
        return <AgentBuilder 
          navigateTo={navigateTo} 
          agentConfig={agentConfig} 
          updateAgentConfig={updateAgentConfig} 
        />;
      case "model-selection":
        return <ModelSelection 
          navigateTo={navigateTo} 
          agentConfig={agentConfig} 
          updateAgentConfig={updateAgentConfig} 
        />;
      case "tools-selection":
        return <ToolsSelection 
          navigateTo={navigateTo} 
          agentConfig={agentConfig} 
          updateAgentConfig={updateAgentConfig} 
        />;
      case "actions-configuration":
        return <ActionsConfiguration 
          navigateTo={navigateTo} 
          agentConfig={agentConfig} 
          updateAgentConfig={updateAgentConfig} 
        />;
      default:
        return <Homepage navigateTo={navigateTo} resetAgentConfig={resetAgentConfig} />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {renderPage()}
    </div>
  );
}