import React from 'react';
import { MessageSquare, BarChart3, Database, Cpu } from 'lucide-react';

interface HeaderProps {
  activeTab: 'chat' | 'evaluation' | 'documents' | 'model';
  setActiveTab: (tab: 'chat' | 'evaluation' | 'documents' | 'model') => void;
  systemHealthy: boolean;
  totalChunks: number;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  systemHealthy,
  totalChunks
}) => {
  return (
    <header className="app-header">
      <div className="header-inner">
        <div className="header-brand">
          <img src="/cit_logo.svg" alt="CIT Logo" className="header-logo" />
          <div>
            <div className="brand-title">
              CIT Knowledge AI
              <span className="brand-badge">RAG ML LAB</span>
            </div>
            <div className="brand-subtitle">
              Coimbatore Institute of Technology &bull; Autonomous Institution Grounded AI
            </div>
          </div>
        </div>

        <nav className="header-nav">
          <button
            className={`nav-btn ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab('chat')}
            id="nav-chat-btn"
          >
            <MessageSquare size={16} />
            <span>Assistant</span>
          </button>
          <button
            className={`nav-btn ${activeTab === 'evaluation' ? 'active' : ''}`}
            onClick={() => setActiveTab('evaluation')}
            id="nav-evaluation-btn"
          >
            <BarChart3 size={16} />
            <span>Evaluation</span>
          </button>
          <button
            className={`nav-btn ${activeTab === 'documents' ? 'active' : ''}`}
            onClick={() => setActiveTab('documents')}
            id="nav-documents-btn"
          >
            <Database size={16} />
            <span>Datasets</span>
          </button>
          <button
            className={`nav-btn ${activeTab === 'model' ? 'active' : ''}`}
            onClick={() => setActiveTab('model')}
            id="nav-model-btn"
          >
            <Cpu size={16} />
            <span>Model Info</span>
          </button>
        </nav>

        <div className="header-status">
          <span className="status-dot"></span>
          <span>{systemHealthy ? `Local Vector DB Online (${totalChunks} Chunks)` : 'Offline'}</span>
        </div>
      </div>
    </header>
  );
};
