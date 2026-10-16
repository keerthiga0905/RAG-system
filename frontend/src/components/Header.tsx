import React from 'react';
import { MessageSquare, Clock, BarChart3, Database, Cpu, User, LogOut, ArrowRightLeft } from 'lucide-react';
import type { UserProfile } from '../types';

interface HeaderProps {
  activeTab: 'chat' | 'history' | 'evaluation' | 'documents' | 'model';
  setActiveTab: (tab: 'chat' | 'history' | 'evaluation' | 'documents' | 'model') => void;
  systemHealthy: boolean;
  totalChunks: number;
  currentUser: UserProfile | null;
  onOpenLogin: () => void;
  onLogout: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  systemHealthy,
  totalChunks,
  currentUser,
  onOpenLogin,
  onLogout
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
            className={`nav-btn ${activeTab === 'history' ? 'active' : ''}`}
            onClick={() => setActiveTab('history')}
            id="nav-history-btn"
          >
            <Clock size={16} />
            <span>History</span>
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

        <div className="header-right-group">
          {currentUser ? (
            <div className="header-user-chip">
              <div 
                className="user-avatar-small" 
                style={{ backgroundColor: currentUser.avatarColor || '#800000' }}
                title={currentUser.email}
              >
                {currentUser.name.charAt(0)}
              </div>
              <div className="user-details-mini">
                <span className="user-name-mini">{currentUser.name}</span>
                <span className="user-email-mini">{currentUser.email}</span>
              </div>
              <button 
                className="user-switch-btn"
                onClick={onOpenLogin}
                title="Switch User / Login Page"
              >
                <ArrowRightLeft size={13} />
                <span>Switch</span>
              </button>
              <button 
                className="user-switch-btn"
                onClick={onLogout}
                title="Sign Out"
                style={{ background: 'rgba(239, 68, 68, 0.25)' }}
              >
                <LogOut size={13} />
                <span>Logout</span>
              </button>
            </div>
          ) : (
            <button className="nav-btn login-trigger-btn" onClick={onOpenLogin}>
              <User size={15} />
              <span>Sign In</span>
            </button>
          )}

          <div className="header-status">
            <span className="status-dot"></span>
            <span>{systemHealthy ? `Vector DB Online (${totalChunks})` : 'Offline'}</span>
          </div>
        </div>
      </div>
    </header>
  );
};
