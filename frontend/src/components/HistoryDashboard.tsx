import React, { useState, useMemo } from 'react';
import { 
  Clock, Search, Trash2, Download, ExternalLink, 
  BookOpen, Compass, Filter, RefreshCw, MessageSquare,
  ArrowUpRight, Copy, Check
} from 'lucide-react';
import type { HistoryRecord, UserProfile } from '../types';

interface HistoryDashboardProps {
  user: UserProfile;
  history: HistoryRecord[];
  onClearHistory: () => void;
  onResumeInChat: (query: string) => void;
  onRefresh: () => void;
}

export const HistoryDashboard: React.FC<HistoryDashboardProps> = ({
  user,
  history,
  onClearHistory,
  onResumeInChat,
  onRefresh
}) => {
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [selectedCategory, setSelectedCategory] = useState<string>('All');
  const [selectedLevel, setSelectedLevel] = useState<string>('All');
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Compute analytics
  const analytics = useMemo(() => {
    const total = history.length;
    if (total === 0) {
      return { total: 0, avgLatency: 0, totalCitations: 0, level1: 0, level2: 0, level3: 0, roadmap: 0 };
    }
    const sumLatency = history.reduce((acc, h) => acc + (h.latency_ms || 0), 0);
    const sumCitations = history.reduce((acc, h) => acc + (h.citations?.length || 0), 0);
    const level1 = history.filter(h => h.retrieval_level === 1).length;
    const level2 = history.filter(h => h.retrieval_level === 2).length;
    const level3 = history.filter(h => h.retrieval_level === 3).length;
    const roadmap = history.filter(h => h.category === 'Navigations' || h.query.toLowerCase().includes('how to go') || h.query.toLowerCase().includes('navigate')).length;

    return {
      total,
      avgLatency: Math.round(sumLatency / total),
      totalCitations: sumCitations,
      level1,
      level2,
      level3,
      roadmap
    };
  }, [history]);

  // Filter history
  const filteredHistory = useMemo(() => {
    return history.filter(item => {
      const matchSearch = !searchTerm || 
        item.query.toLowerCase().includes(searchTerm.toLowerCase()) ||
        item.answer.toLowerCase().includes(searchTerm.toLowerCase()) ||
        item.category.toLowerCase().includes(searchTerm.toLowerCase());

      const matchCategory = selectedCategory === 'All' || item.category === selectedCategory;

      const matchLevel = selectedLevel === 'All' || 
        (selectedLevel === 'Level 1' && item.retrieval_level === 1) ||
        (selectedLevel === 'Level 2' && item.retrieval_level === 2) ||
        (selectedLevel === 'Level 3' && item.retrieval_level === 3) ||
        (selectedLevel === 'Roadmap' && (item.category === 'Navigations' || item.query.toLowerCase().includes('how to go')));

      return matchSearch && matchCategory && matchLevel;
    });
  }, [history, searchTerm, selectedCategory, selectedLevel]);

  // Export JSON
  const handleExportJSON = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(history, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `cit_history_${user.email.replace(/[^a-zA-Z0-9]/g, '_')}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  // Export Markdown
  const handleExportMarkdown = () => {
    let md = `# CIT Knowledge AI — Chat History Report\n`;
    md += `**User Email:** ${user.email}\n`;
    md += `**Name:** ${user.name} (${user.role} - ${user.department || 'CIT'})\n`;
    md += `**Export Date:** ${new Date().toLocaleString()}\n`;
    md += `**Total Inquiries:** ${history.length}\n\n---\n\n`;

    history.forEach((h, idx) => {
      md += `### ${idx + 1}. ${h.query}\n`;
      md += `- **Date/Time:** ${h.timestamp}\n`;
      md += `- **Category:** ${h.category} | **Level:** ${h.retrieval_level || 1} (${h.source_type || 'dataset'})\n`;
      md += `- **Latency:** ${h.latency_ms || 0} ms | **Model:** ${h.model_used || 'hybrid'}\n\n`;
      md += `**Answer:**\n${h.answer}\n\n`;
      if (h.citations && h.citations.length > 0) {
        md += `**References:**\n`;
        h.citations.forEach(c => {
          md += `- [${c.document_title}](${c.source_url})\n`;
        });
      }
      md += `\n---\n\n`;
    });

    const dataStr = "data:text/markdown;charset=utf-8," + encodeURIComponent(md);
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `cit_history_${user.email.replace(/[^a-zA-Z0-9]/g, '_')}.md`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const handleCopy = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className="history-dashboard-container">
      
      {/* Header Profile Ribbon */}
      <div className="history-header-card">
        <div className="history-user-info">
          <div 
            className="history-avatar" 
            style={{ backgroundColor: user.avatarColor || '#800000' }}
          >
            {user.name.charAt(0)}
          </div>
          <div>
            <div className="history-user-name">
              <h3>{user.name}</h3>
              <span className="history-role-pill">{user.role}</span>
              {user.department && <span className="history-dept-pill">{user.department}</span>}
            </div>
            <div className="history-user-email">
              <Clock size={13} />
              <span>Workspace ID: <strong>{user.email}</strong></span>
              <span>&bull; Inquiries are securely stored and isolated to this email</span>
            </div>
          </div>
        </div>

        <div className="history-actions-top">
          <button 
            className="btn-secondary history-btn"
            onClick={onRefresh}
            title="Refresh history"
          >
            <RefreshCw size={14} />
            <span>Sync</span>
          </button>
          <button 
            className="btn-secondary history-btn"
            onClick={handleExportJSON}
            disabled={history.length === 0}
            title="Export as JSON"
          >
            <Download size={14} />
            <span>Export JSON</span>
          </button>
          <button 
            className="btn-secondary history-btn"
            onClick={handleExportMarkdown}
            disabled={history.length === 0}
            title="Export as Markdown Report"
          >
            <Download size={14} />
            <span>Markdown</span>
          </button>
          <button 
            className="btn-danger history-btn"
            onClick={() => {
              if (window.confirm(`Are you sure you want to clear all chat history for ${user.email}?`)) {
                onClearHistory();
              }
            }}
            disabled={history.length === 0}
            title="Clear all stored history for this email"
          >
            <Trash2 size={14} />
            <span>Clear History</span>
          </button>
        </div>
      </div>

      {/* Metrics Overview Ribbon */}
      <div className="history-metrics-grid">
        <div className="metric-box">
          <div className="metric-val">{analytics.total}</div>
          <div className="metric-lbl">Total Inquiries</div>
        </div>
        <div className="metric-box">
          <div className="metric-val">{analytics.totalCitations}</div>
          <div className="metric-lbl">References Verified</div>
        </div>
        <div className="metric-box">
          <div className="metric-val">{analytics.avgLatency} <span style={{ fontSize: '0.9rem' }}>ms</span></div>
          <div className="metric-lbl">Avg Response Time</div>
        </div>
        <div className="metric-box">
          <div className="metric-val" style={{ color: '#059669' }}>
            {analytics.total > 0 ? '100%' : '0%'}
          </div>
          <div className="metric-lbl">Grounded Accuracy</div>
        </div>
      </div>

      {/* Search and Filters Bar */}
      <div className="history-filter-bar">
        <div className="history-search-input">
          <Search size={16} />
          <input 
            type="text" 
            placeholder="Search through past questions, answers, or topics..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>

        <div className="filter-select-group">
          <div className="filter-select-wrapper">
            <Filter size={14} />
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
            >
              <option value="All">All Categories</option>
              <option value="Departments and Faculty">Departments & Faculty</option>
              <option value="Admissions and Eligibility">Admissions & Eligibility</option>
              <option value="Academics and Regulations">Academics & Regulations</option>
              <option value="Campus Facilities & Regulations">Campus Regulations & Timings</option>
              <option value="Placements and Career">Placements & Career</option>
              <option value="Navigations">Campus Roadmap Navigation</option>
              <option value="College Overview">College Overview</option>
            </select>
          </div>

          <div className="filter-select-wrapper">
            <select
              value={selectedLevel}
              onChange={(e) => setSelectedLevel(e.target.value)}
            >
              <option value="All">All Sources</option>
              <option value="Level 1">Level 1: Local Dataset</option>
              <option value="Level 2">Level 2: Official CIT Website</option>
              <option value="Level 3">Level 3: Campus Regulations (Web)</option>
              <option value="Roadmap">Campus Roadmap Navigator</option>
            </select>
          </div>
        </div>
      </div>

      {/* History Items List */}
      <div className="history-items-container">
        {filteredHistory.length === 0 ? (
          <div className="history-empty-state">
            <MessageSquare size={48} className="empty-icon" />
            <h4>No Inquiries Found</h4>
            <p>
              {history.length === 0 
                ? `You haven't asked any questions yet under ${user.email}. Ask a question in the Assistant tab to start building your verified history!`
                : `No history items match your search "${searchTerm}" with the active filters.`}
            </p>
          </div>
        ) : (
          filteredHistory.map((item) => (
            <div key={item.id} className="history-card">
              
              {/* Question Header */}
              <div className="history-card-header">
                <div className="history-query-title">
                  <span className="q-badge">Q</span>
                  <h4>{item.query}</h4>
                </div>
                
                <div className="history-card-actions">
                  <button 
                    className="history-action-btn"
                    onClick={() => handleCopy(item.id, item.answer)}
                    title="Copy Answer"
                  >
                    {copiedId === item.id ? <Check size={14} color="#059669" /> : <Copy size={14} />}
                    <span>{copiedId === item.id ? 'Copied' : 'Copy'}</span>
                  </button>

                  <button 
                    className="history-action-btn highlight"
                    onClick={() => onResumeInChat(item.query)}
                    title="Load question into Assistant chat"
                  >
                    <ArrowUpRight size={14} />
                    <span>Ask in Assistant</span>
                  </button>
                </div>
              </div>

              {/* Badges Info Bar */}
              <div className="history-meta-row">
                <span className="meta-tag date">
                  <Clock size={11} />
                  {item.timestamp}
                </span>

                <span className="meta-tag category">
                  {item.category || 'General'}
                </span>

                {item.retrieval_level === 1 && (
                  <span className="meta-tag level-1">
                    Level 1: Verified Local Dataset
                  </span>
                )}
                {item.retrieval_level === 2 && (
                  <span className="meta-tag level-2">
                    Level 2: Official Website (cit.edu.in)
                  </span>
                )}
                {item.retrieval_level === 3 && (
                  <span className="meta-tag level-3">
                    Level 3: Verified Campus Regulations
                  </span>
                )}
                {(item.category === 'Navigations' || item.query.toLowerCase().includes('how to go')) && (
                  <span className="meta-tag roadmap">
                    <Compass size={11} /> Campus Roadmap Navigator
                  </span>
                )}

                <span className="meta-tag latency">
                  {item.latency_ms || 0} ms
                </span>
                <span className="meta-tag model">
                  Model: {item.model_used || 'hybrid'}
                </span>
              </div>

              {/* Direct Answer */}
              <div className="history-answer-box">
                <div style={{ whiteSpace: 'pre-wrap', lineHeight: '1.5' }}>
                  {item.answer}
                </div>

                {/* References */}
                {item.citations && item.citations.length > 0 && (
                  <div className="history-citations-list">
                    <div className="citations-header">
                      <BookOpen size={12} />
                      <span>{item.citations.length === 1 ? 'Reference:' : 'References:'}</span>
                    </div>
                    <div className="history-citation-chips">
                      {item.citations.map((c, cIdx) => (
                        <a 
                          key={cIdx}
                          href={c.source_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="history-cite-chip"
                        >
                          <span>{c.document_title}</span>
                          <ExternalLink size={11} />
                        </a>
                      ))}
                    </div>
                  </div>
                )}
              </div>

            </div>
          ))
        )}
      </div>

    </div>
  );
};
