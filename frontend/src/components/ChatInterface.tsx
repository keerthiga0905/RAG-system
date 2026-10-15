import React, { useState, useRef, useEffect } from 'react';
import { 
  Send, Sparkles, ExternalLink, AlertTriangle, 
  Layers, ThumbsUp, ThumbsDown, BookOpen, Clock, ChevronRight, X,
  CheckCircle2, Globe, Search
} from 'lucide-react';
import type { ChatMessage, RetrievedEvidence } from '../types';

interface ChatInterfaceProps {
  messages: ChatMessage[];
  onSendMessage: (query: string, modelChoice: string, categoryFilter?: string) => Promise<void>;
  isLoading: boolean;
  selectedEvidence: RetrievedEvidence[];
  setSelectedEvidence: (evidence: RetrievedEvidence[]) => void;
}

const SUGGESTED_QUESTIONS = [
  {
    category: "Departments & Faculty",
    q: "Who is the HOD of M.Sc. Software Systems?"
  },
  {
    category: "Campus Rules & Timings",
    q: "What is the gate closing time at CIT?"
  },
  {
    category: "Admissions & Fees",
    q: "What is the tuition fee for B.E. Government Aided first year for 2026-2027?"
  },
  {
    category: "Placements & Internships",
    q: "Tell me about placements and recruiting companies at CIT."
  },
  {
    category: "Courses & Eligibility",
    q: "What is the eligibility for M.Sc. Software Systems?"
  },
  {
    category: "Hostels & Campus Rules",
    q: "What are the hostel facilities and curfew rules at CIT?"
  }
];

const CATEGORIES = [
  "All Categories",
  "College Overview",
  "Departments and Faculty",
  "Courses and Academic Programmes",
  "Admissions and Eligibility",
  "Academics and Regulations",
  "Placements and Career Development",
  "Scholarships, Fees and Student Support",
  "Research and Innovation",
  "Campus Facilities and FAQs"
];

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  messages,
  onSendMessage,
  isLoading,
  selectedEvidence,
  setSelectedEvidence
}) => {
  const [input, setInput] = useState('');
  const [modelChoice, setModelChoice] = useState<'hybrid' | 'finetuned' | 'baseline'>('hybrid');
  const [selectedCategory, setSelectedCategory] = useState('All Categories');
  const [showDebugSidebar, setShowDebugSidebar] = useState<boolean>(false);
  const [feedbackSent, setFeedbackSent] = useState<Record<string, boolean>>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    const cat = selectedCategory === 'All Categories' ? undefined : selectedCategory;
    onSendMessage(input.trim(), modelChoice, cat);
    setInput('');
  };

  const handleSelectPrompt = (q: string) => {
    if (isLoading) return;
    const cat = selectedCategory === 'All Categories' ? undefined : selectedCategory;
    onSendMessage(q, modelChoice, cat);
  };

  const handleFeedback = async (msgId: string, rating: number, query: string, answer: string) => {
    try {
      await fetch('http://localhost:8000/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, answer, rating, comment: `Rating: ${rating}/5` })
      });
      setFeedbackSent(prev => ({ ...prev, [msgId]: true }));
    } catch {
      setFeedbackSent(prev => ({ ...prev, [msgId]: true }));
    }
  };

  return (
    <div 
      className="chat-layout"
      style={{
        display: 'grid',
        gridTemplateColumns: showDebugSidebar ? '1fr 340px' : '1fr',
        gap: '1.5rem',
        alignItems: 'start'
      }}
    >
      <div className="chat-main">
        {/* Controls Bar */}
        <div className="chat-controls-bar">
          <div className="model-selector-group">
            <span className="control-label">Retrieval Engine:</span>
            <div className="model-pill-selector">
              <button
                type="button"
                className={`model-pill ${modelChoice === 'hybrid' ? 'active' : ''}`}
                onClick={() => setModelChoice('hybrid')}
                title="BM25 Lexical + Fine-tuned Dense Embeddings (Highest Recall & MRR)"
              >
                Hybrid (BM25 + Dense)
              </button>
              <button
                type="button"
                className={`model-pill ${modelChoice === 'finetuned' ? 'active' : ''}`}
                onClick={() => setModelChoice('finetuned')}
                title="Fine-tuned CIT MiniLM-L6-v2 Embeddings"
              >
                Fine-tuned Dense
              </button>
              <button
                type="button"
                className={`model-pill ${modelChoice === 'baseline' ? 'active' : ''}`}
                onClick={() => setModelChoice('baseline')}
                title="Pretrained all-MiniLM-L6-v2 Embeddings"
              >
                Baseline MiniLM
              </button>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginLeft: 'auto' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <span className="control-label">Domain:</span>
              <select
                className="category-select"
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
              >
                {CATEGORIES.map(c => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>

            {/* Developer Debug Toggle Button */}
            <button
              type="button"
              onClick={() => setShowDebugSidebar(!showDebugSidebar)}
              className={`model-pill ${showDebugSidebar ? 'active' : ''}`}
              style={{ fontSize: '0.75rem', padding: '0.35rem 0.65rem' }}
              title="Toggle Developer Debug / Evidence Inspector"
            >
              <Layers size={13} style={{ marginRight: '4px' }} />
              {showDebugSidebar ? 'Debug: On' : 'Debug: Off'}
            </button>
          </div>
        </div>

        {/* Message List */}
        <div className="chat-messages">
          {messages.length === 0 ? (
            <div className="welcome-container animate-fade-in">
              <div className="welcome-icon-box">
                <Sparkles size={32} />
              </div>
              <h2 className="welcome-title">Ask CIT Knowledge AI</h2>
              <p className="welcome-desc">
                An academic RAG assistant trained on official Coimbatore Institute of Technology datasets.
                Answers are grounded strictly in official institutional records with concise, direct responses.
              </p>

              <div className="suggested-prompts-grid">
                {SUGGESTED_QUESTIONS.map((item, idx) => (
                  <button
                    key={idx}
                    type="button"
                    className="prompt-card"
                    onClick={() => handleSelectPrompt(item.q)}
                  >
                    <div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--color-primary)', fontWeight: 700, textTransform: 'uppercase', marginBottom: '0.2rem' }}>
                        {item.category}
                      </div>
                      <div>{item.q}</div>
                    </div>
                    <ChevronRight size={16} className="prompt-arrow" />
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg) => (
              <div key={msg.id} className={`message-row ${msg.sender} animate-fade-in`}>
                <div className={`avatar ${msg.sender}`}>
                  {msg.sender === 'user' ? 'U' : 'CIT'}
                </div>

                <div className="message-bubble">
                  {/* Multi-Source Provenance Badge */}
                  {msg.sender === 'assistant' && msg.source_type && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.65rem', flexWrap: 'wrap' }}>
                      {msg.source_type === 'dataset' && (
                        <span className="source-type-badge source-type-dataset">
                          <CheckCircle2 size={12} /> Source: CIT Knowledge Dataset
                        </span>
                      )}
                      {msg.source_type === 'official_cit_website' && (
                        <span className="source-type-badge source-type-official">
                          <Globe size={12} /> Source: Official CIT Website
                        </span>
                      )}
                      {msg.source_type === 'public_web' && (
                        <span className="source-type-badge source-type-public">
                          <Search size={12} /> Source: Public Web Search
                        </span>
                      )}
                      {msg.confidence !== undefined && msg.confidence > 0 && (
                        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600 }}>
                          • {Math.round(msg.confidence * 100)}% verified
                        </span>
                      )}
                      {msg.verified_at && (
                        <span style={{ fontSize: '0.7rem', color: 'var(--text-light)' }}>
                          • Checked {msg.verified_at}
                        </span>
                      )}
                    </div>
                  )}

                  {/* Clean, Direct User-Facing Answer */}
                  <div style={{ whiteSpace: 'pre-wrap', lineHeight: '1.5' }}>{msg.text}</div>

                  {/* Conflict Notice Alert Box (Displayed only if query explicitly targets discrepancy) */}
                  {msg.has_conflicts && msg.conflict_notes && msg.conflict_notes.length > 0 && (
                    <div className="conflict-box" style={{ marginTop: '0.65rem' }}>
                      <div className="conflict-title">
                        <AlertTriangle size={15} />
                        <span>Official Source Discrepancy Note</span>
                      </div>
                      <ul style={{ paddingLeft: '1.25rem', marginTop: '0.25rem' }}>
                        {msg.conflict_notes.map((note, idx) => (
                          <li key={idx} style={{ fontSize: '0.8rem' }}>{note}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Short, Relevant Citations (1-2 maximum) */}
                  {msg.citations && msg.citations.length > 0 && (
                    <div className="citations-container" style={{ marginTop: '0.65rem' }}>
                      <div className="citations-header">
                        <BookOpen size={13} />
                        <span>{msg.citations.length === 1 ? 'Source:' : 'Sources:'}</span>
                      </div>
                      <div className="citations-list">
                        {msg.citations.map((cite, cIdx) => (
                          <div key={cIdx} className="citation-chip" style={{ padding: '0.35rem 0.65rem' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                              <a
                                href={cite.source_url}
                                target="_blank"
                                rel="noreferrer"
                                className="citation-link"
                                style={{ fontWeight: 600, display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}
                              >
                                <span>{cite.document_title}</span>
                                <ExternalLink size={12} />
                              </a>
                              {cite.source_type === 'official_cit_website' && (
                                <span className="citation-badge" style={{ color: '#1D4ED8', borderColor: '#BFDBFE' }}>Official Web</span>
                              )}
                              {cite.source_type === 'public_web' && (
                                <span className="citation-badge" style={{ color: '#7E22CE', borderColor: '#E9D5FF' }}>Public Web</span>
                              )}
                              {cite.source_type === 'dataset' && (
                                <span className="citation-badge" style={{ color: '#15803D', borderColor: '#BBF7D0' }}>Dataset</span>
                              )}
                              {cite.academic_year && cite.academic_year !== 'N/A' && (
                                <span className="citation-badge">
                                  <Clock size={10} style={{ marginRight: '3px' }} />
                                  {cite.academic_year}
                                </span>
                              )}
                              {cite.verified_date && (
                                <span className="citation-badge">
                                  <Clock size={10} style={{ marginRight: '3px' }} />
                                  {cite.verified_date}
                                </span>
                              )}
                              <span className="citation-badge">{cite.category}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Metadata Bar & Inspect Evidence Action */}
                  {msg.sender === 'assistant' && (
                    <div className="message-meta" style={{ marginTop: '0.65rem' }}>
                      <span className="model-tag">Model: {msg.model_used || modelChoice}</span>
                      {msg.latency_ms !== undefined && (
                        <span className="latency-tag">{msg.latency_ms} ms</span>
                      )}
                      {msg.retrieved_evidence && msg.retrieved_evidence.length > 0 && (
                        <button
                          type="button"
                          style={{
                            background: 'transparent',
                            color: 'var(--color-primary)',
                            fontSize: '0.725rem',
                            fontWeight: 600,
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.25rem',
                            cursor: 'pointer'
                          }}
                          onClick={() => {
                            setSelectedEvidence(msg.retrieved_evidence || []);
                            setShowDebugSidebar(true);
                          }}
                          title="View retrieved chunks in developer panel"
                        >
                          <Layers size={12} />
                          <span>Inspect Evidence ({msg.retrieved_evidence.length})</span>
                        </button>
                      )}
                      <div style={{ marginLeft: 'auto', display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
                        {feedbackSent[msg.id] ? (
                          <span style={{ color: 'var(--color-success)', fontSize: '0.7rem', fontWeight: 600 }}>Feedback Recorded!</span>
                        ) : (
                          <>
                            <button
                              type="button"
                              onClick={() => handleFeedback(msg.id, 5, '', msg.text)}
                              style={{ background: 'transparent', color: 'var(--text-light)', padding: '0.2rem', cursor: 'pointer' }}
                              title="Helpful"
                            >
                              <ThumbsUp size={14} />
                            </button>
                            <button
                              type="button"
                              onClick={() => handleFeedback(msg.id, 1, '', msg.text)}
                              style={{ background: 'transparent', color: 'var(--text-light)', padding: '0.2rem', cursor: 'pointer' }}
                              title="Needs Improvement"
                            >
                              <ThumbsDown size={14} />
                            </button>
                          </>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))
          )}

          {isLoading && (
            <div className="message-row assistant animate-fade-in">
              <div className="avatar assistant">CIT</div>
              <div className="message-bubble" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <div className="status-dot"></div>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                  Searching official CIT records & synthesizing grounded response...
                </span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar */}
        <div className="chat-input-bar">
          <form onSubmit={handleSubmit} className="chat-input-wrapper">
            <input
              type="text"
              className="chat-input"
              placeholder="Ask a question about CIT admissions, fees, placements, regulations, or research..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={isLoading}
              id="chat-input-field"
            />
            <button
              type="submit"
              className="chat-send-btn"
              disabled={!input.trim() || isLoading}
              id="chat-submit-btn"
            >
              <span>Send</span>
              <Send size={15} />
            </button>
          </form>
        </div>
      </div>

      {/* Developer-Only Evidence Inspector Sidebar (Collapsed by default) */}
      {showDebugSidebar && (
        <aside className="evidence-sidebar animate-fade-in">
          <div className="sidebar-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
              <Layers size={18} color="var(--color-primary)" />
              <span>Developer Evidence Inspector</span>
            </div>
            <button
              type="button"
              onClick={() => setShowDebugSidebar(false)}
              style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', padding: '0.2rem' }}
              title="Close Evidence Inspector"
            >
              <X size={16} />
            </button>
          </div>

          {selectedEvidence.length === 0 ? (
            <div style={{ textAlign: 'center', color: 'var(--text-light)', padding: '2rem 1rem', fontSize: '0.85rem' }}>
              Click &quot;Inspect Evidence&quot; on any assistant message to review raw retrieved chunks and similarity scores.
            </div>
          ) : (
            selectedEvidence.map((ev, idx) => (
              <div key={idx} className="evidence-card">
                <div className="evidence-header">
                  <div>
                    <div className="evidence-source">{ev.metadata.document_title}</div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                      {ev.metadata.section_heading}
                    </div>
                  </div>
                  <span className="evidence-score">
                    {ev.rrf_score !== undefined
                      ? `RRF ${ev.rrf_score.toFixed(4)}`
                      : `Sim ${(ev.score || 0).toFixed(4)}`}
                  </span>
                </div>
                <p className="evidence-text">{ev.text}</p>
                <div style={{ marginTop: '0.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.7rem' }}>
                  <span style={{ color: 'var(--text-light)' }}>ID: {ev.chunk_id}</span>
                  <a
                    href={ev.metadata.source_url}
                    target="_blank"
                    rel="noreferrer"
                    style={{ display: 'flex', alignItems: 'center', gap: '0.2rem', fontWeight: 600 }}
                  >
                    Source <ExternalLink size={10} />
                  </a>
                </div>
              </div>
            ))
          )}
        </aside>
      )}
    </div>
  );
};
