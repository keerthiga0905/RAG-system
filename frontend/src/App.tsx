import React, { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header';
import { ChatInterface } from './components/ChatInterface';
import { HistoryDashboard } from './components/HistoryDashboard';
import { EvaluationDashboard } from './components/EvaluationDashboard';
import { DatasetInventory } from './components/DatasetInventory';
import { ModelInfoView } from './components/ModelInfo';
import { LoginPage } from './components/LoginPage';
import type { ChatMessage, RetrievedEvidence, UserProfile, HistoryRecord } from './types';

const DEFAULT_USER: UserProfile = {
  email: 'keerthiga@cit.edu.in',
  name: 'Keerthiga M',
  role: 'Student',
  department: 'M.Sc. Software Systems',
  avatarColor: '#800000',
  createdAt: '2026-10-15'
};

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'chat' | 'history' | 'evaluation' | 'documents' | 'model'>('chat');
  
  // User Authentication State
  const [currentUser, setCurrentUser] = useState<UserProfile>(() => {
    try {
      const saved = localStorage.getItem('cit_active_user');
      return saved ? JSON.parse(saved) : DEFAULT_USER;
    } catch {
      return DEFAULT_USER;
    }
  });
  const [showLoginModal, setShowLoginModal] = useState<boolean>(false);

  // Chat & History State per User
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [userHistory, setUserHistory] = useState<HistoryRecord[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [systemHealthy, setSystemHealthy] = useState<boolean>(false);
  const [totalChunks, setTotalChunks] = useState<number>(290);
  const [selectedEvidence, setSelectedEvidence] = useState<RetrievedEvidence[]>([]);

  // Load chat messages and history for active user
  const loadUserData = useCallback(async (user: UserProfile) => {
    // 1. Load messages from localStorage
    try {
      const savedMsgs = localStorage.getItem(`cit_messages_${user.email}`);
      if (savedMsgs) {
        setMessages(JSON.parse(savedMsgs));
      } else {
        setMessages([]);
      }
    } catch {
      setMessages([]);
    }

    // 2. Load history from localStorage first (for speed)
    let localHistory: HistoryRecord[] = [];
    try {
      const savedHist = localStorage.getItem(`cit_history_${user.email}`);
      if (savedHist) {
        localHistory = JSON.parse(savedHist);
        setUserHistory(localHistory);
      }
    } catch {
      // fallback
    }

    // 3. Sync history with backend
    try {
      const res = await fetch(`http://localhost:8000/api/history?email=${encodeURIComponent(user.email)}`);
      if (res.ok) {
        const data = await res.json();
        if (data.history && Array.isArray(data.history)) {
          setUserHistory(data.history);
          localStorage.setItem(`cit_history_${user.email}`, JSON.stringify(data.history));
        }
      }
    } catch (err) {
      console.warn('Backend history sync unavailable, using local history:', err);
    }
  }, []);

  // On mount or user change, load data
  useEffect(() => {
    loadUserData(currentUser);
  }, [currentUser, loadUserData]);

  // System Health Check
  useEffect(() => {
    fetch('http://localhost:8000/health')
      .then(res => res.json())
      .then(data => {
        if (data.status === 'healthy') {
          setSystemHealthy(true);
          if (data.vector_store_chunks_count) {
            setTotalChunks(data.vector_store_chunks_count);
          }
        }
      })
      .catch(() => setSystemHealthy(false));
  }, []);

  // Handle Login / User Switch
  const handleLogin = (profile: UserProfile) => {
    setCurrentUser(profile);
    localStorage.setItem('cit_active_user', JSON.stringify(profile));
    setShowLoginModal(false);
    loadUserData(profile);
  };

  const handleLogout = () => {
    setShowLoginModal(true);
  };

  // Clear History
  const handleClearHistory = async () => {
    setUserHistory([]);
    setMessages([]);
    localStorage.removeItem(`cit_history_${currentUser.email}`);
    localStorage.removeItem(`cit_messages_${currentUser.email}`);

    try {
      await fetch(`http://localhost:8000/api/history?email=${encodeURIComponent(currentUser.email)}`, {
        method: 'DELETE'
      });
    } catch (err) {
      console.error('Failed to clear backend history:', err);
    }
  };

  // Handle Send Message
  const handleSendMessage = async (query: string, modelChoice: string, categoryFilter?: string) => {
    const userMsgId = `user_${Date.now()}`;
    const userMsg: ChatMessage = {
      id: userMsgId,
      sender: 'user',
      text: query,
      timestamp: new Date().toLocaleTimeString()
    };

    const updatedMessages = [...messages, userMsg];
    setMessages(updatedMessages);
    localStorage.setItem(`cit_messages_${currentUser.email}`, JSON.stringify(updatedMessages));
    setIsLoading(true);

    try {
      const res = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          model_choice: modelChoice,
          category_filter: categoryFilter,
          top_k: 4
        })
      });

      if (!res.ok) {
        throw new Error(`Server returned HTTP ${res.status}`);
      }

      const data = await res.json();
      const assistantMsg: ChatMessage = {
        id: `assistant_${Date.now()}`,
        sender: 'assistant',
        text: data.answer,
        timestamp: new Date().toLocaleTimeString(),
        model_used: data.model_used,
        latency_ms: data.latency_ms,
        citations: data.citations,
        retrieved_evidence: data.retrieved_evidence,
        has_conflicts: data.has_conflicts,
        conflict_notes: data.conflict_notes,
        is_refusal: data.is_refusal,
        category: data.category,
        source_type: data.source_type,
        confidence: data.confidence,
        retrieval_level: data.retrieval_level,
        verified_at: data.verified_at
      };

      const finalMessages = [...updatedMessages, assistantMsg];
      setMessages(finalMessages);
      localStorage.setItem(`cit_messages_${currentUser.email}`, JSON.stringify(finalMessages));

      if (data.retrieved_evidence && data.retrieved_evidence.length > 0) {
        setSelectedEvidence(data.retrieved_evidence);
      }

      // Record in History
      const historyItem: HistoryRecord = {
        id: `hist_${Date.now()}`,
        email: currentUser.email,
        timestamp: new Date().toLocaleString(),
        query,
        answer: data.answer,
        model_used: data.model_used || modelChoice,
        retrieval_level: data.retrieval_level || 1,
        source_type: data.source_type || 'dataset',
        category: data.category || 'General',
        latency_ms: data.latency_ms || 0,
        citations: data.citations || [],
        is_refusal: !!data.is_refusal
      };

      const updatedHistory = [historyItem, ...userHistory];
      setUserHistory(updatedHistory);
      localStorage.setItem(`cit_history_${currentUser.email}`, JSON.stringify(updatedHistory));

      // Persist to backend
      fetch('http://localhost:8000/api/history', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: currentUser.email,
          record: historyItem
        })
      }).catch(e => console.warn('History background sync failed:', e));

    } catch (err: unknown) {
      const errorMsg: ChatMessage = {
        id: `error_${Date.now()}`,
        sender: 'assistant',
        text: `Unable to connect to CIT Knowledge AI backend: ${err instanceof Error ? err.message : 'Connection failed'}. Please ensure the FastAPI backend is running on port 8000.`,
        timestamp: new Date().toLocaleTimeString()
      };
      const finalMessages = [...updatedMessages, errorMsg];
      setMessages(finalMessages);
      localStorage.setItem(`cit_messages_${currentUser.email}`, JSON.stringify(finalMessages));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        systemHealthy={systemHealthy}
        totalChunks={totalChunks}
        currentUser={currentUser}
        onOpenLogin={() => setShowLoginModal(true)}
        onLogout={handleLogout}
      />

      {/* Login Screen / Modal */}
      {showLoginModal && (
        <LoginPage
          activeUser={currentUser}
          onLogin={handleLogin}
          onCancel={() => setShowLoginModal(false)}
        />
      )}

      <main className="main-content">
        {activeTab === 'chat' && (
          <ChatInterface
            messages={messages}
            onSendMessage={handleSendMessage}
            isLoading={isLoading}
            selectedEvidence={selectedEvidence}
            setSelectedEvidence={setSelectedEvidence}
          />
        )}

        {activeTab === 'history' && (
          <HistoryDashboard
            user={currentUser}
            history={userHistory}
            onClearHistory={handleClearHistory}
            onResumeInChat={(query) => {
              setActiveTab('chat');
              handleSendMessage(query, 'hybrid');
            }}
            onRefresh={() => loadUserData(currentUser)}
          />
        )}

        {activeTab === 'evaluation' && <EvaluationDashboard />}

        {activeTab === 'documents' && <DatasetInventory />}

        {activeTab === 'model' && <ModelInfoView />}
      </main>
    </div>
  );
};

export default App;
