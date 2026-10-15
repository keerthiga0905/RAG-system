import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { ChatInterface } from './components/ChatInterface';
import { EvaluationDashboard } from './components/EvaluationDashboard';
import { DatasetInventory } from './components/DatasetInventory';
import { ModelInfoView } from './components/ModelInfo';
import type { ChatMessage, RetrievedEvidence } from './types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'chat' | 'evaluation' | 'documents' | 'model'>('chat');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [systemHealthy, setSystemHealthy] = useState<boolean>(false);
  const [totalChunks, setTotalChunks] = useState<number>(290);
  const [selectedEvidence, setSelectedEvidence] = useState<RetrievedEvidence[]>([]);

  useEffect(() => {
    // Check health
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

  const handleSendMessage = async (query: string, modelChoice: string, categoryFilter?: string) => {
    const userMsgId = `user_${Date.now()}`;
    const userMsg: ChatMessage = {
      id: userMsgId,
      sender: 'user',
      text: query,
      timestamp: new Date().toLocaleTimeString()
    };

    setMessages(prev => [...prev, userMsg]);
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

      setMessages(prev => [...prev, assistantMsg]);
      if (data.retrieved_evidence && data.retrieved_evidence.length > 0) {
        setSelectedEvidence(data.retrieved_evidence);
      }
    } catch (err: unknown) {
      const errorMsg: ChatMessage = {
        id: `error_${Date.now()}`,
        sender: 'assistant',
        text: `Unable to connect to CIT Knowledge AI backend: ${err instanceof Error ? err.message : 'Connection failed'}. Please ensure the FastAPI backend is running on port 8000.`,
        timestamp: new Date().toLocaleTimeString()
      };
      setMessages(prev => [...prev, errorMsg]);
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
      />

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

        {activeTab === 'evaluation' && <EvaluationDashboard />}

        {activeTab === 'documents' && <DatasetInventory />}

        {activeTab === 'model' && <ModelInfoView />}
      </main>
    </div>
  );
};

export default App;
