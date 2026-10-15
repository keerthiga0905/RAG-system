export interface Citation {
  chunk_id: string;
  document_title: string;
  source_url: string;
  category: string;
  academic_year?: string | null;
  similarity_score?: number | null;
  text_snippet: string;
}

export interface RetrievedEvidence {
  chunk_id: string;
  score?: number;
  rrf_score?: number;
  metadata: {
    chunk_id: string;
    original_document_id?: string;
    document_title: string;
    category: string;
    source_url: string;
    academic_year?: string;
    section_heading?: string;
    page_number?: number;
  };
  text: string;
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  timestamp: string;
  model_used?: string;
  latency_ms?: number;
  citations?: Citation[];
  retrieved_evidence?: RetrievedEvidence[];
  has_conflicts?: boolean;
  conflict_notes?: string[];
  is_refusal?: boolean;
  category?: string;
}

export interface DocumentSummary {
  document_id: string;
  title: string;
  category: string;
  format: string;
  chunks_count: number;
  source_url: string;
  academic_year?: string | null;
}

export interface EvaluationMetrics {
  num_evaluated_queries: number;
  "recall@1": number;
  "recall@3": number;
  "recall@5": number;
  "precision@1": number;
  "precision@3": number;
  "precision@5": number;
  mrr: number;
  citation_accuracy: number;
  out_of_scope_refusal_rate: number;
  avg_latency_ms: number;
  category_breakdown: Record<string, {
    count: number;
    "recall@3": number;
    mrr: number;
  }>;
  failures_count: number;
}

export interface EvaluationReport {
  evaluation_timestamp: string;
  held_out_test_set_size: number;
  configurations: {
    baseline_dense: EvaluationMetrics;
    finetuned_dense: EvaluationMetrics;
    hybrid_bm25_dense: EvaluationMetrics;
  };
}

export interface ModelInfo {
  baseline_model: string;
  finetuned_model: string;
  vector_database: string;
  total_chunks_indexed: number;
  default_strategy: string;
  supported_strategies: string[];
  training_loss: number;
  fine_tuning_epochs: number;
  leakage_resistance: string;
}
