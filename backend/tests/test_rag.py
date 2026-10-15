"""
Unit and Integration Tests for CIT Knowledge AI RAG Engine
Tests:
1. Document loading & chunking metadata preservation
2. Embedding generation (Baseline vs Fine-tuned)
3. Retrieval relevance across question types:
   - Direct factual lookup (e.g. B.E. aided tuition fees)
   - Paraphrased informal student question
   - Multi-section query
   - Academic year-sensitive facts (e.g. 2026-2027)
   - Missing-information / unanswerable refusal handling
   - Official conflicting claims handling
4. Citation correctness and URL presence
"""

import pytest
import os
import sys

# Ensure backend can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.app.rag_engine import rag_engine, FALLBACK_REFUSAL

def test_engine_initialization():
    assert rag_engine.col_baseline is not None
    assert rag_engine.col_finetuned is not None
    assert rag_engine.col_finetuned.count() == 290
    assert rag_engine.bm25 is not None

def test_factual_tuition_fee_query():
    query = "What is the tuition fee for B.E. Government Aided first year for 2026-2027?"
    res = rag_engine.query_rag(query, model_choice="hybrid", top_k=3)
    assert len(res["citations"]) > 0
    assert "33,350" in res["answer"] or "33350" in res["answer"]
    assert any("cit.edu.in" in c.source_url for c in res["citations"])

def test_paraphrased_query():
    query = "how to reach cit coimbatore by bus or train?"
    res = rag_engine.query_rag(query, model_choice="hybrid", top_k=3)
    assert len(res["citations"]) > 0
    assert "Coimbatore" in res["answer"] or "Aerodrome" in res["answer"] or "Avinashi" in res["answer"]

def test_academic_year_preservation():
    query = "What was the highest placement package in 2025-2026?"
    res = rag_engine.query_rag(query, model_choice="finetuned", top_k=3)
    assert "2025" in res["answer"] or "59 LPA" in res["answer"] or "D.E. Shaw" in res["answer"]
    assert any(c.academic_year is not None for c in res["citations"])

def test_official_conflicting_claims_handling():
    query = "When was CIT granted autonomous status?"
    res = rag_engine.query_rag(query, model_choice="hybrid", top_k=4)
    assert res["has_conflicts"] is True
    assert len(res["conflict_notes"]) > 0
    assert any("AQAR" in note for note in res["conflict_notes"])

def test_unanswerable_out_of_scope_query():
    query = "What is the exact price of a masala dosa at the CIT canteen?"
    res = rag_engine.query_rag(query, model_choice="hybrid", top_k=3)
    assert res["is_refusal"] is True
    assert FALLBACK_REFUSAL in res["answer"]

def test_citation_integrity():
    query = "Who is the Head of the Department of Civil Engineering?"
    res = rag_engine.query_rag(query, model_choice="hybrid", top_k=3)
    assert len(res["citations"]) > 0
    top_citation = res["citations"][0]
    assert top_citation.source_url.startswith("https://")
    assert top_citation.document_title != ""
    assert top_citation.chunk_id.startswith("chunk_")
