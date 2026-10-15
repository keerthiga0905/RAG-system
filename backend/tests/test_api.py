"""
Unit and Integration Tests for CIT Knowledge AI FastAPI Endpoints
Tests:
- GET /health
- GET /api/documents
- POST /api/chat
- POST /api/search
- POST /api/feedback
- GET /api/evaluation
- GET /api/model-info
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["vector_store_chunks_count"] == 290
    assert data["models_loaded"]["baseline"] is True
    assert data["models_loaded"]["finetuned"] is True

def test_documents_endpoint():
    response = client.get("/api/documents")
    assert response.status_code == 200
    docs = response.json()
    assert len(docs) == 14
    assert any(d["document_id"] == "01_college_overview.txt" for d in docs)
    assert any(d["document_id"] == "06_placements_and_career.txt" for d in docs)

def test_chat_endpoint_valid():
    payload = {
        "query": "What degree programmes are offered in Government Aided mode?",
        "model_choice": "hybrid",
        "top_k": 3
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["citations"]) > 0
    assert data["model_used"] == "hybrid"

def test_chat_endpoint_refusal():
    payload = {
        "query": "Does CIT offer a bachelor degree in astronautical exploration and space shuttle piloting?",
        "model_choice": "hybrid",
        "top_k": 3
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_refusal"] is True
    assert "I could not find sufficient verified information" in data["answer"]

def test_search_endpoint():
    payload = {
        "query": "FESEM facility charges and capabilities",
        "model_choice": "finetuned",
        "top_k": 3
    }
    response = client.post("/api/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_found"] == 3
    assert len(data["results"]) == 3

def test_feedback_endpoint():
    payload = {
        "query": "What is the fee for B.Tech AI?",
        "answer": "The fee for B.Tech AI is Rs. 87,000 for I year.",
        "rating": 5,
        "comment": "Accurate fee information grounded in official brochure."
    }
    response = client.post("/api/feedback", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

def test_evaluation_endpoint():
    response = client.get("/api/evaluation")
    assert response.status_code == 200
    data = response.json()
    assert "configurations" in data
    assert "baseline_dense" in data["configurations"]
    assert "finetuned_dense" in data["configurations"]
    assert "hybrid_bm25_dense" in data["configurations"]

def test_model_info_endpoint():
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "all-MiniLM-L6-v2" in data["baseline_model"]
    assert "cit-minilm-l6-v2" in data["finetuned_model"]
    assert data["total_chunks_indexed"] == 290
