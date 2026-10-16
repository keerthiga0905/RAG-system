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


def test_history_endpoints():
    test_email = "test.keerthiga@cit.edu.in"
    # 1. Post a history record
    payload = {
        "email": test_email,
        "record": {
            "id": "test_record_1",
            "email": test_email,
            "timestamp": "2026-10-16 10:00:00",
            "query": "Who is the HOD of M.Sc. Software Systems?",
            "answer": "Dr. S. Manjula Gandhi is the Head of the Department.",
            "model_used": "hybrid",
            "retrieval_level": 2,
            "source_type": "official_cit_website",
            "category": "Departments and Faculty",
            "latency_ms": 150,
            "citations": [],
            "is_refusal": False
        }
    }
    res_post = client.post("/api/history", json=payload)
    assert res_post.status_code == 200
    assert res_post.json()["status"] == "success"

    # 2. Get history
    res_get = client.get(f"/api/history?email={test_email}")
    assert res_get.status_code == 200
    get_data = res_get.json()
    assert get_data["email"] == test_email
    assert len(get_data["history"]) >= 1
    assert get_data["history"][0]["id"] == "test_record_1"

    # 3. List history users
    res_users = client.get("/api/history/users")
    assert res_users.status_code == 200
    assert any(u["email"] == test_email for u in res_users.json()["users"])

    # 4. Clear history
    res_del = client.delete(f"/api/history?email={test_email}")
    assert res_del.status_code == 200
    assert res_del.json()["status"] == "success"

    # Verify cleared
    res_get_cleared = client.get(f"/api/history?email={test_email}")
    assert res_get_cleared.status_code == 200
    assert len(res_get_cleared.json()["history"]) == 0
