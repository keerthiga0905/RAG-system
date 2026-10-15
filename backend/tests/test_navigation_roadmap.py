import os
import json
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.roadmap_navigator import roadmap_navigator
from backend.app.rag_engine import rag_engine

client = TestClient(app)


def test_roadmap_model_artifact_integrity():
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "cit_roadmap_model.json")
    assert os.path.exists(model_path), "CIT roadmap model artifact must exist"
    with open(model_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["locations"]) == 8
    assert len(data["edges"]) == 18
    assert len(data["precomputed_routes"]) == 56


def test_navigate_computing_to_library():
    query = "Navigate me from Computing block to Library block"
    res = roadmap_navigator.navigate(query)
    assert res is not None
    assert res["success"] is True
    assert "Computing Block" in res["answer"]
    assert "Library Block" in res["answer"]
    assert "IT Block" in res["answer"]
    assert res["distance_m"] == 125
    assert "Step-by-step walking navigation" in res["answer"]


def test_navigate_main_block_to_auditorium():
    query = "Navigate from Main Block to Auditorium"
    res = roadmap_navigator.navigate(query)
    assert res is not None
    assert res["success"] is True
    assert "Main Building / Main Block" in res["answer"]
    assert "CIT Auditorium" in res["answer"]
    assert res["distance_m"] == 70


def test_navigate_to_same_location():
    query = "Navigate from Computing Block to Computing Block"
    res = roadmap_navigator.navigate(query)
    assert res is not None
    assert "already at" in res["answer"].lower()


def test_api_chat_navigation_level1_integration():
    response = client.post("/api/chat", json={
        "query": "Navigate me from Computing block to Library block",
        "model_choice": "hybrid"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["source_type"] == "dataset"
    assert data["retrieval_level"] == 1
    assert data["is_refusal"] is False
    assert "Step-by-step walking navigation from Computing Block to Library Block" in data["answer"]
    assert any("CIT Campus Navigation Roadmap" in c["document_title"] for c in data["citations"])
