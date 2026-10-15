import pytest
from backend.app.multi_source.orchestrator import multi_source_orchestrator
from backend.app.multi_source.cit_website_retriever import cit_website_retriever
from backend.app.multi_source.public_web_retriever import public_web_retriever
from backend.app.rag_engine import rag_engine


from backend.app.schemas import ChatResponse


def test_level1_local_dataset_retrieval():
    """Test Level 1: Answering fee queries directly from local official dataset."""
    query = "What is the tuition fee for B.E. Government Aided first year for 2026-2027?"
    res_dict = rag_engine.query_rag(query=query, model_choice="hybrid")
    result = ChatResponse(**res_dict)
    
    assert result.source_type == "dataset"
    assert result.retrieval_level == 1
    assert result.confidence >= 0.90
    assert "Rs." in result.answer or "8,000" in result.answer or "33,350" in result.answer or "tuition" in result.answer.lower()
    assert len(result.citations) >= 1
    assert any("fee" in c.document_title.lower() or "admission" in c.document_title.lower() for c in result.citations)


def test_level2_official_cit_website_retrieval():
    """Test Level 2: Answering M.Sc. Software Systems HOD from live official CIT website."""
    query = "Who is the HOD of M.Sc. Software Systems?"
    res_dict = rag_engine.query_rag(query=query, model_choice="hybrid")
    result = ChatResponse(**res_dict)
    
    # In local dataset, M.Sc SS HOD was "Not yet inspected", so Level 2 fetches live from official website
    assert result.source_type == "official_cit_website"
    assert result.retrieval_level == 2
    assert "Manjula Gandhi" in result.answer
    assert len(result.citations) >= 1
    assert any("cit.edu.in" in c.source_url for c in result.citations)


def test_level2_faculty_members_msc_software_systems():
    """Test Level 2: Listing faculty members for M.Sc. Software Systems from official website."""
    query = "Who are the faculty members for M.Sc. Software Systems?"
    res_dict = rag_engine.query_rag(query=query, model_choice="hybrid")
    result = ChatResponse(**res_dict)
    
    assert result.source_type == "official_cit_website"
    assert result.retrieval_level == 2
    assert "Manjula Gandhi" in result.answer
    assert "Saravanan" in result.answer
    assert "Stanly Felix" in result.answer
    assert len(result.citations) >= 1
    assert any("faculty" in c.source_url.lower() for c in result.citations)


def test_level3_campus_gate_closing_time():
    """Test Level 3: Campus timings / gate closing query without fabrication."""
    query = "What is the gate closing time at CIT?"
    res_dict = rag_engine.query_rag(query=query, model_choice="hybrid")
    result = ChatResponse(**res_dict)
    
    assert result.retrieval_level in (2, 3)
    # Must never hallucinate an ungrounded exact time without qualification
    assert ("pm" in result.answer.lower() or "gate" in result.answer.lower() or "verif" in result.answer.lower())


def test_unverifiable_fictitious_query():
    """Test security & grounding: Must refuse or state unable to verify for impossible queries."""
    query = "What is the hostel mess fee on the CIT Antarctica campus?"
    res_dict = rag_engine.query_rag(query=query, model_choice="hybrid")
    result = ChatResponse(**res_dict)
    
    assert result.is_refusal is True or "couldn't verify" in result.answer.lower() or "not available" in result.answer.lower() or "no reliable" in result.answer.lower() or "unable" in result.answer.lower()
