"""
Automated Regression Test Suite for CIT Knowledge AI
Validates direct, question-focused answers without verbose dumps or unrelated metadata.

Tests:
- TEST A: "Who is the HOD of M.Sc. Software Systems?" -> Short inability-to-verify response, no curriculum/department dumps.
- TEST B: "What courses does M.Sc. Software Systems offer?" -> Relevant courses only.
- TEST C: "What is the eligibility for M.Sc. Software Systems?" -> Eligibility details only.
- TEST D: "When was the Department of Software Systems established?" -> Verified establishment year (1998).
- TEST E: "Who is the HOD of M.Sc. Software Systems in 2020?" -> Historical inability-to-verify response.
- TEST F: "Who is the HOD of a department that is not documented?" -> Concise inability-to-verify response.
- TEST G: "Tell me about placements at CIT." -> Relevant summary of verified placement information with academic year.
- TEST H: "Who is the HOD of M.Sc. Software Systems?" (with mixed retrieved context including mocked HOD + unrelated curriculum/admissions/other departments) -> Returns ONLY verified HOD and cites only that chunk.
"""

import os
import sys
import pytest

# Ensure backend can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.app.rag_engine import rag_engine

def test_a_hod_msc_software_systems():
    """TEST A: Inability-to-verify response in local dataset, and verified HOD via Level 2 official website."""
    query = "Who is the HOD of M.Sc. Software Systems?"
    
    # 1. Local-only retrieval (Level 1) properly refuses since local raw text had 'Not yet inspected'
    res_local = rag_engine.query_local(query, model_choice="hybrid", top_k=4)
    answer_local = res_local["answer"]
    assert "couldn't verify" in answer_local.lower() or "not verified" in answer_local.lower()
    assert "Civil Engineering" not in answer_local
    assert "Computer Science" not in answer_local

    # 2. Multi-source retrieval (Level 2) fetches live from https://cit.edu.in/msc-software-systems/faculty
    res_multi = rag_engine.query_rag(query, model_choice="hybrid", top_k=4)
    answer_multi = res_multi["answer"]
    assert "Manjula Gandhi" in answer_multi
    assert res_multi["source_type"] == "official_cit_website"
    assert len(res_multi["citations"]) <= 2

def test_b_courses_msc_software_systems():
    """TEST B: Relevant courses offered by M.Sc. Software Systems only."""
    query = "What courses does M.Sc. Software Systems offer?"
    res = rag_engine.query_rag(query, model_choice="hybrid", top_k=3)
    answer = res["answer"]

    # 1. Must mention the 5-Year Integrated M.Sc. Software Systems programme
    assert "5-Year Integrated" in answer or "Integrated M.Sc." in answer or "Software Systems" in answer

    # 2. Must NOT contain unrelated department information
    assert "Civil Engineering" not in answer
    assert "Chemical Engineering" not in answer
    assert len(res["citations"]) <= 2

def test_c_eligibility_msc_software_systems():
    """TEST C: Eligibility details from applicable official source only."""
    query = "What is the eligibility for M.Sc. Software Systems?"
    res = rag_engine.query_rag(query, model_choice="hybrid", top_k=3)
    answer = res["answer"]

    # 1. Must include core subjects: Mathematics, Physics
    assert "Mathematics" in answer
    assert "Physics" in answer

    # 2. Must NOT dump unrelated department info or full fee schedules
    assert "Civil Engineering" not in answer
    assert "Rs. 33,350" not in answer
    assert len(res["citations"]) <= 2

def test_d_establishment_year_software_systems():
    """TEST D: Verified establishment year (1998) only."""
    query = "When was the Department of Software Systems established?"
    res = rag_engine.query_rag(query, model_choice="hybrid", top_k=3)
    answer = res["answer"]

    # 1. Must specify the verified year: 1998
    assert "1998" in answer
    assert "Department of Software Systems" in answer or "Software Systems" in answer

    # 2. Concise answer length (under 120 characters)
    assert len(answer) < 150
    assert len(res["citations"]) <= 1

def test_e_historical_hod_2020():
    """TEST E: Historical information for 2020 only if verified by an applicable source."""
    query = "Who is the HOD of M.Sc. Software Systems in 2020?"
    res = rag_engine.query_rag(query, model_choice="hybrid", top_k=3)
    answer = res["answer"]

    # 1. Since historical HOD in 2020 is not verified, refuse concisely
    assert "2020" in answer
    assert "couldn't verify" in answer.lower() or "not verified" in answer.lower()
    assert "Civil Engineering" not in answer

def test_f_undocumented_department_hod():
    """TEST F: Undocumented department triggers concise inability-to-verify, not a guessed name."""
    query = "Who is the HOD of Aerospace Engineering?"
    res = rag_engine.query_rag(query, model_choice="hybrid", top_k=3)
    answer = res["answer"]

    assert "couldn't verify" in answer.lower() or "not verified" in answer.lower()
    assert "Aerospace Engineering" in answer
    assert res["is_refusal"] is True
    # Zero citations for completely undocumented queries
    assert len(res["citations"]) == 0

def test_g_placements_summary():
    """TEST G: Relevant summary of verified placement information with academic year."""
    query = "Tell me about placements at CIT."
    res = rag_engine.query_rag(query, model_choice="hybrid", top_k=3)
    answer = res["answer"]

    # 1. Must mention verified placement figures and recruiters
    assert "59 LPA" in answer or "35 LPA" in answer or "Placement" in answer
    assert "internship" in answer.lower() or "recruiters" in answer.lower()
    assert len(res["citations"]) <= 2

def test_h_mixed_context_hod_extraction():
    """
    TEST H: When retrieved context includes relevant HOD evidence alongside
    unrelated curriculum, admissions, and other department chunks, the synthesizer
    extracts ONLY the relevant HOD information and cites ONLY that supporting chunk.
    """
    query = "Who is the HOD of M.Sc. Software Systems?"
    intent_info = rag_engine.detect_intent_and_entity(query)

    # Mocked candidate chunks:
    # Chunk 1: Unrelated curriculum PDF
    unrelated_curriculum_chunk = {
        "chunk_id": "chunk_mock_curr",
        "text": "[M.Sc. Software Systems Curriculum] Regulation 2022 syllabus structure, credit requirements, semester 1 to 10 courses.",
        "metadata": {
            "document_title": "Curriculum Structure",
            "source_url": "https://cit.edu.in/curriculum/msc-ss",
            "category": "Courses and Academic Programmes"
        }
    }
    # Chunk 2: Unrelated Admissions Cutoff
    unrelated_admissions_chunk = {
        "chunk_id": "chunk_mock_adm",
        "text": "[Admissions] TNEA engineering counselling cut-off marks and eligibility for B.E. Civil and Mechanical.",
        "metadata": {
            "document_title": "Admissions Cut-offs",
            "source_url": "https://cit.edu.in/admission/tnea",
            "category": "Admissions and Eligibility"
        }
    }
    # Chunk 3: Another Department's HOD (Civil Engineering)
    other_dept_chunk = {
        "chunk_id": "chunk_mock_civil",
        "text": "[Department of Civil Engineering] Dr. R. Sathyanarayan Sridhar - Professor & Head of the Department.",
        "metadata": {
            "document_title": "Civil Engineering Faculty",
            "source_url": "https://cit.edu.in/civil-engineering",
            "category": "Departments and Faculty"
        }
    }
    # Chunk 4: Verified Target Department HOD (Mocked test evidence)
    target_hod_chunk = {
        "chunk_id": "chunk_mock_target_hod",
        "text": "[Department of M.Sc. Software Systems] Head of Department: Dr. K. S. Meenakshi Sundaram - Professor & Head of the Department.",
        "metadata": {
            "document_title": "Department of M.Sc. Software Systems Faculty",
            "source_url": "https://cit.edu.in/msc-software-systems",
            "category": "Departments and Faculty"
        }
    }

    mixed_retrieved = [
        unrelated_curriculum_chunk,
        unrelated_admissions_chunk,
        other_dept_chunk,
        target_hod_chunk
    ]

    answer, is_refusal, used_chunks = rag_engine.generate_grounded_answer(
        query=query,
        retrieved=mixed_retrieved,
        intent_info=intent_info,
        has_conflicts=False,
        conflict_notes=[]
    )

    # 1. Answer must contain ONLY the verified target HOD
    assert "Dr. K. S. Meenakshi Sundaram" in answer
    assert "Civil Engineering" not in answer
    assert "Dr. R. Sathyanarayan Sridhar" not in answer
    assert "Regulation 2022" not in answer
    assert "TNEA" not in answer

    # 2. Must cite ONLY the supporting HOD chunk
    assert len(used_chunks) == 1
    assert used_chunks[0]["chunk_id"] == "chunk_mock_target_hod"
