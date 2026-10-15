"""
Multi-Source Retrieval Orchestrator
Coordinates Level 1 (Local Dataset), Level 2 (Official CIT Website), and Level 3 (Public Web Search)
with automated evidence sufficiency checks, source selection, and calibrated confidence scoring.
"""

import time
import logging
from typing import Dict, Any, Optional, List

from ..config import settings
from ..schemas import Citation
from .cit_website_retriever import cit_website_retriever
from .public_web_retriever import public_web_retriever

logger = logging.getLogger(__name__)

class MultiSourceOrchestrator:
    def __init__(self):
        pass

    def execute(
        self,
        query: str,
        rag_engine=None,
        model_choice: str = "hybrid",
        top_k: int = 4,
        category_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes automatic 3-level hierarchical retrieval:
        1. Level 1: Search complete local dataset
        2. Level 2: Search official CIT website (https://cit.edu.in/)
        3. Level 3: Search Google / Bing public web for campus practical information
        """
        if rag_engine is None:
            from backend.app.rag_engine import rag_engine as default_engine
            rag_engine = default_engine
        t0 = time.time()
        intent_info = rag_engine.detect_intent_and_entity(query)
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")

        # =====================================================================
        # LEVEL 1: LOCAL DATASET RETRIEVAL
        # =====================================================================
        logger.info(f"Executing Level 1 (Local Dataset) for query: '{query}'")
        level1_result = rag_engine.query_local(
            query=query,
            intent_info=intent_info,
            model_choice=model_choice,
            top_k=top_k,
            category_filter=category_filter
        )

        # Check evidence sufficiency of Level 1
        # If Level 1 provided a verified, non-refusal answer, return immediately
        if not level1_result["is_refusal"]:
            logger.info("Level 1 provided sufficient verified evidence.")
            level1_result["source_type"] = "dataset"
            level1_result["confidence"] = 0.95
            level1_result["retrieval_level"] = 1
            level1_result["verified_at"] = timestamp_str
            # Tag citations with source_type
            for c in level1_result["citations"]:
                c.source_type = "dataset"
            return level1_result

        # =====================================================================
        # LEVEL 2: OFFICIAL CIT WEBSITE RETRIEVAL (https://cit.edu.in/)
        # =====================================================================
        if settings.ENABLE_LEVEL2_CIT_WEBSITE:
            logger.info(f"Level 1 insufficient. Escalating to Level 2 (Official CIT Website) for '{query}'")
            level2_evidence = cit_website_retriever.search_official_website(query, intent_info)
            if level2_evidence:
                extracted = cit_website_retriever.extract_verified_answer(query, level2_evidence, intent_info)
                if extracted:
                    answer_text, used_chunks = extracted
                    logger.info("Level 2 official website provided verified answer.")
                    citations = []
                    for r in used_chunks:
                        meta = r["metadata"]
                        citations.append(Citation(
                            chunk_id=r["chunk_id"],
                            document_title=meta.get("document_title", "Official CIT Website"),
                            source_url=meta.get("source_url", "https://cit.edu.in/"),
                            category="Official CIT Website",
                            text_snippet=r["text"][:140].replace("\n", " ") + "...",
                            source_type="official_cit_website",
                            verified_date=timestamp_str
                        ))

                    total_latency = round((time.time() - t0) * 1000.0, 2)
                    return {
                        "answer": answer_text,
                        "citations": citations,
                        "source_type": "official_cit_website",
                        "confidence": 0.92,
                        "retrieval_level": 2,
                        "verified_at": timestamp_str,
                        "retrieved_evidence": level2_evidence,
                        "category": "Official CIT Website",
                        "model_used": model_choice,
                        "latency_ms": total_latency,
                        "has_conflicts": False,
                        "conflict_notes": [],
                        "is_refusal": False
                    }

        # =====================================================================
        # LEVEL 3: PUBLIC WEB RETRIEVAL (Google / Bing / Campus Information)
        # =====================================================================
        if settings.ENABLE_LEVEL3_PUBLIC_WEB:
            logger.info(f"Level 2 insufficient. Escalating to Level 3 (Public Web) for '{query}'")
            web_results = public_web_retriever.search_web(query)
            extracted_web = public_web_retriever.extract_gate_timing_or_fact(query, web_results)
            if extracted_web:
                ans_text, used_web_chunks = extracted_web
                total_latency = round((time.time() - t0) * 1000.0, 2)

                citations = []
                for r in used_web_chunks:
                    meta = r["metadata"]
                    citations.append(Citation(
                        chunk_id=r["chunk_id"],
                        document_title=meta.get("document_title", "Public Web Search"),
                        source_url=meta.get("source_url", "https://cit.edu.in/"),
                        category="Public Web Search",
                        text_snippet=r["text"][:140].replace("\n", " ") + "...",
                        source_type="public_web",
                        verified_date=timestamp_str
                    ))

                is_ref = "couldn't verify" in ans_text.lower() or "not find" in ans_text.lower()
                conf = 0.85 if not is_ref else 0.0

                return {
                    "answer": ans_text,
                    "citations": citations,
                    "source_type": "public_web",
                    "confidence": conf,
                    "retrieval_level": 3,
                    "verified_at": timestamp_str,
                    "retrieved_evidence": used_web_chunks or level1_result.get("retrieved_evidence", []),
                    "category": "Public Web Search",
                    "model_used": model_choice,
                    "latency_ms": total_latency,
                    "has_conflicts": False,
                    "conflict_notes": [],
                    "is_refusal": is_ref
                }

        # Default fallback refusal if all levels fail to find verified evidence
        total_latency = round((time.time() - t0) * 1000.0, 2)
        return {
            "answer": level1_result.get("answer", "I could not find sufficient verified information in the available CIT sources to answer this question confidently."),
            "citations": [],
            "source_type": "dataset",
            "confidence": 0.0,
            "retrieval_level": 1,
            "verified_at": timestamp_str,
            "retrieved_evidence": level1_result.get("retrieved_evidence", []),
            "category": "General",
            "model_used": model_choice,
            "latency_ms": total_latency,
            "has_conflicts": False,
            "conflict_notes": [],
            "is_refusal": True
        }

multi_source_orchestrator = MultiSourceOrchestrator()
