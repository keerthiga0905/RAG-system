"""
Multi-Source Knowledge Retrieval Package for CIT Knowledge AI
Provides:
Level 1: Local Dataset Hybrid Search
Level 2: Official CIT Website Search (https://cit.edu.in/)
Level 3: Public Web Search & Campus Practical Info
"""

from .network_utils import fetch_html, clean_html_to_text, sanitize_external_text
from .cache_manager import cache_manager
from .cit_website_retriever import cit_website_retriever
from .public_web_retriever import public_web_retriever
from .orchestrator import multi_source_orchestrator
