"""
Level 2: Official CIT Website Retriever
Crawls, parses, caches, and extracts verified information directly from official CIT website pages (https://cit.edu.in/).
"""

import os
import re
import time
import logging
from typing import List, Dict, Any, Optional, Tuple

from ..config import settings
from .network_utils import fetch_html, clean_html_to_text
from .cache_manager import cache_manager

logger = logging.getLogger(__name__)

# Map canonical department names to their official CIT website slugs
CIT_DEPARTMENT_URLS = {
    "M.Sc. Software Systems": {
        "home": "https://cit.edu.in/msc-software-systems",
        "faculty": "https://cit.edu.in/msc-software-systems/faculty",
        "courses": "https://cit.edu.in/msc-software-systems/course-offered"
    },
    "Civil Engineering": {
        "home": "https://cit.edu.in/civil-engineering",
        "faculty": "https://cit.edu.in/civil-engineering/faculty",
        "courses": "https://cit.edu.in/civil-engineering/course-offered"
    },
    "Computer Science & Engineering": {
        "home": "https://cit.edu.in/computer-science-engineering",
        "faculty": "https://cit.edu.in/computer-science-engineering/faculty",
        "courses": "https://cit.edu.in/computer-science-engineering/course-offered"
    },
    "Information Technology": {
        "home": "https://cit.edu.in/information-technology",
        "faculty": "https://cit.edu.in/information-technology/faculty",
        "courses": "https://cit.edu.in/information-technology/course-offered"
    },
    "Artificial Intelligence and Data Science": {
        "home": "https://cit.edu.in/artificial-intelligence-and-data-science",
        "faculty": "https://cit.edu.in/artificial-intelligence-and-data-science/faculty",
        "courses": "https://cit.edu.in/artificial-intelligence-and-data-science/course-offered"
    },
    "M.Sc. Data Science": {
        "home": "https://cit.edu.in/msc-data-science",
        "faculty": "https://cit.edu.in/msc-data-science/faculty",
        "courses": "https://cit.edu.in/msc-data-science/course-offered"
    },
    "M.Sc. Decision and Computing Sciences": {
        "home": "https://cit.edu.in/msc-decision-and-computing-sciences",
        "faculty": "https://cit.edu.in/msc-decision-and-computing-sciences/faculty",
        "courses": "https://cit.edu.in/msc-decision-and-computing-sciences/course-offered"
    },
    "M.Sc. Artificial Intelligence and Machine Learning": {
        "home": "https://cit.edu.in/msc-artificial-intelligence-and-machine-learning",
        "faculty": "https://cit.edu.in/msc-artificial-intelligence-and-machine-learning/faculty",
        "courses": "https://cit.edu.in/msc-artificial-intelligence-and-machine-learning/course-offered"
    },
    "Mechanical Engineering": {
        "home": "https://cit.edu.in/mechanical-engineering",
        "faculty": "https://cit.edu.in/mechanical-engineering/faculty",
        "courses": "https://cit.edu.in/mechanical-engineering/course-offered"
    },
    "Electrical & Electronics Engineering": {
        "home": "https://cit.edu.in/electrical-electronics-engineering",
        "faculty": "https://cit.edu.in/electrical-electronics-engineering/faculty",
        "courses": "https://cit.edu.in/electrical-electronics-engineering/course-offered"
    },
    "Electronics & Communication Engineering": {
        "home": "https://cit.edu.in/electronics-communication-engineering",
        "faculty": "https://cit.edu.in/electronics-communication-engineering/faculty",
        "courses": "https://cit.edu.in/electronics-communication-engineering/course-offered"
    },
    "Chemical Engineering": {
        "home": "https://cit.edu.in/chemical-engineering",
        "faculty": "https://cit.edu.in/chemical-engineering/faculty",
        "courses": "https://cit.edu.in/chemical-engineering/course-offered"
    }
}

CIT_FACILITY_URLS = {
    "hostel": "https://cit.edu.in/hostel",
    "transport": "https://cit.edu.in/transport",
    "library": "https://cit.edu.in/library/about",
    "sports": "https://cit.edu.in/physical-education/about",
    "placement": "https://cit.edu.in/placements",
    "admissions": "https://cit.edu.in/admission/admissions-details",
    "departments": "https://cit.edu.in/departments"
}

class CITWebsiteRetriever:
    def __init__(self):
        self.base_url = settings.OFFICIAL_CIT_BASE_URL

    def fetch_and_parse_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetches and parses an official CIT URL, utilizing caching."""
        cached = cache_manager.get(url)
        if cached:
            return cached

        html = fetch_html(url, timeout=settings.WEB_TIMEOUT_SECONDS)
        if not html:
            return None

        title, text = clean_html_to_text(html)
        if not text or len(text) < 30:
            return None

        result = {
            "url": url,
            "title": title or "Coimbatore Institute of Technology",
            "text": text,
            "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Cache with 24-hour TTL
        cache_manager.set(url, result, ttl_seconds=settings.WEB_CACHE_TTL_HOURS * 3600)
        return result

    def search_official_website(
        self,
        query: str,
        intent_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Executes targeted official website search based on user question intent and target entity.
        Returns candidate evidence passages with official source URLs.
        """
        target_entity = intent_info.get("target_entity")
        attribute = intent_info.get("attribute")
        candidate_urls = []

        # 1. Department attribute lookups (HOD, faculty, courses)
        if target_entity and target_entity in CIT_DEPARTMENT_URLS:
            dept_map = CIT_DEPARTMENT_URLS[target_entity]
            if attribute == "hod" or attribute == "faculty":
                candidate_urls.append(dept_map["faculty"])
                candidate_urls.append(dept_map["home"])
            elif attribute == "courses":
                candidate_urls.append(dept_map["courses"])
                candidate_urls.append(dept_map["home"])
            else:
                candidate_urls.append(dept_map["home"])
                candidate_urls.append(dept_map["faculty"])

        # 2. General facility / administrative lookups
        q_lower = query.lower()
        if "hostel" in q_lower or "mess" in q_lower:
            candidate_urls.append(CIT_FACILITY_URLS["hostel"])
        if "transport" in q_lower or "bus" in q_lower:
            candidate_urls.append(CIT_FACILITY_URLS["transport"])
        if "library" in q_lower or "book" in q_lower:
            candidate_urls.append(CIT_FACILITY_URLS["library"])
        if "placement" in q_lower or "recruiter" in q_lower:
            candidate_urls.append(CIT_FACILITY_URLS["placement"])
        if "admission" in q_lower or "counselling" in q_lower:
            candidate_urls.append(CIT_FACILITY_URLS["admissions"])

        # Deduplicate URLs
        candidate_urls = list(dict.fromkeys(candidate_urls))

        evidence_passages = []
        for url in candidate_urls[:3]:
            parsed = self.fetch_and_parse_url(url)
            if not parsed:
                continue

            text = parsed["text"]
            # Create passage chunk
            chunk_data = {
                "chunk_id": f"web_{int(time.time())}_{abs(hash(url)) % 10000}",
                "text": text[:1800],  # Concise top body
                "score": 0.88,
                "metadata": {
                    "document_title": parsed["title"],
                    "source_url": url,
                    "category": "Official CIT Website",
                    "source_type": "official_cit_website",
                    "verified_date": parsed["fetched_at"]
                }
            }
            evidence_passages.append(chunk_data)

        return evidence_passages

    def extract_verified_answer(
        self,
        query: str,
        evidence: List[Dict[str, Any]],
        intent_info: Dict[str, Any]
    ) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
        """
        Inspects official website evidence to extract a direct answer.
        Returns (answer_string, [supporting_chunk]) if verified, or None if evidence is insufficient.
        """
        if not evidence:
            return None

        attribute = intent_info.get("attribute")
        target_entity = intent_info.get("target_entity")

        # CASE: HOD extraction from official website
        if attribute == "hod":
            dept_name = target_entity or "the department"
            year_ctx = intent_info.get("year_context")
            # If the user asked for a specific historical year (e.g. "in 2020"),
            # the current website only reflects current faculty and does not verify 2020
            if year_ctx and year_ctx not in ["2026", "2027"]:
                return f"I couldn't verify the Head of the Department of {dept_name} for {year_ctx} from available official records.", []

            for ev in evidence:
                text = ev["text"]

                # Check for explicit HOD patterns on live CIT website
                # Pattern: Dr. S. Manjula Gandhi ... Associate Professor & HOD(I/c)
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                for idx, line in enumerate(lines):
                    # Check if line indicates HOD or HOD(I/c)
                    if re.search(r"\b(?:HOD|Head of (?:the )?Department)\b", line, re.IGNORECASE):
                        # The name is often on the preceding line or current line
                        candidate_name = None
                        role_desc = line
                        if idx > 0 and re.search(r"^(?:Dr\.?|Prof\.?|Mr\.?|Ms\.?)\s+[A-Z]", lines[idx - 1]):
                            candidate_name = lines[idx - 1]
                        elif re.search(r"(?:Dr\.?|Prof\.?|Mr\.?|Ms\.?)\s+[A-Z][a-zA-Z\.\s]+", line):
                            m = re.search(r"(?:Dr\.?|Prof\.?|Mr\.?|Ms\.?)\s+[A-Z][a-zA-Z\.\s]+", line)
                            candidate_name = m.group(0).strip() if m else None

                        if candidate_name:
                            # Clean role description
                            role = "Head of the Department"
                            if "(I/c)" in role_desc or "In-charge" in role_desc or "i/c" in role_desc.lower():
                                role = "Head of the Department (In-charge)"
                            ans = f"The {role} of {dept_name} is {candidate_name}."
                            return ans, [ev]

                # Fallback regex search in entire text
                m_hod = re.search(r"(Dr\.?\s+[A-Z][a-zA-Z\.\s]+)[\s\n]+(?:Associate\s+Professor\s*&\s*)?HOD\s*(?:\(I/c\))?", text, re.IGNORECASE)
                if m_hod:
                    name = m_hod.group(1).strip()
                    ans = f"The Head of the Department (In-charge) of {dept_name} is {name}."
                    return ans, [ev]

        return None

cit_website_retriever = CITWebsiteRetriever()
