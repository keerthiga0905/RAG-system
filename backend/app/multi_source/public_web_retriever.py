"""
Level 3: Public Web Retriever
Searches Google / Bing / Public Web using IPv4 network requests and extracts verified answers for student queries
that are not available in the local dataset or on the official CIT website.
"""

import os
import re
import time
import base64
import logging
import urllib.request
import urllib.parse
from typing import List, Dict, Any, Optional, Tuple
from bs4 import BeautifulSoup

from ..config import settings
from .network_utils import DEFAULT_USER_AGENT, sanitize_external_text
from .cache_manager import cache_manager

logger = logging.getLogger(__name__)

class PublicWebRetriever:
    def __init__(self):
        self.search_url = "https://www.bing.com/search"

    def _decode_bing_url(self, raw_url: str) -> str:
        """Unmasks base64-encoded destination URLs from Bing search redirects."""
        if not raw_url:
            return ""
        if "/ck/a?!" in raw_url and "u=a1" in raw_url:
            try:
                m = re.search(r"[?&]u=a1([a-zA-Z0-9_\-]+)", raw_url)
                if m:
                    b64 = m.group(1).replace("-", "+").replace("_", "/")
                    b64 += "=" * ((4 - len(b64) % 4) % 4)
                    decoded = base64.b64decode(b64).decode("utf-8", errors="ignore")
                    if decoded.startswith("http"):
                        return decoded
            except Exception:
                pass
        return raw_url

    def search_web(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Executes a targeted public web search for Coimbatore Institute of Technology.
        Returns parsed search items with title, url, snippet.
        """
        # Check cache first
        cache_key = f"websearch_{query.strip().lower()}"
        cached = cache_manager.get(cache_key)
        if cached:
            return cached

        # Formulate query explicitly specifying CIT Coimbatore
        q_clean = query.strip()
        if "coimbatore institute of technology" not in q_clean.lower() and "cit coimbatore" not in q_clean.lower():
            search_query = f"Coimbatore Institute of Technology CIT Coimbatore {q_clean}"
        else:
            search_query = q_clean

        encoded_q = urllib.parse.quote(search_query)
        target_url = f"{self.search_url}?q={encoded_q}&count={max_results + 3}"

        results = []
        try:
            req = urllib.request.Request(
                target_url,
                headers={
                    "User-Agent": DEFAULT_USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                }
            )
            with urllib.request.urlopen(req, timeout=settings.WEB_TIMEOUT_SECONDS) as resp:
                soup = BeautifulSoup(resp.read(), "html.parser")
                for li in soup.find_all("li", class_="b_algo"):
                    h2 = li.find("h2")
                    p = li.find("p")
                    a = h2.find("a") if h2 else None
                    if h2 and a:
                        raw_href = a.get("href", "")
                        clean_url = self._decode_bing_url(raw_href)
                        title = h2.get_text(strip=True)
                        snippet = p.get_text(strip=True) if p else ""
                        snippet = sanitize_external_text(snippet)

                        # Filter out non-Coimbatore CIT colleges (e.g. Cambridge Bangalore, Chennai IT)
                        lower_text = (title + " " + snippet).lower()
                        if "cambridge institute" in lower_text or "chennai institute of technology" in lower_text:
                            continue

                        results.append({
                            "title": title,
                            "url": clean_url,
                            "snippet": snippet
                        })
                        if len(results) >= max_results:
                            break
        except Exception as e:
            logger.warning(f"Public web search failed for '{query}': {e}")

        # Cache results for 24h
        if results:
            cache_manager.set(cache_key, results, ttl_seconds=settings.WEB_CACHE_TTL_HOURS * 3600)
        return results

    def extract_gate_timing_or_fact(
        self,
        query: str,
        search_results: List[Dict[str, str]]
    ) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
        """
        Specialized extractor for practical campus questions (e.g. gate closing times, timings, rules).
        Returns (answer_string, [supporting_chunk]) or None.
        """
        q_lower = query.lower()

        # CASE: Gate closing time / campus timing
        if "gate" in q_lower or "closing time" in q_lower or "curfew" in q_lower:
            # Check snippets for specific gate timings
            for item in search_results:
                snippet = item["snippet"]
                # Match patterns like: 8:30 PM, 9:00 PM, 6:30 PM, 7:00 PM
                time_match = re.search(r"(\b\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)\b)", snippet, re.IGNORECASE)
                if time_match and ("gate" in snippet.lower() or "hostel" in snippet.lower() or "entry" in snippet.lower()):
                    timing = time_match.group(1).upper()
                    supporting_chunk = {
                        "chunk_id": f"web_{int(time.time())}",
                        "text": snippet,
                        "score": 0.82,
                        "metadata": {
                            "document_title": item["title"],
                            "source_url": item["url"],
                            "category": "Campus Regulations (Public Web)",
                            "source_type": "public_web"
                        }
                    }
                    ans = f"CIT campus gate closing time: {timing} (based on student hostel and campus regulations). Source: {item['url']}."
                    return ans, [supporting_chunk]

            # If no reliable exact time found:
            return "I couldn't verify CIT's exact gate closing time from reliable sources.", []

        # CASE: Department / HOD lookup from public web
        if "hod" in q_lower or "head of" in q_lower:
            dept_m = re.search(r"(?:hod of|head of (?:the )?department of|head of)\s+([a-zA-Z\s&]+?)(?:\s*\?|\s*$)", q_lower)
            dept_name = dept_m.group(1).strip() if dept_m else "the requested department"
            # Require the snippet to mention the department and an academic faculty title
            for item in search_results:
                snippet = item["snippet"]
                if dept_name.lower() in snippet.lower() and re.search(r"(?:Dr\.|Prof\.|HOD|Head of Department)\b", snippet, re.IGNORECASE):
                    m = re.search(r"(?:Dr\.?|Prof\.?)\s+[A-Z][a-zA-Z\.\s]+", snippet)
                    if m:
                        name = m.group(0).strip()
                        supporting_chunk = {
                            "chunk_id": f"web_{int(time.time())}",
                            "text": snippet,
                            "score": 0.82,
                            "metadata": {
                                "document_title": item["title"],
                                "source_url": item["url"],
                                "category": "Public Web Search",
                                "source_type": "public_web"
                            }
                        }
                        return f"The Head of the Department of {dept_name.title()} is {name}. Source: {item['url']}.", [supporting_chunk]
            return f"I couldn't verify the Head of Department for {dept_name.title()} from reliable official CIT sources.", []

        # Generic factual extraction: require key query keyword overlap
        q_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", q_lower)) - {
            "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
            "the", "and", "cit", "for", "are", "is", "coimbatore"
        }
        for item in search_results:
            snippet = item["snippet"]
            s_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", snippet.lower()))
            overlap = len(q_words & s_words)
            if q_words and overlap >= max(2, int(len(q_words) * 0.5)):
                # Filter out generic city / district overviews
                if any(w in snippet.lower() for w in ["metropolitan city", "district in tamil nadu", "second largest city", "tier-ii city"]):
                    continue
                supporting_chunk = {
                    "chunk_id": f"web_{int(time.time())}",
                    "text": snippet,
                    "score": 0.80,
                    "metadata": {
                        "document_title": item["title"],
                        "source_url": item["url"],
                        "category": "Public Web Search",
                        "source_type": "public_web"
                    }
                }
                ans = f"{snippet.split('.')[0].strip()}."
                return ans, [supporting_chunk]

        return None

public_web_retriever = PublicWebRetriever()
