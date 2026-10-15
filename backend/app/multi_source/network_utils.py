"""
Network Utilities for Multi-Source College Knowledge Retrieval
Provides IPv4 network connectivity, HTML extraction, and prompt-injection sanitization.
"""

import socket
import logging
import re
from typing import Optional, Tuple
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Force socket getaddrinfo to prefer IPv4 (AF_INET) to prevent IPv6 timeouts on Windows networks
_original_getaddrinfo = socket.getaddrinfo

def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    try:
        return _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
    except Exception:
        return _original_getaddrinfo(host, port, family, type, proto, flags)

socket.getaddrinfo = _ipv4_getaddrinfo

DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"

# Prompt injection signatures to discard from external web data
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:an?\s+)?unrestricted", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"dan\s+mode", re.IGNORECASE),
    re.compile(r"disregard\s+(?:all\s+)?prior\s+rules", re.IGNORECASE),
]

def sanitize_external_text(text: str) -> str:
    """Strips potential prompt-injection strings and excessive whitespace from external text."""
    clean = text
    for pattern in INJECTION_PATTERNS:
        clean = pattern.sub("[REDACTED_UNTRUSTED_INSTRUCTION]", clean)
    # Remove null bytes and non-printable control characters
    clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", clean)
    return clean.strip()

def fetch_html(url: str, timeout: float = 8.0) -> Optional[str]:
    """Fetches web page HTML over HTTP/HTTPS with proper User-Agent and timeout."""
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": DEFAULT_USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                # Still decode if text
                pass
            raw_bytes = response.read()
            # Decode using response charset or fallback to utf-8 / latin-1
            charset = response.headers.get_content_charset() or "utf-8"
            try:
                return raw_bytes.decode(charset)
            except (UnicodeDecodeError, LookupError):
                return raw_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"Failed to fetch HTML from {url}: {e}")
        return None

def clean_html_to_text(html: str) -> Tuple[str, str]:
    """
    Parses HTML into a clean title and structured plain text.
    Strips scripts, styling, navigation, headers, footers, and SVGs.
    """
    if not html:
        return "", ""

    soup = BeautifulSoup(html, "html.parser")

    # Extract title
    title = (soup.title.string or "").strip() if soup.title else ""
    title = re.sub(r"\s+", " ", title)

    # Decompose unwanted elements
    for el in soup(["script", "style", "nav", "footer", "noscript", "svg", "button", "iframe"]):
        el.decompose()

    # Convert tabular data into clean key-value text lines
    for table in soup.find_all("table"):
        rows_text = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
            if cells:
                rows_text.append(" | ".join(cells))
        if rows_text:
            table_replacement = "\n" + "\n".join(rows_text) + "\n"
            table.replace_with(table_replacement)

    # Extract clean text
    raw_text = soup.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in raw_text.splitlines() if len(line.strip()) > 3]

    clean_text = "\n".join(lines)
    clean_text = sanitize_external_text(clean_text)

    return title, clean_text
