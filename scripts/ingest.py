"""
Data Preprocessing and Structured Document Chunking for CIT Knowledge AI
Handles TXT and PDF documents, extracts clean text, strips boilerplate,
preserves section hierarchy, extracts metadata (source URLs, academic years),
performs token-aware sliding window chunking (400-600 tokens, 50-100 overlap),
and tracks deduplication.
"""

import os
import re
import json
import hashlib
import unicodedata
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

CHUNKS_OUTPUT_PATH = os.path.join(PROCESSED_DIR, "chunks.json")
PREPROCESSING_REPORT_PATH = os.path.join(REPORTS_DIR, "preprocessing_report.json")

DOCUMENT_REGISTRY = {
    "01_college_overview.txt": {
        "title": "College Overview and Autonomous Administration",
        "category": "College Overview",
        "default_url": "https://cit.edu.in/",
        "default_year": "2026"
    },
    "02_departments_and_faculty.txt": {
        "title": "Academic Departments, Faculty Directory and Research Guides",
        "category": "Departments and Faculty",
        "default_url": "https://cit.edu.in/departments",
        "default_year": "2026"
    },
    "03_courses_and_programmes.txt": {
        "title": "Degree Programmes, Curricula and Intake Capacity",
        "category": "Courses and Academic Programmes",
        "default_url": "https://cit.edu.in/admission/courses-offered",
        "default_year": "2026-2027"
    },
    "04_admissions_and_eligibility.txt": {
        "title": "TNEA Engineering Admissions, Cut-offs and Eligibility",
        "category": "Admissions and Eligibility",
        "default_url": "https://cit.edu.in/admission/admission-requirements",
        "default_year": "2026-2027"
    },
    "05_academics_and_regulations.txt": {
        "title": "Autonomous Academic Regulations, CBCS and Exam Rules",
        "category": "Academics and Regulations",
        "default_url": "https://cit.edu.in/syllabus",
        "default_year": "2026"
    },
    "06_placements_and_career.txt": {
        "title": "Placements, Recruiters, Statistics and Career Guidance",
        "category": "Placements and Career Development",
        "default_url": "https://cit.edu.in/placements/placement-cell",
        "default_year": "2025-2026"
    },
    "07_scholarships_fees_and_support.txt": {
        "title": "Scholarships, Tuition Fee Structure, Hostel and Support",
        "category": "Scholarships, Fees and Student Support",
        "default_url": "https://cit.edu.in/admission/tuition-fees",
        "default_year": "2026-2027"
    },
    "08_research_and_innovation.txt": {
        "title": "Research Cell, Centers of Excellence, Patents and MoUs",
        "category": "Research and Innovation",
        "default_url": "https://cit.edu.in/research/research-cell",
        "default_year": "2024-2025"
    },
    "09_campus_facilities_and_faqs.txt": {
        "title": "Campus Facilities, Library, Hostels, Transport and FAQs",
        "category": "Campus Facilities and FAQs",
        "default_url": "https://cit.edu.in/general-facilities/",
        "default_year": "2026"
    },
    "10_official_document_inventory.txt": {
        "title": "Official CIT PDF Document Inventory and Regulatory Archive",
        "category": "Official Document Inventory",
        "default_url": "https://cit.edu.in/uploads/epdf_documents/",
        "default_year": "2026"
    },
    "CIT_Academic_Regulations_and_Examination_Rules.pdf": {
        "title": "CIT Academic Regulations & Exam Rules 2026",
        "category": "Academics and Regulations",
        "default_url": "https://cit.edu.in/syllabus",
        "default_year": "2026"
    },
    "CIT_BTech_AIML_and_CSE_Syllabus_Structure.pdf": {
        "title": "CIT B.E. CSE & B.Tech AI & ML Syllabus Structure",
        "category": "Courses and Academic Programmes",
        "default_url": "https://cit.edu.in/syllabus",
        "default_year": "2026"
    },
    "CIT_Official_Brochure_and_Governance.pdf": {
        "title": "CIT Official Institutional Overview & Autonomous Governance 2026",
        "category": "College Overview",
        "default_url": "https://cit.edu.in/uploads/generalfiles/1770898826_BE_Brochure_2026-compressed.pdf",
        "default_year": "2026"
    },
    "CIT_Placement_Statistics_and_Fee_Structure.pdf": {
        "title": "CIT Career Placements & Fee Structure 2026",
        "category": "Placements and Career Development",
        "default_url": "https://cit.edu.in/placements/placement-records",
        "default_year": "2026"
    }
}

def clean_text(text: str) -> str:
    """Normalize unicode, clean excessive punctuation, remove control chars."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    # Replace weird quotes and dashes
    text = text.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
    text = text.replace("—", " - ").replace("–", " - ")
    # Replace non-breaking space
    text = text.replace("\xa0", " ")
    # Strip carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Normalize multiple blank lines to double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_section_url(text: str, default_url: str) -> str:
    """Find the most relevant URL mentioned in a section/chunk."""
    urls = re.findall(r"https?://[^\s<>\"')]+", text)
    if urls:
        # filter out image/icon urls if any
        valid_urls = [u.rstrip(".,;:") for u in urls if not re.search(r"\.(png|jpg|jpeg|gif|svg)$", u, re.I)]
        if valid_urls:
            return valid_urls[0]
    return default_url

def extract_section_year(text: str, default_year: Optional[str]) -> Optional[str]:
    """Extract academic year reference from text."""
    pattern = r"\b(20[12]\d\s*[-–/]\s*20[12]\d|20[12]\d\s*[-–/]\s*\d{2}|AY\s*20[12]\d[-–/]\d{2,4})\b"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if matches:
        return matches[0].strip()
    return default_year

def split_into_sections(content: str) -> List[Dict[str, str]]:
    """Splits text by markdown/numbered/equals headers while keeping headings."""
    lines = content.split("\n")
    sections = []
    current_heading = "General Overview"
    current_lines = []

    header_pattern = re.compile(
        r"^(?:[=]{4,}|[-]{4,}|[#]{1,4}\s+(.+)|(?:\d{1,2}\.|\b(?:CATEGORY|SECTION)\s+\d{1,2}\b|[A-Z\s]{4,}:)\s*(.+)?)"
    )

    for line in lines:
        stripped = line.strip()
        # Check if line looks like a major separator or section header
        if stripped.startswith("===") or stripped.startswith("---"):
            if current_lines:
                text_block = "\n".join(current_lines).strip()
                if text_block:
                    sections.append({"heading": current_heading, "text": text_block})
                current_lines = []
            continue

        match = header_pattern.match(stripped)
        if match and len(stripped) < 120 and not stripped.endswith("."):
            if current_lines:
                text_block = "\n".join(current_lines).strip()
                if text_block:
                    sections.append({"heading": current_heading, "text": text_block})
                current_lines = []
            current_heading = stripped
        else:
            current_lines.append(line)

    if current_lines:
        text_block = "\n".join(current_lines).strip()
        if text_block:
            sections.append({"heading": current_heading, "text": text_block})

    return sections

def chunk_section(
    section_text: str,
    heading: str,
    target_tokens: int = 450,
    overlap_tokens: int = 60
) -> List[str]:
    """
    Sliding window chunking based on approximate word/token counts.
    1 word ~ 1.33 tokens. Target tokens 450 -> ~340 words. Overlap 60 -> ~45 words.
    """
    target_words = int(target_tokens / 1.33)
    overlap_words = int(overlap_tokens / 1.33)

    paragraphs = [p.strip() for p in section_text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [section_text]

    chunks = []
    current_chunk_words = []
    current_word_count = 0

    for p in paragraphs:
        p_words = p.split()
        if not p_words:
            continue

        # If a single paragraph is larger than target_words, split it by sentences
        if len(p_words) > target_words * 1.5:
            sentences = re.split(r"(?<=[.!?])\s+", p)
            for s in sentences:
                s_words = s.split()
                if current_word_count + len(s_words) > target_words and current_chunk_words:
                    chunk_text = " ".join(current_chunk_words)
                    chunks.append(f"[{heading}]\n{chunk_text}" if not chunk_text.startswith(f"[{heading}]") else chunk_text)
                    # keep overlap
                    overlap = current_chunk_words[-overlap_words:] if len(current_chunk_words) > overlap_words else []
                    current_chunk_words = overlap + s_words
                    current_word_count = len(current_chunk_words)
                else:
                    current_chunk_words.extend(s_words)
                    current_word_count += len(s_words)
        else:
            if current_word_count + len(p_words) > target_words and current_chunk_words:
                chunk_text = " ".join(current_chunk_words)
                chunks.append(f"[{heading}]\n{chunk_text}" if not chunk_text.startswith(f"[{heading}]") else chunk_text)
                overlap = current_chunk_words[-overlap_words:] if len(current_chunk_words) > overlap_words else []
                current_chunk_words = overlap + p_words
                current_word_count = len(current_chunk_words)
            else:
                current_chunk_words.extend(p_words)
                current_word_count += len(p_words)

    if current_chunk_words:
        chunk_text = " ".join(current_chunk_words)
        chunks.append(f"[{heading}]\n{chunk_text}" if not chunk_text.startswith(f"[{heading}]") else chunk_text)

    return chunks

def extract_pdf_pages(file_path: str) -> List[Dict[str, Any]]:
    """Extract page-level text from PDF using PyMuPDF."""
    pages_data = []
    import pymupdf
    doc = pymupdf.open(file_path)
    for idx, page in enumerate(doc, start=1):
        text = page.get_text()
        cleaned = clean_text(text)
        if cleaned:
            pages_data.append({
                "page_number": idx,
                "text": cleaned
            })
    return pages_data

def run_ingestion(
    chunk_size_tokens: int = 450,
    overlap_tokens: int = 60
) -> List[Dict[str, Any]]:
    print("=" * 70)
    print("STARTING DATA PREPROCESSING AND CHUNKING PIPELINE")
    print(f"Config: chunk_size={chunk_size_tokens} tokens, overlap={overlap_tokens} tokens")
    print("=" * 70)

    raw_files = sorted(os.listdir(RAW_DIR))
    all_chunks = []
    seen_hashes = set()
    duplicate_count = 0
    doc_stats = {}

    for fname in raw_files:
        fpath = os.path.join(RAW_DIR, fname)
        ext = os.path.splitext(fname)[1].lower()
        reg_info = DOCUMENT_REGISTRY.get(fname, {
            "title": fname,
            "category": "Unclassified",
            "default_url": "https://cit.edu.in/",
            "default_year": "2026"
        })

        doc_chunks = []

        if ext == ".txt":
            with open(fpath, "r", encoding="utf-8", errors="replace") as fp:
                raw_content = fp.read()
            cleaned_content = clean_text(raw_content)
            sections = split_into_sections(cleaned_content)

            # Combine tiny sections with previous or next to avoid sub-30-word stubs
            combined_sub_chunks = []
            for sec in sections:
                heading = sec["heading"]
                sec_text = sec["text"]
                if len(sec_text.strip()) < 30:
                    continue

                sub_chunks = chunk_section(sec_text, heading, target_tokens=chunk_size_tokens, overlap_tokens=overlap_tokens)
                for sc in sub_chunks:
                    ch_clean = sc.strip()
                    # If this chunk is very short, append to previous if possible
                    if len(ch_clean.split()) < 35 and combined_sub_chunks:
                        prev_sc, prev_hd = combined_sub_chunks[-1]
                        combined_sub_chunks[-1] = (prev_sc + "\n\n" + ch_clean, prev_hd)
                    else:
                        combined_sub_chunks.append((ch_clean, heading))

            for ch_clean, heading in combined_sub_chunks:
                if len(ch_clean.split()) < 20:
                    continue
                ch_hash = hashlib.md5(" ".join(ch_clean.split()).lower().encode("utf-8")).hexdigest()
                if ch_hash in seen_hashes:
                    duplicate_count += 1
                    continue
                seen_hashes.add(ch_hash)

                ch_url = extract_section_url(ch_clean, reg_info["default_url"])
                ch_year = extract_section_year(ch_clean, reg_info.get("default_year"))
                words = len(ch_clean.split())
                approx_tokens = int(words * 1.33)

                chunk_id = f"chunk_{len(all_chunks) + 1:04d}"
                chunk_obj = {
                    "chunk_id": chunk_id,
                    "original_document_id": fname,
                    "document_title": reg_info["title"],
                    "category": reg_info["category"],
                    "source_url": ch_url,
                    "academic_year": ch_year,
                    "page_number": None,
                    "section_heading": heading,
                    "text": ch_clean,
                    "word_count": words,
                    "approx_tokens": approx_tokens,
                    "content_hash": ch_hash
                }
                all_chunks.append(chunk_obj)
                doc_chunks.append(chunk_obj)

        elif ext == ".pdf":
            pages = extract_pdf_pages(fpath)
            for p in pages:
                p_num = p["page_number"]
                p_text = p["text"]
                sub_chunks = chunk_section(p_text, f"Page {p_num}", target_tokens=chunk_size_tokens, overlap_tokens=overlap_tokens)
                for sc in sub_chunks:
                    ch_clean = sc.strip()
                    ch_hash = hashlib.md5(" ".join(ch_clean.split()).lower().encode("utf-8")).hexdigest()
                    if ch_hash in seen_hashes:
                        duplicate_count += 1
                        continue
                    seen_hashes.add(ch_hash)

                    ch_url = extract_section_url(ch_clean, reg_info["default_url"])
                    ch_year = extract_section_year(ch_clean, reg_info.get("default_year"))
                    words = len(ch_clean.split())
                    approx_tokens = int(words * 1.33)

                    chunk_id = f"chunk_{len(all_chunks) + 1:04d}"
                    chunk_obj = {
                        "chunk_id": chunk_id,
                        "original_document_id": fname,
                        "document_title": reg_info["title"],
                        "category": reg_info["category"],
                        "source_url": ch_url,
                        "academic_year": ch_year,
                        "page_number": p_num,
                        "section_heading": f"Official Document Page {p_num}",
                        "text": ch_clean,
                        "word_count": words,
                        "approx_tokens": approx_tokens,
                        "content_hash": ch_hash
                    }
                    all_chunks.append(chunk_obj)
                    doc_chunks.append(chunk_obj)

        doc_stats[fname] = {
            "title": reg_info["title"],
            "category": reg_info["category"],
            "chunks_generated": len(doc_chunks)
        }
        print(f"Processed {fname:<45} -> {len(doc_chunks):>3} chunks")

    # Save to chunks.json
    with open(CHUNKS_OUTPUT_PATH, "w", encoding="utf-8") as fp:
        json.dump(all_chunks, fp, indent=2)

    total_tokens = sum(c["approx_tokens"] for c in all_chunks)
    avg_tokens = round(total_tokens / len(all_chunks), 1) if all_chunks else 0

    report = {
        "total_source_documents": len(raw_files),
        "total_chunks_created": len(all_chunks),
        "duplicates_filtered": duplicate_count,
        "total_approx_tokens": total_tokens,
        "avg_tokens_per_chunk": avg_tokens,
        "chunk_size_target": chunk_size_tokens,
        "overlap_tokens_target": overlap_tokens,
        "document_breakdown": doc_stats
    }

    with open(PREPROCESSING_REPORT_PATH, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    print("=" * 70)
    print(f"PREPROCESSING COMPLETE: {len(all_chunks)} unique chunks generated.")
    print(f"Total tokens: {total_tokens:,} | Average per chunk: {avg_tokens} tokens")
    print(f"Saved chunks to: {CHUNKS_OUTPUT_PATH}")
    print(f"Saved preprocessing report to: {PREPROCESSING_REPORT_PATH}")
    print("=" * 70)

    return all_chunks

if __name__ == "__main__":
    run_ingestion()
