# CIT Official Dataset Inventory & Data Quality Audit

**Audit Timestamp:** 2026-10-15  
**Total Files Discovered:** 14 (10 Primary Datasets + 4 Official Supplementary PDFs)  
**Total Word Count:** 41,758 words (~55,532 tokens)  
**Total Distinct Source URLs Extracted:** 423  

## 1. Discovered Datasets & Physical Metadata

| # | Filename | Format | Encoding | Size (Bytes) | Category | Words | Approx Tokens | URLs |
|---|----------|--------|----------|--------------|----------|-------|---------------|------|
| 1 | `01_college_overview.txt` | TXT | utf-8 | 30,241 | College Overview | 3,545 | 4,714 | 42 |
| 2 | `02_departments_and_faculty.txt` | TXT | utf-8 | 23,985 | Departments and Faculty | 2,745 | 3,650 | 69 |
| 3 | `03_courses_and_programmes.txt` | TXT | utf-8 | 51,797 | Courses and Academic Programmes | 6,302 | 8,381 | 85 |
| 4 | `04_admissions_and_eligibility.txt` | TXT | utf-8 | 53,024 | Admissions and Eligibility | 6,629 | 8,816 | 44 |
| 5 | `05_academics_and_regulations.txt` | TXT | utf-8 | 34,316 | Academics and Regulations | 3,581 | 4,762 | 63 |
| 6 | `06_placements_and_career.txt` | TXT | utf-8 | 31,125 | Placements and Career Development | 3,861 | 5,135 | 13 |
| 7 | `07_scholarships_fees_and_support.txt` | TXT | utf-8 | 18,714 | Scholarships, Fees and Student Support | 2,650 | 3,524 | 11 |
| 8 | `08_research_and_innovation.txt` | TXT | utf-8 | 45,770 | Research and Innovation | 5,553 | 7,385 | 17 |
| 9 | `09_campus_facilities_and_faqs.txt` | TXT | utf-8 | 22,123 | Campus Facilities and FAQs | 2,704 | 3,596 | 22 |
| 10 | `10_official_document_inventory.txt` | TXT | utf-8 | 34,178 | Official Document Inventory | 3,758 | 4,998 | 57 |
| 11 | `CIT_Academic_Regulations_and_Examination_Rules.pdf` | PDF | PDF Binary (Machine-readable) | 2,979 | Academics and Regulations (Official PDF) | 113 | 150 | 0 |
| 12 | `CIT_BTech_AIML_and_CSE_Syllabus_Structure.pdf` | PDF | PDF Binary (Machine-readable) | 3,161 | Courses and Academic Programmes (Official PDF) | 91 | 121 | 0 |
| 13 | `CIT_Official_Brochure_and_Governance.pdf` | PDF | PDF Binary (Machine-readable) | 3,671 | College Overview (Official PDF) | 122 | 162 | 0 |
| 14 | `CIT_Placement_Statistics_and_Fee_Structure.pdf` | PDF | PDF Binary (Machine-readable) | 3,144 | Placements and Career Development (Official PDF) | 104 | 138 | 0 |

## 2. Identified Official Data Conflicts & Discrepancies

The following official discrepancies were identified during cross-dataset verification and are tracked systematically in the RAG retrieval pipeline:

- **CONF-001: Autonomous Status Conferred Date**
  - Claim A: 01/06/1987 (Stated in AQAR 2022-2023 report)
  - Claim B: 08/07/1987 (Stated in AQAR 2023-2024 report)
  - System RAG Policy: *Preserve both claims; explicitly cite reporting year and document source.*
- **CONF-002: Placed Students Count for 2022-2023**
  - Claim A: 757 placed students (Stated in 2022-2023 Summary table)
  - Claim B: 782 placed students (Stated in 6-year historical summary table)
  - System RAG Policy: *Retain both figures noting differing denominator and table contexts.*
- **CONF-003: Principal Contact Channels**
  - Claim A: 94868 37757 / principal.citoffice@cit.edu.in (Current Contact Us page)
  - Claim B: +91-422-2573187 / principal@cit.edu.in (Official Student Information page)
  - System RAG Policy: *Present both valid published channels with context.*
- **CONF-004: B.E. VLSI Design & Technology Approval Status**
  - Claim A: Programme listed as active since 2026 in brochure
  - Claim B: Marked with asterisk (* awaiting AICTE/University approval)
  - System RAG Policy: *Flag programme approval status explicitly to user queries.*

## 3. Data Quality & Limitations Observations
- **Duplicate Content**: Boilerplate headings and recurring navigation headers exist across several scraped files, but detailed paragraphs contain distinct tables and domain information.
- **Honest Handling of Missing Information**: Several files (e.g., `09_campus_facilities_and_faqs.txt` and `06_placements_and_career.txt`) explicitly note that specific granular details (e.g., exact daily bus timings, complete current all-recruiter lists) are not published on the official site. The RAG system must never hallucinate these and must fall back appropriately.
- **Time-Bound Policies**: Fee structures, hostel tariffs, and cut-offs apply to specific academic years (2021-22, 2023-24, 2025-26, 2026-27). All chunked evidence retains academic year metadata.
