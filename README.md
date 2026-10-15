# CIT Knowledge AI — Complete Local RAG System with Embedding Fine-Tuning

A complete, locally runnable College Knowledge AI and Retrieval-Augmented Generation (RAG) system grounded in official Coimbatore Institute of Technology (CIT) website datasets and regulatory publications. 

Developed as an academic Machine Learning Laboratory project featuring **actual embedding model fine-tuning**, leakage-resistant dataset partitioning, ChromaDB vector indexing, BM25 lexical hybrid fusion, strict hallucination suppression, and an academic web interface.

---

## 1. Project Objectives
1. **Official Information Grounding**: Answer complex student queries on admissions, eligibility cut-offs, fee structures, academic regulations, examination grading, placements, scholarships, and research facilities strictly using verified CIT records.
2. **Elimination of Hallucination**: Enforce an honest refusal fallback (*"I could not find sufficient verified information in the available CIT sources to answer this question confidently."*) on out-of-scope or unverified inquiries.
3. **Transparent Evidence & Citations**: Generate clickable citations linked to official URLs and highlight time-bound applicable academic years.
4. **Reproducible ML Experiments**: Conduct two distinct retrieval experiments comparing **Baseline Pretrained Embeddings** vs. **Domain Fine-Tuned Embeddings** vs. **Hybrid (Dense + Sparse BM25)** retrieval on a held-out test set.

---

## 2. System Architecture

```
User Query (Browser / API)
        │
        ├──► BM25 Lexical Keyword Search (vector_store/bm25_corpus.json)
        │           │
        ├──► Dense Vector Search (ChromaDB Collection: cit_finetuned)
        │           │
        ▼           ▼
   Reciprocal Rank Fusion (RRF: k=60)
        │
        ▼
   Conflict Detector & Relevance Gate
   (Checks official discrepancies & similarity threshold)
        │
        ▼
   Context-Grounded Synthesizer & Citation Builder
        │
        ▼
   Response + Clickable Citations + Evidence Sidebar
```

---

## 3. Why `sentence-transformers/all-MiniLM-L6-v2`?
- **Efficiency & Compactness**: 22.7M parameters with a 384-dimensional embedding space, running at ultra-low latency (<10ms per query on CPU) without requiring a discrete GPU.
- **Strong Baseline Retrieval**: Trained on >1 billion sentence pairs via contrastive learning, providing robust semantic sentence representations.
- **Domain Adaptability**: Easily fine-tunable on institutional question-passage pairs using `MultipleNegativesRankingLoss` to adapt vector space to college administrative terminology.

---

## 4. Pretraining vs. Fine-Tuning vs. RAG

| Dimension | Pretraining | Embedding Fine-Tuning (Exp B) | Retrieval-Augmented Generation (RAG) |
|---|---|---|---|
| **Goal** | Learn general linguistic syntax and broad world representations. | Align student question embeddings with institutional CIT passages. | Retrieve verified official text at runtime to eliminate hallucination. |
| **Data Scope** | Trillions of web tokens (CommonCrawl, Wikipedia). | Structured `(Question, Positive Passage, Hard Negative)` CIT triples. | 290 chunked segments from 14 verified CIT website documents. |
| **Hardware** | Massive GPU clusters (weeks/months). | Standard multi-core CPU (~6 minutes) or single GPU. | Lightweight local CPU execution with ChromaDB & BM25. |
| **Hallucination** | High parametric memory drift. | Low (affects geometric similarity space only). | Zero/Extremely Low (answers strictly from retrieved context). |

---

## 5. Dataset Inventory and Preprocessing

All 10 primary datasets and 4 supplementary official PDFs are preserved in `data/raw/`:
1. `01_college_overview.txt`: History, founder, autonomous status, administration.
2. `02_departments_and_faculty.txt`: Department heads, faculty lists, research guides.
3. `03_courses_and_programmes.txt`: B.E./B.Tech, M.E., 5-year Integrated M.Sc., MBA, MCA intake.
4. `04_admissions_and_eligibility.txt`: TNEA counseling cutoffs, eligibility criteria, lateral entry.
5. `05_academics_and_regulations.txt`: Autonomous regulations, 75% attendance rule, grading.
6. `06_placements_and_career.txt`: Placement statistics, highest CTC (59 LPA), recruiter lists.
7. `07_scholarships_fees_and_support.txt`: 2026-27 tuition fee structure, hostel tariffs.
8. `08_research_and_innovation.txt`: FESEM facility, AIMMRG, MoUs, patents, funded projects.
9. `09_campus_facilities_and_faqs.txt`: Library hours, sports, barrier-free accessibility.
10. `10_official_document_inventory.txt`: Official PDF metadata, URLs, and regulatory archive.
11. `CIT_Academic_Regulations_and_Examination_Rules.pdf`: Exam rules & evaluation criteria.
12. `CIT_BTech_AIML_and_CSE_Syllabus_Structure.pdf`: Curriculum structure.
13. `CIT_Official_Brochure_and_Governance.pdf`: Institutional brochure & autonomous governance.
14. `CIT_Placement_Statistics_and_Fee_Structure.pdf`: Fee structure & placement highlights.

### Preprocessing Pipeline (`scripts/ingest.py`)
- Unicode normalization (NFKC) and smart quote cleanup.
- Recursive paragraph sliding-window chunking (target: 450 tokens, overlap: 60 tokens).
- Deduplication via MD5 content hashing.
- Produced **290 rich chunks** serialized to `data/processed/chunks.json`.

---

## 6. Leakage-Resistant Dataset Partitioning

To ensure scientific validity, the corpus was partitioned **by Document ID and Topic Group** before generating training and evaluation queries:
- **Training Documents (117 samples)**: Overview (01), Faculty (02), Placements (06), Fees/Scholarships (07), Facilities (09).
- **Validation Documents (13 samples)**: Regulations (05) and Document Inventory (10). Used for early stopping.
- **Held-Out Test Documents (23 samples)**: Courses (03), Admissions (04), Research (08), and unanswerable adversarial queries.

Zero document overlap exists between the training set and the held-out test set.

---

## 7. Model Training & Fine-Tuning Procedure (`scripts/train_embeddings.py`)
- **Loss Objective**: `MultipleNegativesRankingLoss` with `(Anchor Question, Positive Passage, Hard Negative)` triples.
- **Optimizer**: AdamW with linear warmup (10% warmup steps).
- **Hyperparameters**: Epochs = 4, Batch Size = 8, Learning Rate = 2e-5, Seed = 42.
- **Training Convergence**:
  - Initial Loss: 2.148  
  - Final Loss: **0.5253**  
  - Validation MRR: **0.4231 ➔ 0.5192** (+22.7% relative improvement)  
  - Validation Recall@1: **30.8% ➔ 46.2%** (+50.0% relative improvement)  
- Checkpoint saved to `models/finetuned/cit-minilm-l6-v2`.

---

## 8. Empirical Evaluation Results (Held-Out Test Set)

Evaluated across 23 held-out test queries using `scripts/evaluate_retrieval.py`:

| Retrieval Strategy | Recall@1 | Recall@3 | Recall@5 | Precision@1 | Precision@3 | MRR | Citation Accuracy | Out-of-Scope Refusal | Latency |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline Dense (`all-MiniLM-L6-v2`)** | 23.8% | 28.6% | 38.1% | 23.8% | 9.5% | 0.2833 | 100.0% | 100.0% | 12.4 ms |
| **Fine-Tuned CIT Dense (`cit-minilm-l6-v2`)** | 19.0% | 33.3% | 42.9% | 19.0% | 11.1% | 0.2754 | 100.0% | 100.0% | 12.8 ms |
| **Hybrid Retrieval (Dense + BM25 RRF)** | **23.8%** | **47.6%** | **47.6%** | **23.8%** | **15.9%** | **0.3492** | **100.0%** | **100.0%** | **13.5 ms** |

### Key ML Findings
1. **Acronym Sensitivity**: Academic college datasets frequently involve dense administrative acronyms (*TNEA, FESEM, MPC, AIMMRG, ARPIT*). Pure semantic embeddings occasionally suffer from semantic dispersion on rare acronyms.
2. **Hybrid Superiority**: Combining fine-tuned dense embeddings with lexical BM25 using Reciprocal Rank Fusion yielded the highest MRR (**0.3492**) and increased Recall@3 to **47.6%** (a **+66.4% relative gain** over baseline dense search).
3. **Hallucination Prevention**: On adversarial questions (e.g., canteen menu prices, international astronaut scholarships), the system achieved **100% honest refusal compliance**.

---

## 9. Official Conflicting Claims Governance

The system tracks documented discrepancies in official CIT records:
- **CONF-001 (Autonomous Date)**: AQAR 2022-2023 reports `01/06/1987`; AQAR 2023-2024 reports `08/07/1987`. System cites both official years.
- **CONF-002 (2022-23 Placed Students)**: Summary table states `757`; 6-year trend table states `782` due to differing backlogs/interested student denominators.
- **CONF-003 (Principal Contacts)**: Contact Us lists `94868 37757 / principal.citoffice@cit.edu.in`; Student Info lists `+91-422-2573187 / principal@cit.edu.in`.
- **CONF-004 (B.E. VLSI Status)**: Brochure lists programme as active since 2026 marked `* awaiting approval`. System explicitly flags approval status.

---

## 10. Installation & Execution Guide

### Prerequisites
- Python 3.12 (64-bit)
- Node.js v18+ and npm v9+

### Step 1: Virtual Environment Setup
```bash
# Clone or navigate to the project directory
cd "c:\Users\KEERTHIGA M\Desktop\ML-project"

# Activate the existing virtual environment or create one
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Reproduce Data Preprocessing, Training, and Evaluation
```bash
# 1. Inspect raw datasets and generate inventory report
python scripts/inspect_datasets.py

# 2. Ingest documents and generate 290 chunks
python scripts/ingest.py

# 3. Create leakage-resistant train, val, and test splits
python scripts/prepare_training_data.py

# 4. Fine-tune sentence-transformer embedding model
python scripts/train_embeddings.py

# 5. Build persistent ChromaDB vector collections & BM25 index
python scripts/build_index.py

# 6. Run benchmark evaluation on held-out test set
python scripts/evaluate_retrieval.py
```

### Step 3: Run Automated Test Suite
```bash
pytest backend/tests/ -v
# 15 passed in ~8 seconds
```

### Step 4: Start Backend and Frontend Servers
```bash
# Terminal 1: Start FastAPI backend (port 8000)
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

# Terminal 2: Start React + Vite frontend (port 5173)
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

Open browser at `http://127.0.0.1:5173/`.

---

## 11. Project Directory Structure

```
ML-project/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py         # Configuration & environment variables
│   │   ├── main.py           # FastAPI application endpoints
│   │   ├── rag_engine.py     # Hybrid search, conflict logic, grounded synthesis
│   │   └── schemas.py        # Pydantic request/response models
│   └── tests/
│       ├── test_api.py       # API integration tests
│       └── test_rag.py       # Retrieval, refusal & citation unit tests
├── data/
│   ├── raw/                  # 10 raw CIT text datasets + 4 supplementary PDFs
│   ├── processed/            # chunks.json, train.json, val.json, test.json
│   └── feedback_logs.json    # User feedback logs
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatInterface.tsx        # Chat UI, citations, evidence sidebar
│   │   │   ├── DatasetInventory.tsx     # 14 official documents table
│   │   │   ├── EvaluationDashboard.tsx  # ML metrics matrix & category scores
│   │   │   ├── Header.tsx               # Academic navigation bar
│   │   │   └── ModelInfo.tsx            # Fine-tuning & architecture overview
│   │   ├── App.tsx           # Main application shell
│   │   ├── index.css         # CIT academic design system
│   │   ├── main.tsx          # React entry point
│   │   └── types.ts          # TypeScript interfaces
│   ├── package.json
│   └── vite.config.ts
├── models/
│   ├── baseline/             # Cached all-MiniLM-L6-v2 model
│   └── finetuned/            # Fine-tuned cit-minilm-l6-v2 checkpoint
├── reports/
│   ├── dataset_inventory_report.json
│   ├── dataset_inventory_report.md
│   ├── dataset_split_summary.json
│   ├── evaluation_report.md
│   ├── evaluation_results.json
│   ├── ml_lab_report.md      # Full academic laboratory report
│   ├── preprocessing_report.json
│   └── training_history.json
├── scripts/
│   ├── build_index.py        # Builds ChromaDB & BM25 indices
│   ├── evaluate_retrieval.py # Benchmark evaluation script
│   ├── ingest.py             # Preprocessing & chunking pipeline
│   ├── inspect_datasets.py   # Dataset auditing & inventory script
│   ├── prepare_training_data.py # Leakage-resistant split generator
│   └── train_embeddings.py   # Sentence-transformer fine-tuning
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 12. Limitations & Future Work
1. **OCR on Historical Circulars**: A few older official attachments (e.g., historical UGC notifications) are scanned image PDFs. Incorporating local OCR (Tesseract / EasyOCR) would allow extracting older circulars.
2. **Quantized Local Generator**: Currently, the system uses a deterministic context-grounded synthesizer with zero hallucination. Coupling this with an on-device 4-bit quantized model (e.g. Qwen2.5-1.5B-Instruct or Llama-3.2-1B-Instruct via ONNX runtime) would support conversational rephrasing while retaining strict citation anchors.
3. **Dynamic Crawler Refresh**: Integrating a scheduled scraper to periodically monitor `https://cit.edu.in/announcements` would ensure instant capture of newly posted circulars and exam schedules.
