# CIT Knowledge AI — Complete System Documentation & Machine Learning Guide

---

## 1. Executive Project Overview

**CIT Knowledge AI** is an intelligent, multi-source Retrieval-Augmented Generation (RAG) and campus navigation system built specifically for **Coimbatore Institute of Technology (CIT)**.

The system serves students, faculty, and prospective applicants by providing:
1. **Accurate, Fact-Grounded Answers** about CIT departments, faculty directories, academic regulations, admissions, fees, and placements without hallucinations.
2. **Turn-by-Turn Campus Navigation** with step-by-step pedestrian walking routes, distances in metres, walk times, and landmarks across all campus blocks.
3. **Multi-Source Intelligent Escalation**: A 3-tier hierarchical engine querying:
   * **Level 1**: Curated local institutional dataset.
   * **Level 2**: Live verified official CIT website ([cit.edu.in](https://cit.edu.in)).
   * **Level 3**: Public web fallback for dynamic campus practicals (e.g. gate closing hours).

---

## 2. Complete Technology Stack & Architecture

```
                          ┌─────────────────────────────────────┐
                          │     React + TypeScript Frontend     │
                          │        (Vite Dev & Production)      │
                          └──────────────────┬──────────────────┘
                                             │ HTTP REST
                                             ▼
                          ┌─────────────────────────────────────┐
                          │       FastAPI Backend Engine        │
                          │   (Uvicorn ASGI, Python 3.13)       │
                          └──────────────────┬──────────────────┘
                                             │
                  ┌──────────────────────────┴──────────────────────────┐
                  ▼                                                     ▼
      ┌───────────────────────┐                             ┌───────────────────────┐
      │  RAG Enquiry Pipeline │                             │ CIT Campus Navigator  │
      │   (rag_engine.py)     │                             │ (roadmap_navigator.py)│
      └───────────┬───────────┘                             └───────────┬───────────┘
                  │                                                     │
      ┌───────────┴───────────┐                             ┌───────────┴───────────┐
      ▼                       ▼                             ▼                       ▼
┌──────────────┐      ┌──────────────┐              ┌──────────────┐        ┌──────────────┐
│  ChromaDB    │      │  BM25 Okapi  │              │ 8-Node Graph │        │ 56 Precomputed│
│ Vector Store │      │ Sparse Index │              │ Dijkstra BFS │        │ Route Pairs  │
└──────────────┘      └──────────────┘              └──────────────┘        └──────────────┘
```

### 2.1 Backend Stack
* **Language & Runtime**: Python 3.13.14 (Virtual Environment `.venv`)
* **Web Framework**: FastAPI (`0.143.2`) with Pydantic (`2.14.1`) data validation
* **ASGI Server**: Uvicorn (`0.54.0`)
* **Vector Database**: ChromaDB (`1.5.9`) persistent client
* **Lexical Search Engine**: `rank-bm25` (`0.2.2`)
* **Deep Learning & NLP**:
  * PyTorch (`2.14.1+cpu`)
  * Sentence Transformers (`6.1.0`)
  * Hugging Face Transformers (`5.19.0`)
  * Scikit-Learn (`1.9.1`) & NumPy (`2.5.4`)
* **Web Scraping & Network**:
  * HTTPX (`0.28.1`) asynchronous & synchronous client
  * BeautifulSoup4 (`4.15.0`) & Soupsieve (`3.0.3`) for HTML parsing
* **Testing Framework**: Pytest (`9.1.1`) with Starlette TestClient (33 automated unit & integration tests)

### 2.2 Frontend Stack
* **Framework**: React 18 with TypeScript
* **Build Tool & Bundler**: Vite 8.3
* **Styling**: Modern dark/light glassmorphic UI, responsive layouts, Tailwind & curated CSS
* **Icons**: Lucide React
* **Real-time Features**: Source badges, latency counter, multi-model switcher, feedback buttons, interactive citation modal

---

## 3. Core System Components

### 3.1 Retrieval Engines
The user can toggle between three retrieval strategies in the UI:
1. **Hybrid (BM25 + Dense) [Default & Recommended]**: Combines sparse lexical keyword matching with dense semantic embeddings via Reciprocal Rank Fusion (RRF). Delivers the highest Recall@5 (47.6%) and MRR (0.3492).
2. **Fine-Tuned Dense**: Custom domain-adapted `cit-minilm-l6-v2` trained on CIT question-passage pairs using Multiple Negatives Ranking Loss (MNRL).
3. **Baseline MiniLM**: Standard `sentence-transformers/all-MiniLM-L6-v2` dense embedding model.

### 3.2 Multi-Source Orchestration Hierarchy
* **Level 1 (Local Institutional Dataset)**: Queries 290 high-density chunks extracted from 10 raw institutional documents.
* **Level 2 (Official CIT Website Retriever)**: If Level 1 cannot verify the answer (e.g. up-to-date faculty rosters), the system crawls the official college website (`cit.edu.in`), cleans the HTML, verifies the content, and cites the live URL.
* **Level 3 (Public Web Retriever)**: If both Level 1 and 2 lack data, queries public web channels for campus operational policies (like visitor entry and gate timings) with a strict unverified fallback.

### 3.3 Campus Roadmap Navigation Engine
* **Dataset Foundation**: Built on the schema of `expanded_roadmap_dataset`.
* **Graph Structure**: 8 campus hubs (Campus Main Gate, Main Block, Central Library, Computing Block, IT Block, Auditorium, MSc/MBA Block, Mechanical & Civil Block) connected by 18 bidirectional walkways.
* **Algorithm**: Dijkstra's Shortest Path Algorithm over weighted edges (`distance_m`, `estimated_walk_seconds`, `direction`, `path_type`, `landmarks`).
* **Trigger**: Automatically extracts origin (*from where*) and destination (*to where*) whenever the prompt contains *"navigate"*, *"how to reach"*, or *"directions to"*.

---

## 4. Glossary of Machine Learning & AI Terms Used

Below is a complete explanation of every machine learning, NLP, and information retrieval concept used in this project, explaining **what it is** and **for what purpose it is used**.

---

### 1. RAG (Retrieval-Augmented Generation)
* **What it is:** An AI architecture where an information retrieval system first searches a curated knowledge base for relevant facts before a generation module synthesizes the final answer.
* **For what it is used in this project:** Prevents the AI from making up facts (hallucinating) about Coimbatore Institute of Technology. The AI only responds using verified evidence retrieved from the CIT database or official site.

---

### 2. Dense Vector Embeddings
* **What it is:** A numerical representation of text where sentences are converted into high-dimensional vectors (arrays of 384 floating-point numbers). In this vector space, texts with similar meanings are positioned close together.
* **For what it is used in this project:** Allows the system to understand synonyms and paraphrasing. For example, a student asking *"What is the cost of studying?"* is mapped close to document chunks containing *"tuition fees"*.

---

### 3. Sparse Retrieval / Lexical Search (BM25 Okapi)
* **What it is:** A probabilistic information retrieval algorithm based on word frequency, term frequency (TF), and inverse document frequency (IDF).
* **For what it is used in this project:** Dense vector search sometimes overlooks exact codes, numbers, or acronyms. BM25 guarantees that queries with exact keywords (like course code *"TNEA 2007"*, *"1956"*, or specific phone numbers) are matched with 100% precision.

---

### 4. Hybrid Retrieval
* **What it is:** The simultaneous execution of both **Dense Retrieval** (semantic vectors) and **Sparse Retrieval** (lexical keywords).
* **For what it is used in this project:** Combines the best of both worlds: semantic understanding of conversational language and exact-match precision for college codes and numbers.

---

### 5. RRF (Reciprocal Rank Fusion)
* **What it is:** A rank aggregation algorithm defined by the formula:
  $$\text{RRF\_Score}(d) = \sum_{m \in \text{models}} \frac{1}{k + \text{rank}_m(d)}$$
  where $k=60$ is a smoothing constant.
* **For what it is used in this project:** Merges the ranked search results of ChromaDB and BM25 into a single, unified ranking without needing to normalize scores from completely different scales.

---

### 6. Fine-Tuning & Domain Adaptation
* **What it is:** Taking a general pre-trained deep neural network and continuing its training on a smaller, domain-specific dataset.
* **For what it is used in this project:** The generic `all-MiniLM-L6-v2` model does not know what *"MSc SS"*, *"DCS"*, or *"CIT"* mean. Fine-tuning adjusts the model weights so that CIT-specific department names and abbreviations are properly recognized in vector space.

---

### 7. Contrastive Learning & MNRL (Multiple Negatives Ranking Loss)
* **What it is:** A loss function used to train embedding models where each positive (query, relevant passage) pair in a batch treats all other passages in that batch as negative examples.
* **For what it is used in this project:** Pulls the vector representation of a student question closer to the correct document chunk while pushing away all irrelevant college passages.

---

### 8. Cosine Similarity
* **What it is:** A mathematical metric that measures the cosine of the angle between two multi-dimensional vectors:
  $$\text{Cosine Similarity}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$
* **For what it is used in this project:** Measures how semantically relevant a document chunk is to the user's query on a scale from `0.0` to `1.0`.

---

### 9. Vector Database (ChromaDB)
* **What it is:** A specialized database engine designed to store, index, and query high-dimensional vector embeddings using Approximate Nearest Neighbor (ANN) search algorithms (like HNSW).
* **For what it is used in this project:** Stores the 384-dimensional embeddings of all 290 CIT institutional text chunks in persistent storage (`vector_store/chroma`) and retrieves top candidates in milliseconds.

---

### 10. Chunking & Chunk Overlap
* **What it is:** The process of breaking large text documents into smaller, coherent text passages (chunks), often with overlapping sentences between adjacent segments.
* **For what it is used in this project:** Transformer models have maximum sequence length limits (e.g., 256 or 512 tokens). Splitting CIT regulations and brochures into 200–500 word chunks ensures every chunk contains complete facts without overflowing the model context.

---

### 11. MRR (Mean Reciprocal Rank)
* **What it is:** An evaluation metric for information retrieval:
  $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
  where $\text{rank}_i$ is the position of the first correct chunk for query $i$.
* **For what it is used in this project:** Evaluates how high up the relevant source document is placed in search results. Hybrid retrieval achieved the highest MRR of **0.3492**.

---

### 12. Recall@K and Precision@K
* **What it is:**
  * **Recall@K**: The percentage of queries where the correct ground-truth document was retrieved within the top $K$ results.
  * **Precision@K**: The proportion of retrieved documents in the top $K$ that are actually relevant.
* **For what it is used in this project:** Benchmark metrics in `scripts/evaluate_retrieval.py` proving that Hybrid retrieval achieves **Recall@5 of 47.62%** compared to 38.1% for the baseline model.

---

### 13. Grounding & Hallucination Prevention
* **What it is:**
  * **Hallucination**: When an AI generates plausible-sounding but factually incorrect information.
  * **Grounding**: Constraining the AI to generate answers strictly from retrieved, cited facts.
* **For what it is used in this project:** If information is missing or contradictory (e.g. historical HOD in 2020), the system outputs a calibrated refusal rather than fabricating a name.

---

### 14. Graph Pathfinder / Dijkstra's Algorithm
* **What it is:** A graph search algorithm that finds the shortest path between nodes in a weighted graph by maintaining a priority queue of minimum distances.
* **For what it is used in this project:** Computes the optimal walking route between any two buildings on campus using distances in metres and walking times from the CIT campus roadmap graph.

---

## 5. Directory Structure & Key Files

```
e:\Own model RAG\
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI application setup and API endpoints
│   │   ├── config.py                # App configuration, directories, and thresholds
│   │   ├── schemas.py               # Pydantic request/response data models
│   │   ├── rag_engine.py            # Level 1 RAG engine & answer synthesis
│   │   ├── roadmap_navigator.py     # Campus roadmap inference & route predictor
│   │   └── multi_source/
│   │       ├── orchestrator.py      # 3-tier multi-source query coordinator
│   │       ├── cit_website_retriever.py # Live official website crawler (cit.edu.in)
│   │       ├── public_web_retriever.py  # Public web search fallback
│   │       └── network_utils.py     # HTML cleaning, caching, and text sanitization
│   └── tests/
│       ├── test_api.py              # API endpoint unit tests
│       ├── test_rag.py              # RAG retrieval and citation integrity tests
│       ├── test_regression.py       # HOD, course, and fee regression tests
│       ├── test_multi_source.py     # Multi-source hierarchy tests
│       └── test_navigation_roadmap.py # Turn-by-turn roadmap navigation tests
├── data/
│   ├── raw/                         # 10 raw institutional TXT files + Navigations/
│   │   └── Navigations/             # CIT locations.csv and walking_path_edges.csv
│   └── processed/                   # Ingested chunks.json and train/val/test splits
├── expanded_roadmap_dataset/        # Text-to-roadmap synthetic dataset & schema
├── frontend/
│   ├── src/
│   │   ├── App.tsx                  # Main React container
│   │   ├── components/ChatInterface.tsx # Chat UI, model switcher, and citations
│   │   └── types.ts                 # TypeScript type interfaces
│   └── package.json
├── models/
│   ├── baseline/                    # Local all-MiniLM-L6-v2 model files
│   ├── finetuned/                   # Domain-adapted cit-minilm-l6-v2 model files
│   └── cit_roadmap_model.json       # Compiled CIT campus roadmap model
├── reports/                         # Preprocessing and evaluation reports
├── scripts/
│   ├── ingest.py                    # Chunking and ingestion into ChromaDB
│   ├── build_index.py               # BM25 index generator
│   ├── train_embeddings.py          # Fine-tuning SentenceTransformer with MNRL
│   ├── train_roadmap_model.py       # Campus roadmap graph compiler
│   └── evaluate_retrieval.py        # Retrieval benchmark evaluation suite
└── requirements.txt                 # Python dependencies
```

---

## 6. How to Run the Project

### 1. Start the FastAPI Backend
```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
Backend runs at: [http://127.0.0.1:8000](http://127.0.0.1:8000) (Docs at `/docs`)

### 2. Start the React Frontend
```powershell
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```
Frontend runs at: [http://127.0.0.1:5173](http://127.0.0.1:5173)

### 3. Run the Automated Test Suite (33 Tests)
```powershell
.\.venv\Scripts\pytest.exe backend/tests/ -v
```

### 4. Run Retrieval Benchmark Evaluation
```powershell
.\.venv\Scripts\python.exe scripts/evaluate_retrieval.py
```
Outputs report to `reports/evaluation_report.md`.
