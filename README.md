# PolicyPal

> RAG-powered chatbot that answers questions from insurance policy documents using LangChain, ChromaDB, and FastAPI

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.138-green)
![LangChain](https://img.shields.io/badge/LangChain-1.3-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-purple)
![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.3%2070B-red)

---

## The Problem

Insurance policy documents are dense, lengthy, and full of legal language. Finding a specific clause like the waiting period for pre-existing diseases or the room rent limit requires reading dozens of pages. PolicyPal lets you ask plain English questions and get accurate, cited answers directly from the policy document.

---

## Architecture

```
                        INDEX PIPELINE (run once)
┌─────────────┐    ┌──────────────┐    ┌────────────────┐    ┌─────────────┐
│  Policy PDF │───▶│  PyMuPDF     │───▶│ RecursiveChar  │───▶│  ChromaDB   │
│             │    │  Loader      │    │ TextSplitter   │    │  Vector     │
└─────────────┘    └──────────────┘    │ (500/80)       │    │  Store      │
                                       └────────────────┘    └─────────────┘
                                                                     │
                        QUERY PIPELINE (every question)              │
┌─────────────┐    ┌──────────────┐    ┌────────────────┐           │
│    User     │───▶│  BGE-small   │───▶│  Similarity    │───────────┘
│  Question   │    │  Embeddings  │    │  Search (k=3)  │
└─────────────┘    └──────────────┘    └────────────────┘
                                                │
                                                ▼
                                       ┌────────────────┐    ┌─────────────┐
                                       │  RAG Prompt    │───▶│  Groq LLM   │
                                       │  Template      │    │  Llama 3.3  │
                                       └────────────────┘    │  70B        │
                                                             └─────────────┘
                                                                     │
                                                                     ▼
                                                             ┌─────────────┐
                                                             │   Answer +  │
                                                             │   Sources   │
                                                             └─────────────┘
```

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| LLM | Groq Llama 3.3 70B | Free, fast inference, strong reasoning |
| Embeddings | BAAI/bge-small-en-v1.5 | Free, runs on CPU, 384 dims |
| Vector Store | ChromaDB | Local persistence, no infra needed |
| Orchestration | LangChain LCEL | Composable, transparent pipeline |
| PDF Parsing | PyMuPDF | Fast, reliable text extraction |
| API | FastAPI | Auto docs, Pydantic validation |

---

## Project Structure

```
policypal/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── routers/
│   │   ├── ingestion.py     # POST /api/upload
│   │   └── query.py         # POST /api/query
│   ├── services/
│   │   ├── document_loader.py  # PDF loading + chunking
│   │   ├── embedder.py         # HuggingFace embeddings (singleton)
│   │   ├── vectorstore.py      # ChromaDB wrapper (singleton)
│   │   └── rag_chain.py        # LCEL RAG chain (singleton)
│   ├── schemas/
│   │   └── query.py         # Pydantic request/response models
│   ├── prompts/
│   │   └── qa_prompt.py     # RAG prompt template
│   └── core/
│       ├── config.py        # Settings loader
│       └── logger.py        # Structured JSON logging
├── data/
│   ├── raw/                 # Uploaded PDFs
│   └── processed/           # ChromaDB persistence
├── notebooks/
│   ├── 01_explore_pdf.ipynb
│   ├── 02_embeddings.ipynb
│   └── 03_rag_chain.ipynb
├── notes/
│   └── eval_set.md          # Manual evaluation results
├── .env.example
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.11+
- [Groq API key](https://console.groq.com) (free)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/policypal.git
cd policypal

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Run the server
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` to see the interactive API.

---

## API Usage

### Upload a policy PDF
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@data/raw/star_health_family_optima.pdf"
```

Response:
```json
{
  "job_id": "a1b2c3d4-...",
  "status": "completed",
  "filename": "star_health_family_optima.pdf",
  "chunks_created": 324
}
```

### Ask a question
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the waiting period for pre-existing diseases?"}'
```

Response:
```json
{
  "answer": "The waiting period for pre-existing diseases is 36 months from the date of inception of the policy (Page 11).",
  "sources": "[Page 11]\nwaiting period of 36 months..."
}
```

### Health check
```bash
curl http://localhost:8000/health
```

---

## Evaluation Results

Tested on Star Health Assure Insurance Policy with 10 questions:

| Result | Count |
|---|---|
| ✅ Fully correct | 4/10 |
| ⚠️ Partially correct | 5/10 |
| ❌ Wrong | 1/10 |

**No hallucinations observed** - the system correctly refused to invent answers when information wasn't in the document.

Main failure mode: table data in PDFs gets flattened during extraction, losing column context (affects room rent tiers, sum insured options).

---

## Known Limitations

1. **Table extraction** - PyMuPDF flattens tables into plain text, losing column relationships. Queries about tiered data (room rent by sum insured) return incomplete answers.
2. **Single document** - re-uploading a PDF wipes the previous one. Multi-document support requires deduplication logic.
3. **Page number mismatch** - cited page numbers are PDF page indices, which may differ from printed page numbers in the document.
4. **Synchronous ingestion** - large PDFs block the `/upload` endpoint until fully processed. No background processing yet.
5. **No conversation memory** - each query is independent. Follow-up questions don't retain context from previous answers.

---

## Future Work

| Priority | Feature | Why |
|---|---|---|
| P0 | Streamlit UI | Demo-able without curl commands |
| P0 | RAGAS evaluation | Formal metrics (faithfulness, recall) |
| P1 | Table-aware parsing | Fix biggest retrieval failure mode |
| P1 | Hybrid search (BM25 + vector) | Better keyword matching |
| P1 | Cross-encoder reranking | Improve retrieval precision |
| P2 | Multi-turn chat memory | Handle follow-up questions |
| P3 | Docker + deployment | Live demo URL |
| P3 | LangSmith tracing | Production observability |

---

## Why I Built This

Insurance policy documents are one of the most important and most unread documents people sign. This project explores whether RAG can make dense legal documents genuinely accessible to everyday users, while staying grounded in the actual document without hallucinating coverage that doesn't exist.

## Author
S.Sreeja
