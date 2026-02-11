# Vanilla RAG vs. Query Expansion RAG vs. Agentic RAG for Healthcare QA

A comparative study of three Retrieval-Augmented Generation approaches for healthcare question answering, using [MedlinePlus](https://medlineplus.gov/) as the external knowledge source. The Query Expansion and Agentic pipelines are served through a Streamlit interface that also supports medical image analysis and crisis/emergency detection. The Vanilla RAG runs as a standalone CLI for baseline benchmarking.

## Research Question

How do three RAG architectures — a simple retrieve-and-generate pipeline (Vanilla RAG), a fixed retrieval pipeline with rule-based query decomposition (Query Expansion RAG), and a ReAct agent with tool access (Agentic RAG) — compare in completeness and accuracy for real-world healthcare queries, particularly multi-condition questions common among elderly patients?

## Key Findings

- **Vanilla RAG has a hard ceiling.** It works well for straightforward single-topic queries but completely fails when the question involves multiple medical concepts. Sending raw user queries to search without any preprocessing is a fundamental limitation — both the Complex and Very Complex test queries returned zero useful information.
- **Query decomposition is the critical differentiator.** The Query Expansion RAG splits queries on explicit conjunctions ("and"/"or") via `_extract_topics`. When patients phrase multi-condition questions without these conjunctions — which is common in natural speech — the decomposition fails and the entire query becomes a single noisy search term. The Agentic RAG's `decompose_query` tool handles arbitrary phrasing through pattern matching and semantic analysis.
- **Query Expansion RAG is the best all-rounder.** Topic extraction + alias mapping + combined search gives it reliable performance across all complexity levels. It produced the most detailed answers and maintained consistent response times (10–21s).
- **Agentic RAG shines on reasoning-heavy queries.** Its ability to decompose questions, plan search strategies, and synthesize across multiple tool results makes it the strongest choice for complex drug-interaction and comorbidity questions. The trade-off is higher latency on complex queries (up to 22s).
- **The gap widens with complexity.** For simple queries all three systems are comparable. The real differentiation appears at the Complex and Very Complex levels, where architectural differences in retrieval strategy become decisive.

## Example: Where the Approaches Diverge

**Query**: *"I have bipolar disorder and insomnia, what should I be aware of?"*

This multi-condition query requires the system to retrieve and synthesize information about two distinct medical topics.

### Vanilla RAG

Sends the raw query directly to MedlinePlus search. The natural-language phrasing fails to match any single page:

> I couldn't find relevant information on MedlinePlus for your question. Please try rephrasing or consult a healthcare professional.

Response time: 1.4s (fast, but only because it gave up immediately).

### Query Expansion RAG

`_extract_topics` splits on "and", producing two separate searches for "bipolar disorder" and "insomnia". It retrieves relevant pages for both and combines them into a 550-word response covering symptoms, causes, treatment options, and — critically — the overlap between the two conditions.

### Agentic RAG

The agent decomposes the query via its `decompose_query` tool, then performs multi-hop searches across both topics. It produces a 337-word response with an "Overlapping Considerations" section addressing medication interactions between bipolar treatments and sleep aids.

## Features

- **Three RAG Architectures** — Vanilla (baseline), Query Expansion (topic splitting + alias mapping), and Agentic (ReAct agent with tool use). The Streamlit UI supports Query Expansion and Agentic modes; Vanilla RAG runs as a standalone CLI for benchmarking.
- **Automated Comparison Harness** — `test_rag_comparison.py` runs all three systems against queries of increasing complexity and generates a Markdown report (`RAG_Comparison_Report.md`)
- **Medical Image Analysis** — upload lab reports, prescription labels, or symptom photos for AI-powered extraction and follow-up MedlinePlus search
- **Crisis & Emergency Detection** — automatically surfaces hotline numbers when queries mention suicide, self-harm, or medical emergencies
- **Multi-hop Reasoning** — the agentic mode decomposes complex questions (e.g. drug interactions with comorbidities) into sequential searches
- **Cross-encoder Reranking** — retrieved chunks are reranked with `cross-encoder/ms-marco-MiniLM-L-6-v2` for higher relevance
- **Debug Panel** — expandable section showing raw content retrieved from MedlinePlus before LLM transformation

## Architecture

| File | Role |
|---|---|
| `app.py` | Streamlit UI — chat interface, sidebar controls, crisis detection, image upload |
| `Vanilla_RAG.py` | Vanilla RAG — sends the raw query directly to MedlinePlus search, chunks, embeds into Chroma, reranks, and answers via LCEL chain. No query preprocessing or topic extraction. Standalone CLI. |
| `rag_medlineplus.py` | Query Expansion RAG — scrapes MedlinePlus per extracted topic, chunks with `RecursiveCharacterTextSplitter`, embeds into Chroma, reranks, and answers via LCEL chain |
| `agentic_rag_medlineplus.py` | Agentic RAG — LangChain ReAct agent with tools for topic search, symptom lookup, treatment info, query decomposition, and URL fetching |
| `image_processor.py` | Medical image analysis — uses GPT-4o-mini vision to extract text, medical terms, and a search query from uploaded images |
| `test_rag_comparison.py` | Comparison test harness — runs a battery of queries at increasing complexity across all three RAG systems and generates `RAG_Comparison_Report.md` |

### Vanilla RAG Pipeline

```
User query → MedlinePlus search (raw query) → scrape top results → chunk → Chroma vector store → cross-encoder rerank → GPT-4o-mini → response
```

### Query Expansion RAG Pipeline

```
User query → _extract_topics (split on "and"/"or") → MedlinePlus scraper (per topic) → Chroma vector store → cross-encoder rerank → GPT-4o-mini → response
```

### Agentic RAG Pipeline

```
User query → ReAct agent → [decompose_query | search_health_topic | search_symptoms | search_treatment_info | fetch_url] → Chroma vector store → cross-encoder rerank → GPT-4o-mini → response
```

## Prerequisites

- Python 3.13+
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd LangchainProj2
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\activate      # Windows
   pip install .
   ```

3. **Configure environment variables**

   Create a `.env` file in the project root:

   ```
   OPENAI_API_KEY=sk-...
   ```

4. **Run the application**

   ```bash
   streamlit run app.py
   ```

   The app will open at `http://localhost:8501`.

## Usage

### Streamlit UI (Query Expansion & Agentic RAG)

1. Select a retrieval mode (**Query Expansion RAG** or **Agentic RAG**) in the sidebar.
2. Choose your country for localized emergency numbers.
3. Type a health question in the chat input — or click one of the example cards.
4. Optionally upload a medical image (PNG, JPG, WEBP, GIF, up to 5 MB) for AI analysis before the MedlinePlus search.
5. Expand **Raw Content Retrieved from MedlinePlus** to inspect the source material.

### Vanilla RAG (standalone CLI)

```bash
python Vanilla_RAG.py
```

Interactive prompt — type a health question and get a MedlinePlus-sourced answer. Type `quit` to exit.

### Running the Comparison Test Harness

```bash
python test_rag_comparison.py
```

Runs all three RAG systems against a battery of queries at four complexity levels (simple, moderate, complex, very complex) and writes the results to `RAG_Comparison_Report.md`.

## Key Dependencies

| Package | Purpose |
|---|---|
| `langchain` / `langchain-openai` | LLM orchestration and LCEL chains |
| `langchain-chroma` | Ephemeral vector store (both modes) |
| `langchain-huggingface` | Local embeddings (`all-MiniLM-L6-v2`) and cross-encoder reranking |
| `streamlit` | Web UI |
| `beautifulsoup4` / `requests` | MedlinePlus scraping |
| `python-dotenv` | Environment variable loading |

## Disclaimer

This tool provides general health information sourced from MedlinePlus and is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.
