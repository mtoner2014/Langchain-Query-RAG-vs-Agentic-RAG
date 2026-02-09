# Query Expansion RAG vs. Agentic RAG for Healthcare QA

A comparative study of two Retrieval-Augmented Generation approaches for healthcare question answering, using [MedlinePlus](https://medlineplus.gov/) as the external knowledge source. Both pipelines are served through a Streamlit interface that also supports medical image analysis and crisis/emergency detection.

## Research Question

Which RAG architecture — a fixed retrieval pipeline with rule-based query decomposition (Query Expansion RAG) or a ReAct agent with tool access (Agentic RAG) — produces more complete and accurate answers for real-world healthcare queries, particularly multi-condition questions common among elderly patients?

## Key Findings

- **Query decomposition is the critical differentiator.** The Query Expansion RAG splits queries on explicit conjunctions ("and"/"or") via `_extract_topics`. When patients phrase multi-condition questions without these conjunctions — which is common in natural speech — the decomposition fails and the entire query becomes a single noisy search term. The Agentic RAG's `decompose_query` tool handles arbitrary phrasing through pattern matching and semantic analysis.
- **Agentic RAG advantages**: The agent can ask clarifying questions for vague queries before searching, adapt its search strategy mid-conversation, and access arbitrary URLs via `fetch_url`. Tool design quality is the primary determinant of agentic performance.
- **Query Expansion RAG advantages**: Lower latency (no agent reasoning loop), lower cost (fewer LLM calls), and more predictable behavior. For well-structured queries that use explicit conjunctions, it matches agentic quality.

## Example: Where the Approaches Diverge

**Query**: *"I'm diabetic, recently diagnosed with depression, what medications should I watch out for?"*

This query contains two conditions (diabetes, depression) joined by a comma and natural phrasing — no explicit "and"/"or".

### Query Expansion RAG

`_extract_topics` produces a single garbled topic because there is no "and"/"or" to split on:

```
Extracted topics: ["i'm diabetic recently diagnosed depression medications watch out"]
```

The scraper fails to match any MedlinePlus page with this string:

> I couldn't find relevant information on MedlinePlus for your question. Please try rephrasing or consult a healthcare professional.

### Agentic RAG

The agent decomposes the query intelligently, then calls three tools:

```
decompose_query("What medications should I watch out for as a diabetic recently diagnosed with depression?")
search_health_topic("diabetes and depression")
search_treatment_info("depression")
search_treatment_info("diabetes")
```

And produces a comprehensive response covering antidepressant classes (SSRIs, SNRIs, tricyclics), their effects on blood sugar, interactions with diabetes medications, and lifestyle considerations — with MedlinePlus source links for both conditions.

## Features

- **Dual RAG Modes** — switch between Query Expansion (direct retrieval pipeline) and Agentic (ReAct agent with tool use) from the sidebar
- **Medical Image Analysis** — upload lab reports, prescription labels, or symptom photos for AI-powered extraction and follow-up MedlinePlus search
- **Crisis & Emergency Detection** — automatically surfaces hotline numbers when queries mention suicide, self-harm, or medical emergencies
- **Multi-hop Reasoning** — the agentic mode decomposes complex questions (e.g. drug interactions with comorbidities) into sequential searches
- **Cross-encoder Reranking** — retrieved chunks are reranked with `cross-encoder/ms-marco-MiniLM-L-6-v2` for higher relevance
- **Debug Panel** — expandable section showing raw content retrieved from MedlinePlus before LLM transformation

## Architecture

| File | Role |
|---|---|
| `app.py` | Streamlit UI — chat interface, sidebar controls, crisis detection, image upload |
| `rag_medlineplus.py` | Query Expansion RAG — scrapes MedlinePlus per extracted topic, chunks with `RecursiveCharacterTextSplitter`, embeds into Chroma, reranks, and answers via LCEL chain |
| `agentic_rag_medlineplus.py` | Agentic RAG — LangChain ReAct agent with tools for topic search, symptom lookup, treatment info, query decomposition, and URL fetching |
| `image_processor.py` | Medical image analysis — uses GPT-4o-mini vision to extract text, medical terms, and a search query from uploaded images |

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

1. Select a retrieval mode (**Query Expansion RAG** or **Agentic RAG**) in the sidebar.
2. Choose your country for localized emergency numbers.
3. Type a health question in the chat input — or click one of the example cards.
4. Optionally upload a medical image (PNG, JPG, WEBP, GIF, up to 5 MB) for AI analysis before the MedlinePlus search.
5. Expand **Raw Content Retrieved from MedlinePlus** to inspect the source material.

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
