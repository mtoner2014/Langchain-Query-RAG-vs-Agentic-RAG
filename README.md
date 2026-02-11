# Vanilla RAG vs. Agentic RAG for Healthcare QA

A comparative study of two Retrieval-Augmented Generation approaches for healthcare question answering, using [MedlinePlus](https://medlineplus.gov/) as the external knowledge source. Both pipelines are served through a Streamlit interface that also supports medical image analysis and crisis/emergency detection.

## Features

- **Two RAG Architectures** — Vanilla (baseline) and Agentic (ReAct agent with tool use), both accessible from the Streamlit UI
- **Medical Image Analysis** — upload lab reports, prescription labels, or symptom photos for AI-powered extraction and follow-up MedlinePlus search
- **Crisis & Emergency Detection** — automatically surfaces hotline numbers when queries mention suicide, self-harm, or medical emergencies
- **Multi-hop Reasoning** — the agentic mode decomposes complex questions (e.g. drug interactions with comorbidities) into sequential searches
- **FlashRank Reranking** — retrieved chunks are reranked with FlashRank for higher relevance
- **Debug Panel** — expandable section showing raw content retrieved from MedlinePlus before LLM transformation

## Architecture

| File | Role |
|---|---|
| `app.py` | Streamlit UI — chat interface, sidebar controls, crisis detection, image upload |
| `Vanilla_RAG.py` | Vanilla RAG — sends the raw query directly to MedlinePlus search, chunks, embeds into Chroma, reranks, and answers via LCEL chain. No query preprocessing or topic extraction. |
| `agentic_rag_medlineplus.py` | Agentic RAG — LangChain ReAct agent with tools for topic search, symptom lookup, treatment info, query decomposition, and URL fetching |
| `image_processor.py` | Medical image analysis — uses GPT-4o-mini vision to extract text, medical terms, and a search query from uploaded images |

### Vanilla RAG Pipeline

```
User query → MedlinePlus search (raw query) → scrape top results → chunk → Chroma vector store → FlashRank rerank → GPT-4o-mini → response
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

### Streamlit UI (Vanilla & Agentic RAG)

1. Select a retrieval mode (**Agentic RAG** or **Vanilla RAG**) in the sidebar.
2. Choose your country for localized emergency numbers.
3. Type a health question in the chat input — or click one of the example cards.
4. Optionally upload a medical image (PNG, JPG, WEBP, GIF, up to 5 MB) for AI analysis before the MedlinePlus search.
5. Expand **Raw Content Retrieved from MedlinePlus** to inspect the source material.

### Vanilla RAG (standalone CLI)

```bash
python Vanilla_RAG.py
```

Interactive prompt — type a health question and get a MedlinePlus-sourced answer. Type `quit` to exit.

## Key Dependencies

| Package | Purpose |
|---|---|
| `langchain` / `langchain-openai` | LLM orchestration and LCEL chains |
| `langchain-chroma` | Ephemeral vector store (both modes) |
| `langchain-huggingface` | Local embeddings (`all-MiniLM-L6-v2`) |
| `flashrank` | Fast ONNX-based reranking via `langchain-community` |
| `streamlit` | Web UI |
| `beautifulsoup4` / `requests` | MedlinePlus scraping |
| `python-dotenv` | Environment variable loading |

## Disclaimer

This tool provides general health information sourced from MedlinePlus and is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.
