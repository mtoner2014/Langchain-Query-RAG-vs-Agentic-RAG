"""
Agentic RAG System for MedlinePlus Healthcare Information
An agent that can reason about healthcare queries and use tools to retrieve information.
"""

import os
import re
import requests
from urllib.parse import urlparse, parse_qs, unquote
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Annotated
from dotenv import load_dotenv

import shutil
import tempfile

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

load_dotenv()


class MedlinePlusTools:
    """Tools for the agent to interact with MedlinePlus."""

    BASE_URL = "https://medlineplus.gov"

    # Map common medical terms/synonyms to their MedlinePlus page slugs
    TOPIC_ALIASES = {
        'hypertension': 'highbloodpressure',
        'high blood pressure': 'highbloodpressure',
        'hypotension': 'lowbloodpressure',
        'low blood pressure': 'lowbloodpressure',
        'hyperlipidemia': 'cholesterol',
        'high cholesterol': 'cholesterol',
        'dyslipidemia': 'cholesterol',
        'diabetes mellitus': 'diabetes',
        'type 2 diabetes': 'diabetestype2',
        'type 1 diabetes': 'diabetestype1',
        'myocardial infarction': 'heartattack',
        'heart attack': 'heartattack',
        'cerebrovascular accident': 'stroke',
        'cva': 'stroke',
        'uri': 'commoncold',
        'common cold': 'commoncold',
        'gerd': 'gastroesophagealreflux',
        'acid reflux': 'gastroesophagealreflux',
        'uti': 'urinarytractinfections',
        'urinary tract infection': 'urinarytractinfections',
        'copd': 'copd',
        'obesity': 'obesity',
        'overweight': 'obesity',
        'depression': 'depression',
        'anxiety': 'anxiety',
        'insomnia': 'insomnia',
        'sleep disorders': 'sleepdisorders',
        'migraine': 'migraine',
        'migraines': 'migraine',
        'alzheimers': 'alzheimersdisease',
        "alzheimer's": 'alzheimersdisease',
        "alzheimer's disease": 'alzheimersdisease',
        'parkinsons': 'parkinsonsdisease',
        "parkinson's": 'parkinsonsdisease',
        "parkinson's disease": 'parkinsonsdisease',
        'osteoporosis': 'osteoporosis',
        'arthritis': 'arthritis',
        'rheumatoid arthritis': 'rheumatoidarthritis',
        'osteoarthritis': 'osteoarthritis',
        'pneumonia': 'pneumonia',
        'bronchitis': 'bronchitis',
        'anemia': 'anemia',
        'hypothyroidism': 'hypothyroidism',
        'hyperthyroidism': 'hyperthyroidism',
        'eczema': 'eczema',
        'psoriasis': 'psoriasis',
        'kidney disease': 'kidneydiseases',
        'chronic kidney disease': 'chronickidneydisease',
        'liver disease': 'liverdiseases',
        'hepatitis': 'hepatitis',
        'hiv': 'hivaids',
        'aids': 'hivaids',
        'tuberculosis': 'tuberculosis',
        'tb': 'tuberculosis',
        'cancer': 'cancer',
        'leukemia': 'leukemia',
        'lymphoma': 'lymphoma',
        'epilepsy': 'epilepsy',
        'seizures': 'seizures',
        'allergies': 'allergy',
        'allergy': 'allergy',
    }

    TOP_K = 4
    RERANK_FETCH_K = 10

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Educational RAG System)'
        })
        # Cache to avoid repeated fetches
        self._cache = {}
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        # Cross-encoder for reranking retrieved chunks
        self.cross_encoder = HuggingFaceCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        )

    @staticmethod
    def _relevance_score(query: str, link_text: str, url: str) -> float:
        """Score how relevant a search result is to the query (higher is better)."""
        query_words = set(query.lower().split())
        link_words = set(link_text.lower().split())
        # Extract slug from topic URLs or drug URLs (/druginfo/meds/a682159.html)
        slug_match = re.search(r'medlineplus\.gov/(?:druginfo/\w+/)?([a-zA-Z0-9]+)\.html$', url)
        slug = slug_match.group(1).lower() if slug_match else ''

        # Word overlap between query and link text
        overlap = query_words & link_words
        score = len(overlap) * 2.0

        # Bonus if the slug contains any query word
        for word in query_words:
            if word in slug:
                score += 3.0

        # Bonus if the link text contains the full query as a substring
        if query.lower() in link_text.lower():
            score += 5.0

        # Boost if the URL slug matches a known alias for any query term
        # (e.g. query has "hypertension" and slug is "highbloodpressure")
        for qw in query_words:
            alias_slug = MedlinePlusTools.TOPIC_ALIASES.get(qw)
            if alias_slug and alias_slug == slug:
                score += 5.0

        # Penalty for very generic pages
        generic_slugs = {'heartdiseases', 'heartdisease', 'healthtopics'}
        if slug in generic_slugs:
            score -= 2.0

        return score

    def _search_medlineplus(self, query: str) -> List[str]:
        """Search MedlinePlus and return relevant health topic page URLs, ranked by relevance."""
        candidates = []
        seen = set()
        try:
            search_url = "https://vsearch.nlm.nih.gov/vivisimo/cgi-bin/query-meta"
            params = {
                'v:project': 'medlineplus',
                'v:sources': 'medlineplus-bundle',
                'query': query,
            }
            response = self.session.get(search_url, params=params, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    # Search results use redirect URLs with the real URL in a 'url' param
                    if 'v:frame=redirect' in href or 'v%3aframe=redirect' in href:
                        parsed = urlparse(href)
                        qs = parse_qs(parsed.query)
                        target = qs.get('url', [None])[0]
                        if target:
                            target = unquote(target)
                            if re.match(r'https://medlineplus\.gov/(?:[a-zA-Z]+|druginfo/\w+/\w+)\.html$', target):
                                if target not in seen:
                                    seen.add(target)
                                    link_text = link.get_text(strip=True)
                                    candidates.append((target, link_text))
        except Exception:
            pass

        # Rank by relevance to the query
        if candidates:
            candidates.sort(
                key=lambda c: self._relevance_score(query, c[1], c[0]),
                reverse=True,
            )
            return [c[0] for c in candidates[:3]]
        return []

    def _scrape_topic_page(self, topic: str) -> Optional[Dict]:
        """Scrape a health topic page."""
        # Check cache first
        cache_key = topic.lower().strip()
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Build candidate URLs: alias → direct slug guess → search results
        urls_to_try = []

        # Check alias map first for known medical synonyms
        if cache_key in self.TOPIC_ALIASES:
            alias_slug = self.TOPIC_ALIASES[cache_key]
            urls_to_try.append(f"{self.BASE_URL}/{alias_slug}.html")
        else:
            # Check if any known alias appears within the topic
            for alias in sorted(self.TOPIC_ALIASES, key=len, reverse=True):
                if re.search(r'\b' + re.escape(alias) + r'\b', cache_key):
                    alias_slug = self.TOPIC_ALIASES[alias]
                    urls_to_try.append(f"{self.BASE_URL}/{alias_slug}.html")
                    break

        slug = cache_key.replace(' ', '').replace('-', '')
        direct_url = f"{self.BASE_URL}/{slug}.html"
        if direct_url not in urls_to_try:
            urls_to_try.append(direct_url)

        # For multi-word topics, also try individual words as slugs
        words = cache_key.split()
        if len(words) > 1:
            for word in words:
                word = word.strip()
                if len(word) > 2:
                    candidate = f"{self.BASE_URL}/{word}.html"
                    if candidate not in urls_to_try:
                        urls_to_try.append(candidate)

        # Add URLs discovered via MedlinePlus search (now ranked by relevance)
        search_urls = self._search_medlineplus(topic)
        for u in search_urls:
            if u not in urls_to_try:
                urls_to_try.append(u)

        for url in urls_to_try:
            try:
                response = self.session.get(url, timeout=8)
                if response.status_code != 200:
                    continue

                soup = BeautifulSoup(response.text, 'html.parser')

                # Extract title
                title = ""
                title_tag = soup.find('h1')
                if title_tag:
                    title = title_tag.get_text(strip=True)

                # Extract content from the topic-summary div (health topic pages)
                content_parts = []

                topic_summary = soup.find('div', {'id': 'topic-summary'})
                if topic_summary:
                    for child in topic_summary.children:
                        if not hasattr(child, 'name') or not child.name:
                            continue
                        if child.name == 'h3':
                            content_parts.append(f"\n{child.get_text(strip=True)}")
                        elif child.name in ['p', 'ul', 'ol']:
                            text = child.get_text(strip=True)
                            if text and len(text) > 20:
                                content_parts.append(text[:500])

                # Fallback: drug info pages use div.section with div.section-body
                if not content_parts:
                    sections = soup.find_all('div', class_='section')
                    for section in sections:
                        heading = section.find(['h2', 'h3'])
                        if heading:
                            content_parts.append(f"\n{heading.get_text(strip=True)}")
                        body = section.find('div', class_='section-body')
                        container = body if body else section
                        for elem in container.find_all(['p', 'ul', 'ol']):
                            text = elem.get_text(strip=True)
                            if text and len(text) > 20:
                                content_parts.append(text[:500])

                # Fallback: grab paragraphs directly
                if not content_parts:
                    paragraphs = soup.find_all('p', limit=10)
                    content_parts = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50]

                if content_parts:
                    content = '\n\n'.join(content_parts[:8])
                    result = {
                        'title': title,
                        'url': url,
                        'content': content[:3000]
                    }
                    self._cache[cache_key] = result
                    return result

            except Exception as e:
                continue

        return None

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """Rerank documents using the cross-encoder model."""
        if len(docs) <= self.TOP_K:
            return docs
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.cross_encoder.score(pairs)
        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[: self.TOP_K]]

    def _filter_relevant_content(
        self, content: str, query: str, max_tokens: int = 500,
        title: str = "", url: str = ""
    ) -> str:
        """Filter content to only include query-relevant parts using Milvus semantic search + cross-encoder reranking."""
        if not content:
            return ""

        doc = Document(
            page_content=content,
            metadata={"title": title, "url": url},
        )
        chunks = self.text_splitter.split_documents([doc])

        # Short-circuit: if few chunks, skip vector store overhead
        if len(chunks) <= self.TOP_K:
            return "\n\n".join(c.page_content for c in chunks)

        tmp_dir = tempfile.mkdtemp()

        try:
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=tmp_dir,
                collection_name="medlineplus_tool_filter",
            )
            # Fetch extra candidates for cross-encoder reranking
            fetch_k = min(len(chunks), self.RERANK_FETCH_K)
            candidate_docs = vectorstore.similarity_search(query, k=fetch_k)

            # Rerank with cross-encoder for final selection
            reranked = self._rerank(query, candidate_docs)

            return "\n\n".join(d.page_content for d in reranked)
        except Exception:
            return content[:800]
        finally:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except OSError:
                pass


# Global tools instance
_tools_instance = MedlinePlusTools()


@tool
def search_health_topic(topic: str) -> str:
    """
    Search MedlinePlus for information about a specific health topic.
    Use this when you need to find information about a disease, condition, or health concept.

    Args:
        topic: The health topic to search for. Use simple, common terms
               (e.g., "diabetes", "high blood pressure", "asthma", "back pain").
               Avoid adding extra qualifiers like "symptoms" or "treatment".

    Returns:
        Relevant health information from MedlinePlus (filtered for relevance)
    """
    result = _tools_instance._scrape_topic_page(topic)
    if result:
        filtered_content = _tools_instance._filter_relevant_content(
            result['content'], topic,
            title=result.get('title', ''), url=result.get('url', '')
        )
        return f"[Source: {result['title']} - {result['url']}]\n{filtered_content}"
    return f"No information found for topic: {topic}. Try a different search term."


@tool
def search_symptoms(symptoms: str) -> str:
    """
    Search MedlinePlus for information related to specific symptoms.
    Use this when the user describes symptoms and you need to find relevant information.

    Args:
        symptoms: The primary symptom to search for. Use simple terms
                  (e.g., "chest pain", "fever", "headache", "cough").
                  For multiple symptoms, use the single most prominent one.

    Returns:
        Relevant symptom information from MedlinePlus (filtered for relevance)
    """
    # Try the symptom as-is first (preserving spaces for search)
    result = _tools_instance._scrape_topic_page(symptoms)

    if not result:
        # Try with first keyword
        main_symptom = symptoms.split()[0] if symptoms else ""
        result = _tools_instance._scrape_topic_page(main_symptom)

    if result:
        filtered_content = _tools_instance._filter_relevant_content(
            result['content'], symptoms,
            title=result.get('title', ''), url=result.get('url', '')
        )
        return f"[Source: {result['title']} - {result['url']}]\n{filtered_content}"
    return f"No specific information found for: {symptoms}. Consider consulting a healthcare provider."


@tool
def search_treatment_info(condition: str) -> str:
    """
    Search for treatment information about a specific condition.
    Use this when the user asks about treatments, medications, or therapies.

    Args:
        condition: The condition name to find treatment info for. Use the simple
                   condition name only (e.g., "diabetes", "asthma", "arthritis").
                   Do NOT include the word "treatment" — this tool already filters
                   for treatment content.

    Returns:
        Treatment-related information from MedlinePlus (filtered for relevance)
    """
    result = _tools_instance._scrape_topic_page(condition)
    if result:
        filtered_content = _tools_instance._filter_relevant_content(
            result['content'], f"treatment therapy medication {condition}",
            title=result.get('title', ''), url=result.get('url', '')
        )
        return f"[Source: {result['title']} - {result['url']}]\n{filtered_content}"
    return f"No treatment information found for: {condition}."

@tool 
def fetch_url(url:str) -> str:
    """fetch text content from url"""
    response = requests.get(url,timeout=10.0)
    response.raise_for_status()
    return response.text

@tool
def decompose_query(query: str) -> str:
    """
    Analyze a complex health question to identify key medical concepts and plan
    an efficient multi-hop search strategy. Use this FIRST for questions involving
    multiple conditions, drug interactions, or combined medical factors
    (e.g., "Can I take X if I have Y and am on Z?").
    Do NOT use this for straightforward single-topic questions.

    Args:
        query: The user's complex health question to break down

    Returns:
        Identified concepts, query type, and suggested sequential search strategy
    """
    query_lower = query.lower()

    # Detect interaction patterns
    interaction_phrases = [
        'if i have', 'if i am', 'while on', 'and am on', 'while taking',
        'can i take', 'safe to take', 'interact', 'combine', 'together with',
        'along with', 'mixed with', 'compatible', 'is it safe', 'can i use',
        'should i avoid', 'conflict', 'taking it with', 'on top of',
    ]
    is_interaction = any(phrase in query_lower for phrase in interaction_phrases)

    # Clean common question prefixes so we can isolate the medical concepts
    cleaned = query_lower
    for prefix in ['can i ', 'should i ', 'is it safe to ', 'what happens if i ',
                   'what happens when i ', 'tell me about ', 'what are the ',
                   'how does ', 'what should i ', 'could i ', 'is it ok to ']:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]

    # Split on conjunctions / conditionals to extract concept chunks
    chunks = re.split(
        r'\s+(?:and|if|while|when|but|or|plus|as well as|in addition to)\s+',
        cleaned,
    )

    # Clean each chunk to isolate the medical concept
    concepts = []
    for chunk in chunks:
        chunk = re.sub(
            r'^(?:i\s+(?:have|had|am|take|use|get|suffer\s+from)\s+'
            r'|(?:am|being|im|i\'m)\s+(?:on|taking|using|diagnosed\s+with)\s+'
            r'|(?:take|taking|use|using|have|having)\s+)',
            '', chunk.strip(),
        )
        chunk = re.sub(r'[\?\.\!]+$', '', chunk).strip()
        if chunk and len(chunk) > 2:
            concepts.append(chunk)

    # Build structured analysis
    lines = []

    if is_interaction and len(concepts) >= 2:
        lines.append("QUERY TYPE: Drug/condition interaction — requires multi-hop search")
        lines.append(f"CONCEPTS IDENTIFIED ({len(concepts)}):")
        for i, concept in enumerate(concepts, 1):
            lines.append(f"  {i}. {concept}")
        lines.append("")
        primary = concepts[0]
        lines.append("SEARCH STRATEGY (sequential — check each interaction):")
        for i, secondary in enumerate(concepts[1:], 1):
            lines.append(
                f"  Step {i}: Search '{primary} {secondary}' for interaction/safety info"
            )
        lines.append("  Final: Synthesize all findings — highlight conflicts and safety concerns")

    elif len(concepts) >= 2:
        lines.append("QUERY TYPE: Multi-concept health question")
        lines.append(f"CONCEPTS IDENTIFIED ({len(concepts)}):")
        for i, concept in enumerate(concepts, 1):
            lines.append(f"  {i}. {concept}")
        lines.append("")
        lines.append("SEARCH STRATEGY:")
        for i, concept in enumerate(concepts, 1):
            lines.append(f"  Step {i}: Search '{concept}'")
        lines.append("  Final: Combine findings — identify overlapping advice or conflicts")

    else:
        lines.append("QUERY TYPE: Single-concept question — direct search is sufficient")
        if concepts:
            lines.append(f"  Search for: '{concepts[0]}'")

    return '\n'.join(lines)


SYSTEM_PROMPT = """You are a helpful healthcare information agent with access to MedlinePlus.

Your job is to answer healthcare questions by:
1. First assessing whether the user's query is clear and specific enough to search effectively
2. If the query is vague or ambiguous, asking targeted follow-up questions BEFORE searching
3. For complex multi-part questions, using decompose_query to plan a strategic search sequence
4. Using the appropriate tool(s) to search MedlinePlus for relevant information
5. Synthesizing the information into a helpful, accurate response

QUERY REFINEMENT — ASSESS BEFORE SEARCHING:
Before using any search tools, evaluate the user's query for clarity:
- Is it specific enough to produce useful search results?
- Are there critical missing details (e.g., which body part, type of pain, duration, age group)?
- Could the query mean multiple different things?

If the query is UNCLEAR or TOO VAGUE, do NOT search. Instead:
- Briefly acknowledge what you understand so far
- Ask 2-3 specific, targeted follow-up questions to narrow down their needs
- Keep your questions concise and easy to answer
- Examples of vague queries that need refinement:
  * "I don't feel well" → Ask about specific symptoms, duration, and severity
  * "pain" → Ask about location, type of pain, duration, and triggers
  * "medicine" → Ask what condition they need medicine for
  * "help" → Ask what health topic they need help with
  * "treatment" → Ask which condition they want treatment info for
  * "my kid is sick" → Ask about the child's age, specific symptoms, and how long

If the query IS clear and specific (e.g., "What are the symptoms of diabetes?",
"Tell me about high blood pressure treatment", "What causes migraines?"),
proceed directly to searching — do not ask unnecessary follow-up questions.

When the user replies to your follow-up questions, use their answers together with
the original query to perform a focused search.

MULTI-HOP REASONING — FOR COMPLEX QUESTIONS:
Some questions involve multiple interacting medical concepts (e.g., drug interactions,
medication safety with comorbidities, combined conditions). Handle these strategically:

Step 1 — DECOMPOSE: Use the decompose_query tool to break the question into its
  key medical concepts and get a structured search plan.

Step 2 — SEARCH SEQUENTIALLY: Execute searches ONE AT A TIME, following the plan.
  After EACH search, review the results before deciding your next search.
  - Start with the most safety-critical concept pair
  - Let findings from earlier searches inform later ones
  - Skip a planned search if prior results already answer that part

Step 3 — SYNTHESIZE: Combine ALL findings into a single cohesive answer that:
  - Addresses every part of the original question
  - Highlights interactions, conflicts, or compounding risks between concepts
  - Notes when one finding changes the relevance of another
  - Provides a clear, actionable summary

Examples of multi-hop queries:
- "Can I take ibuprofen if I have kidney disease and am on blood thinners?"
  → decompose → search ibuprofen+kidney disease → review (kidney risk!) →
    search ibuprofen+blood thinners → review (bleeding risk!) → synthesize both
- "I have diabetes and high blood pressure, what diet should I follow?"
  → decompose → search diabetic diet → search hypertension diet →
    synthesize overlapping advice
- "What are the side effects of metformin for someone with liver problems?"
  → decompose → search metformin side effects → search metformin+liver →
    synthesize with emphasis on liver-specific concerns

For multi-hop queries you may use up to 3 search tools (plus decompose_query).
For simple single-topic queries, 1 search is usually enough — do NOT over-search.

IMPORTANT GUIDELINES:
- Use tools SPARINGLY — match the number of searches to question complexity
-Use fetch_url when you need to fetch information from a web-page; quote relevant snippets.

- Always cite your sources from MedlinePlus
- Recommend consulting a healthcare professional for medical advice
- Be concise in your responses
- If you don't find relevant information, say so honestly

Available tools:
- decompose_query: Break down complex multi-part questions into a search strategy
- search_health_topic: For general health topics and conditions
- search_symptoms: For symptom-related queries
- search_treatment_info: For treatment and medication questions"""


class AgenticMedlinePlusRAG:
    """Wrapper around LangChain's create_agent for healthcare Q&A."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)

        # Create agent using LangChain's built-in create_agent
        self.agent = create_agent(
            model=self.llm,
            tools=[decompose_query, search_health_topic, search_symptoms, search_treatment_info, fetch_url],
            system_prompt=SYSTEM_PROMPT,
        )

        self.chat_history = []
        self.last_retrieval_debug = []

    def query(self, user_question: str) -> str:
        """Process a healthcare question using the agent."""
        try:
            # Build messages list with history + new question
            messages = list(self.chat_history[-4:]) + [
                HumanMessage(content=user_question)
            ]

            result = self.agent.invoke({"messages": messages})

            # Capture tool interactions for debug display
            debug_info = []
            for m in result["messages"]:
                if isinstance(m, AIMessage) and m.tool_calls:
                    for tc in m.tool_calls:
                        debug_info.append({
                            "type": "tool_call",
                            "tool": tc["name"],
                            "args": tc["args"],
                        })
                elif isinstance(m, ToolMessage):
                    debug_info.append({
                        "type": "tool_result",
                        "tool": m.name,
                        "content": m.content,
                    })
            self.last_retrieval_debug = debug_info

            # Extract the final AI response from returned messages
            ai_messages = [
                m for m in result["messages"]
                if isinstance(m, AIMessage) and m.content and not m.tool_calls
            ]
            response_text = ai_messages[-1].content if ai_messages else "No response generated."

            # Update chat history (keep it small for token efficiency)
            self.chat_history.append(HumanMessage(content=user_question))
            self.chat_history.append(AIMessage(content=response_text))

            # Trim history to last 4 messages
            if len(self.chat_history) > 4:
                self.chat_history = self.chat_history[-4:]

            return response_text

        except Exception as e:
            return f"I encountered an error while processing your question: {str(e)}. Please try rephrasing your question."

    def clear_history(self):
        """Clear chat history."""
        self.chat_history = []


def main():
    """Main function to run the Agentic RAG system."""
    print("=" * 60)
    print("MedlinePlus Agentic RAG Healthcare Assistant")
    print("=" * 60)
    print("I'm an AI agent that can search MedlinePlus to answer your")
    print("healthcare questions. I'll reason about what to search.")
    print("Type 'quit' to exit, 'clear' to reset conversation.\n")

    agent_rag = AgenticMedlinePlusRAG()

    while True:
        user_input = input("\nYour question: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! Remember to consult healthcare professionals for medical advice.")
            break

        if user_input.lower() == 'clear':
            agent_rag.clear_history()
            print("Conversation history cleared.")
            continue

        if not user_input:
            continue

        print("\nAgent is thinking and searching...\n")

        try:
            response = agent_rag.query(user_input)
            print("\n" + "=" * 40)
            print("Agent Response:")
            print("=" * 40)
            print(response)
            print("=" * 40)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
