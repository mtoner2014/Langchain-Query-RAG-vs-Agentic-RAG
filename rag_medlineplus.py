"""
RAG System for MedlinePlus Healthcare Information
Retrieves relevant healthcare information and augments LLM responses.
"""

import re
import requests
from urllib.parse import urlparse, parse_qs, unquote
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from dotenv import load_dotenv

import os
import shutil
import tempfile

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder



load_dotenv()


class MedlinePlusScraper:
    """Scrapes health information from MedlinePlus website."""

    BASE_URL = "https://medlineplus.gov"
    SEARCH_URL = "https://vsearch.nlm.nih.gov/vivisimo/cgi-bin/query-meta"

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

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Educational RAG System)'
        })

    @staticmethod
    def _relevance_score(query: str, link_text: str, url: str) -> float:
        """Score how relevant a search result is to the query (higher is better)."""
        query_words = set(query.lower().split())
        link_words = set(link_text.lower().split())
        # Extract the slug from the URL (e.g. "highbloodpressure" from topic URLs,
        # or "a682159" from drug URLs like /druginfo/meds/a682159.html)
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
            alias_slug = MedlinePlusScraper.TOPIC_ALIASES.get(qw)
            if alias_slug and alias_slug == slug:
                score += 5.0

        # Penalty for very generic pages (these tend to be peripheral)
        generic_slugs = {'heartdiseases', 'heartdisease', 'healthtopics'}
        if slug in generic_slugs:
            score -= 2.0

        return score

    def search_topics(self, query: str, max_results: int = 3) -> List[str]:
        """Search MedlinePlus and return relevant page URLs, ranked by relevance."""
        params = {
            'v:project': 'medlineplus',
            'v:sources': 'medlineplus-bundle',
            'query': query,
        }

        try:
            response = self.session.get(self.SEARCH_URL, params=params, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            candidates = []  # (url, link_text) pairs
            seen = set()
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

            # Rank candidates by relevance to the query
            if candidates:
                candidates.sort(
                    key=lambda c: self._relevance_score(query, c[1], c[0]),
                    reverse=True,
                )
                urls = [c[0] for c in candidates[:max_results]]
            else:
                urls = []

            # Fallback: construct direct topic URL
            if not urls:
                topic_slug = query.lower().replace(' ', '').replace('-', '')
                urls = [f"{self.BASE_URL}/{topic_slug}.html"]

            return urls
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def get_topic_url(self, topic: str) -> str:
        """Construct a direct URL for a health topic, checking aliases first."""
        key = topic.lower().strip()
        if key in self.TOPIC_ALIASES:
            return f"{self.BASE_URL}/{self.TOPIC_ALIASES[key]}.html"
        # Check if any known alias appears as a whole phrase within the topic
        # (e.g. "hypertension risk factors" contains alias "hypertension")
        for alias in sorted(self.TOPIC_ALIASES, key=len, reverse=True):
            if re.search(r'\b' + re.escape(alias) + r'\b', key):
                return f"{self.BASE_URL}/{self.TOPIC_ALIASES[alias]}.html"
        topic_slug = key.replace(' ', '').replace('-', '')
        return f"{self.BASE_URL}/{topic_slug}.html"

    def scrape_page(self, url: str) -> Optional[Dict]:
        """Scrape a MedlinePlus page and extract relevant content."""
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return None

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

            # Fallback: get all paragraphs but limit them
            if not content_parts:
                paragraphs = soup.find_all('p', limit=10)
                content_parts = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50]

            content = '\n\n'.join(content_parts[:8])  # Limit to 8 sections max

            return {
                'title': title,
                'url': url,
                'content': content[:3000]  # Hard limit on content length for token efficiency
            }

        except Exception as e:
            print(f"Scraping error for {url}: {e}")
            return None

    def _extract_topics(self, query: str) -> List[str]:
        """Extract individual health topics from a natural language query."""
        text = query.lower().strip()
        text = re.sub(r'[?.,!]', '', text)

        # Keep "and its/their/the" as part of the same topic — these are
        # qualifiers, not separate topics (e.g. "hypertension and its risk
        # factors" should stay together, not split into two topics)
        text = re.sub(r'\s+and\s+(?=(?:its|their|the)\s+)', ' ', text)

        # Split on remaining "and" / "or" connectors
        parts = re.split(r'\s+and\s+|\s+or\s+', text)

        # Strip common non-medical words from each part
        stop_words = {
            'what', 'how', 'why', 'should', 'can', 'could', 'would', 'do',
            'does', 'if', 'i', 'have', 'had', 'has', 'am', 'about', 'the',
            'a', 'an', 'my', 'is', 'are', 'it', 'its', 'to', 'for', 'with',
            'in', 'on', 'of', 'be', 'been', 'being', 'get', 'getting',
            'when', 'where', 'me', 'we', 'they', 'their', 'tell', 'know',
            'explain', 'describe', 'cause', 'causes',
        }

        topics = []
        for part in parts:
            words = [w for w in part.split() if w not in stop_words]
            topic = ' '.join(words).strip()
            if len(topic) > 2:
                topics.append(topic)

        return topics if topics else [query]

    def fetch_health_info(self, query: str) -> List[Dict]:
        """Fetch health information for a query."""
        topics = self._extract_topics(query)
        results = []
        seen_urls = set()

        for topic in topics:
            # Try direct topic URL first (fast path for exact matches)
            direct_url = self.get_topic_url(topic)
            if direct_url not in seen_urls:
                page_data = self.scrape_page(direct_url)
                if page_data and page_data['content']:
                    results.append(page_data)
                    seen_urls.add(direct_url)
                    continue

            # Direct URL failed — search MedlinePlus for the right page
            search_urls = self.search_topics(topic)
            for url in search_urls:
                if url not in seen_urls:
                    page_data = self.scrape_page(url)
                    if page_data and page_data['content']:
                        results.append(page_data)
                        seen_urls.add(url)
                        break

        # For multi-topic queries, also search the combined query for cross-topic pages
        if len(topics) > 1:
            combined_urls = self.search_topics(query, max_results=2)
            for url in combined_urls:
                if url not in seen_urls:
                    page_data = self.scrape_page(url)
                    if page_data and page_data['content']:
                        results.append(page_data)
                        seen_urls.add(url)

        return results


class MedlinePlusRAG:
    """RAG system for healthcare Q&A using MedlinePlus.

    Uses LangChain LCEL retrieval pipeline with:
    - sentence-transformers/all-MiniLM-L6-v2 for embeddings (local, no GPU needed)
    - Chroma as the ephemeral vector store
    - cross-encoder/ms-marco-MiniLM-L-6-v2 for reranking retrieved chunks
    """

    TOP_K = 4
    RERANK_FETCH_K = 10

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.scraper = MedlinePlusScraper()
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
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

        self.last_retrieval_debug = []

    def _build_retriever(self, query: str):
        """Scrape MedlinePlus, chunk, embed into Chroma, and return a retriever.

        Returns:
            Tuple of (retriever | None, tmp_dir_path | chunks_list | None).
        """
        health_info = self.scraper.fetch_health_info(query)
        self.last_retrieval_debug = health_info

        if not health_info:
            return None, None

        documents = []
        for info in health_info:
            if info and info.get('content'):
                documents.append(Document(
                    page_content=info['content'],
                    metadata={'title': info.get('title', ''), 'url': info.get('url', '')},
                ))
        if not documents:
            return None, None

        chunks = self.text_splitter.split_documents(documents)

        # Short-circuit: if few chunks, skip vector store overhead
        if len(chunks) <= self.TOP_K:
            return None, chunks

        # Create ephemeral Chroma store in a temp directory
        tmp_dir = tempfile.mkdtemp()

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=tmp_dir,
            collection_name="medlineplus_filter",
        )

        # Fetch extra candidates for cross-encoder reranking
        fetch_k = min(len(chunks), self.RERANK_FETCH_K)
        retriever = vectorstore.as_retriever(search_kwargs={"k": fetch_k})
        return retriever, tmp_dir

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """Rerank documents using the cross-encoder model."""
        if len(docs) <= self.TOP_K:
            return docs
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.cross_encoder.score(pairs)
        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[: self.TOP_K]]

    @staticmethod
    def _format_chunks(docs: List[Document]) -> tuple[str, List[Dict]]:
        """Format Documents into a context string and deduplicated source list."""
        sources: List[Dict] = []
        seen_urls: set = set()
        context_parts: List[str] = []

        for doc in docs:
            url = doc.metadata.get("url", "")
            title = doc.metadata.get("title", "MedlinePlus")
            if url and url not in seen_urls:
                sources.append({"title": title, "url": url})
                seen_urls.add(url)
            context_parts.append(f"[Source: {title}]\n{doc.page_content}")

        return "\n\n---\n\n".join(context_parts), sources

    def query(self, user_question: str) -> str:
        """Answer a healthcare question using RAG."""
        result, tmp_path_or_chunks = self._build_retriever(user_question)

        # No content found at all
        if result is None and tmp_path_or_chunks is None:
            return (
                "I couldn't find relevant information on MedlinePlus for your question. "
                "Please try rephrasing or consult a healthcare professional."
            )

        try:
            # Determine docs: either short-circuited chunks or retriever results
            if result is None:
                # Short-circuit path — few chunks, no vector store needed
                docs = tmp_path_or_chunks
                tmp_path = None
            else:
                # Retriever path — invoke retriever then rerank
                retriever = result
                tmp_path = tmp_path_or_chunks
                docs = retriever.invoke(user_question)
                docs = self._rerank(user_question, docs)

            context, sources = self._format_chunks(docs)

            # LCEL RAG chain: retriever context → prompt → LLM → parse
            prompt = ChatPromptTemplate.from_template(
                "You are a knowledgeable healthcare information assistant.\n"
                "Use the following context from MedlinePlus to answer the "
                "user's question in detail.\n\n"
                "Instructions:\n"
                "- Provide specific, evidence-based information drawn from the context.\n"
                "- When the question involves multiple conditions or topics, dedicate a "
                "section to each and then explain how they may interact or influence "
                "each other.\n"
                "- Use clear headings or bullet points to organize the response.\n"
                "- Include relevant details such as symptoms, causes, treatment options, "
                "and lifestyle considerations.\n"
                "- If the context doesn't fully cover the question, clearly state what "
                "is and isn't covered.\n"
                "- Always recommend consulting a healthcare professional for personalized "
                "medical advice.\n\n"
                "Context from MedlinePlus:\n{context}\n\n"
                "User Question: {question}\n\nDetailed Answer:"
            )

            chain = (
                {"context": lambda _: context, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )

            response = chain.invoke(user_question)

            # Append source links
            if sources:
                response += "\n\n**Relevant MedlinePlus Articles:**"
                for source in sources:
                    response += f"\n- [{source['title']}]({source['url']})"

            return response
        finally:
            if result is not None and tmp_path_or_chunks:
                try:
                    shutil.rmtree(tmp_path_or_chunks, ignore_errors=True)
                except OSError:
                    pass


def main():
    """Main function to run the RAG system."""
    print("=" * 60)
    print("MedlinePlus RAG Healthcare Assistant")
    print("=" * 60)
    print("Ask healthcare questions and get answers from MedlinePlus.")
    print("Type 'quit' to exit.\n")

    rag = MedlinePlusRAG()

    while True:
        user_input = input("\nYour question: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not user_input:
            continue

        print("\nSearching MedlinePlus and generating response...\n")

        try:
            response = rag.query(user_input)
            print("-" * 40)
            print("Answer:")
            print("-" * 40)
            print(response)
            print("-" * 40)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
