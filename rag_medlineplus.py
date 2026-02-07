"""
RAG System for MedlinePlus Healthcare Information
Retrieves relevant healthcare information and augments LLM responses.
"""

import os
import re
import requests
from urllib.parse import urlparse, parse_qs, unquote
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class MedlinePlusScraper:
    """Scrapes health information from MedlinePlus website."""

    BASE_URL = "https://medlineplus.gov"
    SEARCH_URL = "https://vsearch.nlm.nih.gov/vivisimo/cgi-bin/query-meta"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Educational RAG System)'
        })

    def search_topics(self, query: str, max_results: int = 3) -> List[str]:
        """Search MedlinePlus and return relevant page URLs."""
        params = {
            'v:project': 'medlineplus',
            'v:sources': 'medlineplus-bundle',
            'query': query,
        }

        try:
            response = self.session.get(self.SEARCH_URL, params=params, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            urls = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                # Search results use redirect URLs with the real URL in a 'url' param
                if 'v:frame=redirect' in href or 'v%3aframe=redirect' in href:
                    parsed = urlparse(href)
                    qs = parse_qs(parsed.query)
                    target = qs.get('url', [None])[0]
                    if target:
                        target = unquote(target)
                        if re.match(r'https://medlineplus\.gov/[a-zA-Z]+\.html$', target):
                            if target not in urls:
                                urls.append(target)
                                if len(urls) >= max_results:
                                    break

            # Fallback: construct direct topic URL
            if not urls:
                topic_slug = query.lower().replace(' ', '').replace('-', '')
                urls = [f"{self.BASE_URL}/{topic_slug}.html"]

            return urls
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def get_topic_url(self, topic: str) -> str:
        """Construct a direct URL for a health topic."""
        topic_slug = topic.lower().replace(' ', '').replace('-', '')
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

            # Extract content from the topic-summary div
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

    def fetch_health_info(self, query: str) -> List[Dict]:
        """Fetch health information for a query."""
        results = []

        # Try direct topic URL first (fast path for exact matches)
        direct_url = self.get_topic_url(query)
        page_data = self.scrape_page(direct_url)
        if page_data and page_data['content']:
            results.append(page_data)
            return results

        # Direct URL failed — search MedlinePlus for the right page
        search_urls = self.search_topics(query)
        for url in search_urls:
            if url != direct_url:
                page_data = self.scrape_page(url)
                if page_data and page_data['content']:
                    results.append(page_data)
                    if len(results) >= 2:
                        break

        return results


class MedlinePlusRAG:
    """RAG system for healthcare Q&A using MedlinePlus."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.scraper = MedlinePlusScraper()
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Small chunks for efficiency
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " "]
        )
        self.vector_store = None

    def _create_documents(self, health_info: List[Dict]) -> List[Document]:
        """Convert scraped info to LangChain documents."""
        documents = []
        for info in health_info:
            if info and info.get('content'):
                doc = Document(
                    page_content=info['content'],
                    metadata={
                        'title': info.get('title', ''),
                        'source': info.get('url', '')
                    }
                )
                documents.append(doc)
        return documents

    def _filter_chunks(self, chunks: List[Document], query: str, top_k: int = 3) -> List[Document]:
        """Filter chunks to only the most relevant ones (TOKEN EFFICIENT!)."""
        if not chunks:
            return []

        # Create temporary vector store for similarity search
        temp_store = FAISS.from_documents(chunks, self.embeddings)

        # Get only top_k most relevant chunks
        relevant_chunks = temp_store.similarity_search_with_score(query, k=top_k)

        # Filter by similarity score threshold (lower is better for FAISS L2 distance)
        filtered = []
        for doc, score in relevant_chunks:
            if score < 1.5:  # Only include highly relevant chunks
                filtered.append(doc)

        return filtered if filtered else [relevant_chunks[0][0]]  # At least return best match

    def build_context(self, query: str) -> str:
        """Build context from MedlinePlus for the query."""
        # Fetch health information
        health_info = self.scraper.fetch_health_info(query)

        if not health_info:
            return ""

        # Convert to documents
        documents = self._create_documents(health_info)

        if not documents:
            return ""

        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)

        # Filter to most relevant chunks (TOKEN EFFICIENT!)
        relevant_chunks = self._filter_chunks(chunks, query, top_k=3)

        # Build context string
        context_parts = []
        for chunk in relevant_chunks:
            source = chunk.metadata.get('title', 'MedlinePlus')
            context_parts.append(f"[Source: {source}]\n{chunk.page_content}")

        return "\n\n---\n\n".join(context_parts)

    def query(self, user_question: str) -> str:
        """Answer a healthcare question using RAG."""

        # Build context from MedlinePlus
        context = self.build_context(user_question)

        # Create prompt template
        prompt = ChatPromptTemplate.from_template("""You are a helpful healthcare information assistant.
Use the following context from MedlinePlus to answer the user's question.
If the context doesn't contain relevant information, say so and provide general guidance.
Always recommend consulting a healthcare professional for medical advice.

Context from MedlinePlus:
{context}

User Question: {question}

Answer (be concise and helpful):""")

        # Create chain
        chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Get response
        response = chain.invoke(user_question)

        return response


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
