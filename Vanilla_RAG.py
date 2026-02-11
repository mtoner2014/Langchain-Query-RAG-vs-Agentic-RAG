"""
Vanilla RAG System for MedlinePlus Healthcare Information
Simple retrieve-and-generate pipeline without query expansion.
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
from langchain_community.document_compressors import FlashrankRerank
from flashrank import Ranker

FlashrankRerank.model_rebuild(_types_namespace={"Ranker": Ranker})

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
            seen = set()
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'v:frame=redirect' in href or 'v%3aframe=redirect' in href:
                    parsed = urlparse(href)
                    qs = parse_qs(parsed.query)
                    target = qs.get('url', [None])[0]
                    if target:
                        target = unquote(target)
                        if re.match(r'https://medlineplus\.gov/(?:[a-zA-Z]+|druginfo/\w+/\w+)\.html$', target):
                            if target not in seen:
                                seen.add(target)
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

            content = '\n\n'.join(content_parts[:8])

            return {
                'title': title,
                'url': url,
                'content': content[:3000]
            }

        except Exception as e:
            print(f"Scraping error for {url}: {e}")
            return None

    def fetch_health_info(self, query: str) -> List[Dict]:
        """Fetch health information by searching the raw query directly."""
        results = []
        seen_urls = set()

        search_urls = self.search_topics(query)
        for url in search_urls:
            if url not in seen_urls:
                page_data = self.scrape_page(url)
                if page_data and page_data['content']:
                    results.append(page_data)
                    seen_urls.add(url)

        return results


class VanillaRAG:
    """Vanilla RAG system for healthcare Q&A using MedlinePlus.

    Uses LangChain LCEL retrieval pipeline with:
    - sentence-transformers/all-MiniLM-L6-v2 for embeddings
    - Chroma as the ephemeral vector store
    - FlashRank for reranking retrieved chunks
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
        self.reranker = FlashrankRerank(
            top_n=self.TOP_K,
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

        if len(chunks) <= self.TOP_K:
            return None, chunks

        tmp_dir = tempfile.mkdtemp()

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=tmp_dir,
            collection_name="medlineplus_vanilla",
        )

        fetch_k = min(len(chunks), self.RERANK_FETCH_K)
        retriever = vectorstore.as_retriever(search_kwargs={"k": fetch_k})
        return retriever, tmp_dir

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """Rerank documents using FlashRank."""
        if len(docs) <= self.TOP_K:
            return docs
        return self.reranker.compress_documents(documents=docs, query=query)

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
        """Answer a healthcare question using Vanilla RAG."""
        result, tmp_path_or_chunks = self._build_retriever(user_question)

        if result is None and tmp_path_or_chunks is None:
            return (
                "I couldn't find relevant information on MedlinePlus for your question. "
                "Please try rephrasing or consult a healthcare professional."
            )

        try:
            if result is None:
                docs = tmp_path_or_chunks
                tmp_path = None
            else:
                retriever = result
                tmp_path = tmp_path_or_chunks
                docs = retriever.invoke(user_question)
                docs = self._rerank(user_question, docs)

            context, sources = self._format_chunks(docs)

            prompt = ChatPromptTemplate.from_template(
                "You are a knowledgeable healthcare information assistant.\n"
                "Use the following context from MedlinePlus to answer the "
                "user's question in detail.\n\n"
                "Instructions:\n"
                "- Provide specific, evidence-based information drawn from the context.\n"
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
    """Main function to run the Vanilla RAG system."""
    print("=" * 60)
    print("MedlinePlus Vanilla RAG Healthcare Assistant")
    print("=" * 60)
    print("Ask healthcare questions and get answers from MedlinePlus.")
    print("Type 'quit' to exit.\n")

    rag = VanillaRAG()

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
