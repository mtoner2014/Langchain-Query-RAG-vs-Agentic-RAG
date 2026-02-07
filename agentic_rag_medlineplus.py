"""
Agentic RAG System for MedlinePlus Healthcare Information
An agent that can reason about healthcare queries and use tools to retrieve information.
"""

import os
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Annotated
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()


class MedlinePlusTools:
    """Tools for the agent to interact with MedlinePlus."""

    BASE_URL = "https://medlineplus.gov"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Educational RAG System)'
        })
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,  # Small chunks for token efficiency
            chunk_overlap=30
        )
        # Cache to avoid repeated fetches
        self._cache = {}

    def _scrape_topic_page(self, topic: str) -> Optional[Dict]:
        """Scrape a health topic page."""
        # Check cache first
        cache_key = topic.lower()
        if cache_key in self._cache:
            return self._cache[cache_key]

        urls_to_try = [
            f"{self.BASE_URL}/{topic.lower().replace(' ', '')}.html",
            f"{self.BASE_URL}/healthtopics/{topic.lower().replace(' ', '')}.html",
            f"{self.BASE_URL}/druginfo/meds/a{topic.lower()[:3]}.html",
        ]

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

                # Extract content efficiently
                content_parts = []

                # Look for summary first (most important!)
                summary = soup.find(['div', 'section'], {'id': 'topic-summary'})
                if summary:
                    summary_text = summary.get_text(strip=True)[:800]
                    content_parts.append(f"Summary: {summary_text}")

                # Get key sections but limit content
                main = soup.find('article') or soup.find('main') or soup.find('div', {'role': 'main'})
                if main:
                    headers = main.find_all(['h2', 'h3'], limit=5)
                    for header in headers:
                        header_text = header.get_text(strip=True)
                        # Get next sibling paragraph
                        next_p = header.find_next('p')
                        if next_p:
                            p_text = next_p.get_text(strip=True)[:300]
                            content_parts.append(f"{header_text}: {p_text}")

                # Fallback
                if not content_parts:
                    paragraphs = soup.find_all('p', limit=5)
                    content_parts = [p.get_text(strip=True)[:300] for p in paragraphs if len(p.get_text()) > 30]

                if content_parts:
                    result = {
                        'title': title,
                        'url': url,
                        'content': '\n\n'.join(content_parts[:5])  # Max 5 sections
                    }
                    self._cache[cache_key] = result
                    return result

            except Exception as e:
                continue

        return None

    def _filter_relevant_content(self, content: str, query: str, max_tokens: int = 500) -> str:
        """Filter content to only include query-relevant parts (TOKEN EFFICIENT!)."""
        if not content:
            return ""

        # Split into chunks
        chunks = self.text_splitter.split_text(content)
        if not chunks:
            return content[:max_tokens * 4]  # Rough char estimate

        # Create documents
        docs = [Document(page_content=chunk) for chunk in chunks]

        # Use embeddings to find most relevant chunks
        try:
            temp_store = FAISS.from_documents(docs, self.embeddings)
            results = temp_store.similarity_search_with_score(query, k=2)

            # Take only high-relevance chunks
            relevant = []
            for doc, score in results:
                if score < 1.2:  # Relevance threshold
                    relevant.append(doc.page_content)

            if relevant:
                return '\n'.join(relevant)
        except Exception:
            pass

        # Fallback: return first chunk
        return chunks[0] if chunks else content[:max_tokens * 4]


# Global tools instance
_tools_instance = MedlinePlusTools()


@tool
def search_health_topic(topic: str) -> str:
    """
    Search MedlinePlus for information about a specific health topic.
    Use this when you need to find information about a disease, condition, or health concept.

    Args:
        topic: The health topic to search for (e.g., "diabetes", "hypertension", "asthma")

    Returns:
        Relevant health information from MedlinePlus (filtered for relevance)
    """
    result = _tools_instance._scrape_topic_page(topic)
    if result:
        # Filter to relevant content only (TOKEN EFFICIENT!)
        filtered_content = _tools_instance._filter_relevant_content(
            result['content'], topic, max_tokens=400
        )
        return f"[Source: {result['title']} - {result['url']}]\n{filtered_content}"
    return f"No information found for topic: {topic}. Try a different search term."


@tool
def search_symptoms(symptoms: str) -> str:
    """
    Search MedlinePlus for information related to specific symptoms.
    Use this when the user describes symptoms and you need to find relevant information.

    Args:
        symptoms: Description of symptoms (e.g., "chest pain", "fever and cough")

    Returns:
        Relevant symptom information from MedlinePlus (filtered for relevance)
    """
    # Extract main symptom keyword
    main_symptom = symptoms.split()[0] if symptoms else ""
    result = _tools_instance._scrape_topic_page(symptoms.replace(' ', ''))

    if not result:
        # Try with first keyword
        result = _tools_instance._scrape_topic_page(main_symptom)

    if result:
        filtered_content = _tools_instance._filter_relevant_content(
            result['content'], symptoms, max_tokens=400
        )
        return f"[Source: {result['title']} - {result['url']}]\n{filtered_content}"
    return f"No specific information found for: {symptoms}. Consider consulting a healthcare provider."


@tool
def search_treatment_info(condition: str) -> str:
    """
    Search for treatment information about a specific condition.
    Use this when the user asks about treatments, medications, or therapies.

    Args:
        condition: The condition to find treatment info for (e.g., "diabetes treatment")

    Returns:
        Treatment-related information from MedlinePlus (filtered for relevance)
    """
    result = _tools_instance._scrape_topic_page(condition)
    if result:
        # Filter specifically for treatment-related content
        filtered_content = _tools_instance._filter_relevant_content(
            result['content'], f"treatment therapy medication {condition}", max_tokens=400
        )
        return f"[Source: {result['title']} - {result['url']}]\n{filtered_content}"
    return f"No treatment information found for: {condition}."


SYSTEM_PROMPT = """You are a helpful healthcare information agent with access to MedlinePlus.

Your job is to answer healthcare questions by:
1. Analyzing the user's question to understand what information they need
2. Using the appropriate tool(s) to search MedlinePlus for relevant information
3. Synthesizing the information into a helpful, accurate response

IMPORTANT GUIDELINES:
- Use tools SPARINGLY - only search for what's truly needed (1-2 searches max)
- If the question is simple, one search is usually enough
- Always cite your sources from MedlinePlus
- Recommend consulting a healthcare professional for medical advice
- Be concise in your responses
- If you don't find relevant information, say so honestly

Available tools:
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
            tools=[search_health_topic, search_symptoms, search_treatment_info],
            system_prompt=SYSTEM_PROMPT,
        )

        self.chat_history = []

    def query(self, user_question: str) -> str:
        """Process a healthcare question using the agent."""
        try:
            # Build messages list with history + new question
            messages = list(self.chat_history[-4:]) + [
                HumanMessage(content=user_question)
            ]

            result = self.agent.invoke({"messages": messages})

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
