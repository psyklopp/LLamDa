"""
Diary RAG with full Ollama support and all capabilities.

"""

from __future__ import annotations as _annotations

import asyncio
import json
import os
import re
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import asyncpg
import httpx
import logfire
import pydantic_core
from pydantic import BaseModel
from typing_extensions import AsyncGenerator

from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

# Configure logfire
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_asyncpg()
logfire.instrument_pydantic_ai()


class OllamaClient:
    """Simple Ollama client for embeddings and chat."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        
    async def create_embedding(self, text: str, model: str = "nomic-embed-text") -> List[float]:
        """Create embedding using Ollama."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": model, "prompt": text}
            )
            response.raise_for_status()
            return response.json()["embedding"]
    
    async def chat_completion(self, messages: List[dict], model: str = "qwen3:8b", temperature: float = 0.1) -> str:
        """Chat completion using Ollama."""
        # Convert messages to a single prompt for Ollama
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt = "\n\n".join(prompt_parts) + "\n\nAssistant: "
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature}
                }
            )
            response.raise_for_status()
            return response.json()["response"]


@dataclass
class Deps:
    ollama: OllamaClient
    pool: asyncpg.Pool


class DiaryEntry(BaseModel):
    """Represents a diary entry section."""
    filepath: str
    date_str: str
    title: str
    content: str
    section_num: int = 0
    
    def get_url(self) -> str:
        """Generate a unique identifier for this diary entry."""
        return f"diary://{self.filepath}#{self.section_num}"
    
    def embedding_content(self) -> str:
        """Content to use for embedding generation."""
        return '\n\n'.join([
            f'Date: {self.date_str}',
            f'Title: {self.title}',
            self.content
        ])


ollama_model = OpenAIChatModel(
    model_name='ollama:qwen3:8b',
    provider=OllamaProvider(base_url='http://localhost:11434/v1'),
)

# Create agent with Ollama support
agent = Agent(
    ollama_model,  # This won't work directly, we'll handle it manually
    deps_type=Deps,
    system_prompt="""You are a helpful AI assistant that helps users understand their diary entries and thoughts. 

When responding to queries about diary content:
1. Be personal and understanding - these are the user's private thoughts
2. Help identify patterns, connections, and insights
3. Provide thoughtful analysis and suggestions
4. Reference specific diary entries when relevant
5. Be encouraging and supportive in your responses

Always use the diary search tool to find relevant entries before responding."""
)


async def retrieve_diary_entries(deps: Deps, search_query: str) -> str:
    """Retrieve diary entries based on a search query.
    
    Args:
        deps: The dependencies (ollama client and database pool).
        search_query: The search query about diary content.
    """
    with logfire.span('create embedding for diary search', search_query=search_query):
        # Get embedding using Ollama
        embedding_vector = await deps.ollama.create_embedding(search_query)
        embedding_json = pydantic_core.to_json(embedding_vector).decode()
        
        # Search diary entries
        rows = await deps.pool.fetch(
            'SELECT url, date_str, title, content FROM diary_entries ORDER BY embedding <-> $1 LIMIT 5',
            embedding_json,
        )
        
        if not rows:
            return "No relevant diary entries found for your query."
        
        return '\n\n'.join(
            f'# {row["title"]} ({row["date_str"]})\n{row["content"]}'
            for row in rows
        )


async def run_agent(question: str):
    """Entry point to run the agent and perform RAG-based diary questioning."""
    # Initialize Ollama client
    ollama = OllamaClient()
    
    # Test Ollama connection
    try:
        async with httpx.AsyncClient() as client:
            await client.get("http://localhost:11434/api/tags", timeout=5.0)
    except Exception as e:
        print("❌ Cannot connect to Ollama. Please ensure:")
        print("1. Ollama is installed: curl -fsSL https://ollama.ai/install.sh | sh")
        print("2. Models are pulled: ollama pull qwen3:8b && ollama pull nomic-embed-text")
        print("3. Ollama is running: ollama serve")
        return
    
    logfire.info('Asking diary question with Ollama', question=question)
    
    async with database_connect(False) as pool:
        deps = Deps(ollama=ollama, pool=pool)
        
        # Search for relevant diary entries
        search_results = await retrieve_diary_entries(deps, question)
        
        # Then ask Ollama to respond based on the results
        messages = [
            {
                "role": "system", 
                "content": """You are a helpful AI assistant that helps users understand their diary entries and thoughts. 

When responding to queries about diary content:
1. Be personal and understanding - these are the user's private thoughts
2. Help identify patterns, connections, and insights  
3. Provide thoughtful analysis and suggestions
4. Reference specific diary entries when relevant
5. Be encouraging and supportive in your responses"""
            },
            {
                "role": "user",
                "content": f"""Based on these diary entries, please answer the question: "{question}"

Relevant diary entries:
{search_results}

Please provide a thoughtful, personal response that helps the user understand their thoughts and patterns."""
            }
        ]
        
        response = await ollama.chat_completion(messages)
        print(response)


async def chat_with_diary():
    """Interactive chat mode with diary."""
    ollama = OllamaClient()
    
    # Test connection
    try:
        async with httpx.AsyncClient() as client:
            await client.get("http://localhost:11434/api/tags", timeout=5.0)
    except Exception:
        print("❌ Cannot connect to Ollama. Please start with: ollama serve")
        return
    
    print("💬 Chat with your diary! Type 'quit' to exit.")
    print("Examples: 'What have I been thinking about?', 'How have my moods changed?'")
    
    async with database_connect(False) as pool:
        deps = Deps(ollama=ollama, pool=pool)
        conversation_history = []
        
        while True:
            try:
                question = input("\n🔍 You: ").strip()
                if question.lower() in ['quit', 'exit', 'bye']:
                    break
                
                if not question:
                    continue
                
                # Search for relevant entries
                search_results = await retrieve_diary_entries(deps, question)
                
                # Build conversation context
                messages = [
                    {
                        "role": "system",
                        "content": """You are a thoughtful AI companion helping someone understand their diary entries. Be personal, insightful, and supportive. Help them see patterns and connections in their thoughts. Make sure you follow user instructions."""
                    }
                ]
                
                # Add conversation history (last 4 exchanges)
                for hist in conversation_history[-4:]:
                    messages.append(hist)
                
                # Add current question with diary context
                messages.append({
                    "role": "user",
                    "content": f"""Question: {question}
                    Relevant diary entries:
                    {search_results}
                    Please provide a thoughtful response based on the diary content."""
                })
                print("\n🤖 Assistant: ", end="")
                response = await ollama.chat_completion(messages)
                logfire.info("\n - Response generated - \n")
                print(response)
                
                # Save to conversation history
                conversation_history.append({"role": "user", "content": question})
                conversation_history.append({"role": "assistant", "content": response})
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    print("\n👋 Goodbye! Thanks for chatting with your diary.")


async def build_search_db(diary_folder: str = "diary"):
    """Build the search database from markdown diary files using Ollama."""
    # Initialize Ollama
    ollama = OllamaClient()
    
    # Test Ollama connection
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags", timeout=5.0)
            models = response.json()["models"]
            model_names = [model["name"] for model in models]
            
            if "nomic-embed-text:latest" not in model_names:
                print("❌ nomic-embed-text model not found!")
                print("Please run: ollama pull nomic-embed-text")
                return
                
            if not any("qwen" in name for name in model_names):
                print("❌ No Qwen/Llama model found!")
                print("Please run: ollama pull qwen3:8b")
                return
                
    except Exception as e:
        print("❌ Cannot connect to Ollama. Please ensure:")
        print("1. Ollama is running: ollama serve")
        print("2. Models are available: ollama list")
        return
    
    diary_path = Path(diary_folder)
    
    if not diary_path.exists():
        print(f"❌ Diary folder '{diary_folder}' not found!")
        print("Please create a 'diary' folder with your markdown files, or specify a different path.")
        print("Example:")
        print("  mkdir diary")
        print("  echo '# Today I learned about RAG systems' > diary/2024-01-15.md")
        return
    
    # Find all markdown files
    md_files = list(diary_path.glob("*.md")) + list(diary_path.glob("**/*.md"))
    
    if not md_files:
        print(f"❌ No markdown files found in '{diary_folder}'!")
        print("Please add some .md files to your diary folder.")
        return
    
    print(f"📚 Found {len(md_files)} markdown files to process...")
    
    # Extract diary entries from markdown files
    diary_entries = []
    for md_file in md_files:
        entries = extract_diary_entries(md_file)
        diary_entries.extend(entries)
    
    if not diary_entries:
        print("❌ No diary entries extracted from markdown files!")
        print("Make sure your .md files contain some text content.")
        return
    
    print(f"📝 Extracted {len(diary_entries)} diary entries...")
    
    # Build the database
    async with database_connect(True) as pool:
        with logfire.span('create schema'):
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(DB_SCHEMA)
        
        # Use semaphore to limit concurrent requests
        print("🤖 Generating embeddings with Ollama...")
        sem = asyncio.Semaphore(3)  # Reduced for local LLM
        
        async with asyncio.TaskGroup() as tg:
            for entry in diary_entries:
                tg.create_task(insert_diary_entry(sem, ollama, pool, entry))
        
        print("✅ Database built successfully!")
        print(f"📊 Stored {len(diary_entries)} entries with embeddings")


def extract_diary_entries(filepath: Path) -> List[DiaryEntry]:
    """Extract diary entries from a markdown file."""
    try:
        content = filepath.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        print(f"⚠️  Warning: Could not read {filepath} as UTF-8, skipping...")
        return []
    
    entries = []
    
    # Try to extract date from filename (common formats: YYYY-MM-DD, YYYY_MM_DD, etc.)
    date_from_filename = extract_date_from_filename(filepath.name)
    
    # Split content by headers or use the whole file as one entry
    sections = split_markdown_content(content)
    
    for i, (title, section_content) in enumerate(sections):
        if section_content.strip():  # Only add non-empty sections
            entry = DiaryEntry(
                filepath=str(filepath.relative_to(filepath.parent.parent) if filepath.parent.parent.exists() else filepath),
                date_str=date_from_filename or "Unknown date",
                title=title or filepath.stem,
                content=section_content.strip(),
                section_num=i
            )
            entries.append(entry)
    
    return entries


def extract_date_from_filename(filename: str) -> str | None:
    """Extract date from filename using common patterns."""
    # Remove extension
    name = Path(filename).stem
    
    # Common date patterns
    patterns = [
        r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
        r'(\d{4}_\d{2}_\d{2})',  # YYYY_MM_DD
        r'(\d{2}-\d{2}-\d{4})',  # MM-DD-YYYY
        r'(\d{2}_\d{2}_\d{4})',  # MM_DD_YYYY
        r'(\d{4}\d{2}\d{2})',    # YYYYMMDD
    ]
    
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            date_str = match.group(1)
            # Normalize to YYYY-MM-DD format
            date_str = date_str.replace('_', '-')
            try:
                # Try to parse and reformat
                if len(date_str) == 8:  # YYYYMMDD
                    date_obj = datetime.strptime(date_str, '%Y%m%d')
                elif date_str.count('-') == 2:
                    parts = date_str.split('-')
                    if len(parts[0]) == 4:  # YYYY-MM-DD
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    else:  # MM-DD-YYYY
                        date_obj = datetime.strptime(date_str, '%m-%d-%Y')
                    
                return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                continue
    
    return None


def split_markdown_content(content: str) -> List[tuple[str, str]]:
    """Split markdown content into sections based on headers."""
    lines = content.split('\n')
    sections = []
    current_title = None
    current_content = []
    
    for line in lines:
        # Check if it's a header
        if line.strip().startswith('#'):
            # Save previous section if exists
            if current_content:
                sections.append((current_title, '\n'.join(current_content)))
                current_content = []
            
            # Extract title
            current_title = line.strip('#').strip()
        else:
            current_content.append(line)
    
    # Add the last section
    if current_content:
        sections.append((current_title, '\n'.join(current_content)))
    
    # If no headers found, treat entire content as one section
    if not sections and content.strip():
        sections.append((None, content))
    
    return sections


async def insert_diary_entry(
    sem: asyncio.Semaphore,
    ollama: OllamaClient,
    pool: asyncpg.Pool,
    entry: DiaryEntry,
) -> None:
    """Insert a diary entry into the database with embedding."""
    async with sem:
        url = entry.get_url()
        
        # Check if already exists
        exists = await pool.fetchval('SELECT 1 FROM diary_entries WHERE url = $1', url)
        if exists:
            logfire.info('Skipping existing entry', url=url)
            return
        
        with logfire.span('create embedding for diary entry', url=url):
            embedding_vector = await ollama.create_embedding(entry.embedding_content())
            embedding_json = pydantic_core.to_json(embedding_vector).decode()
            
            await pool.execute(
                'INSERT INTO diary_entries (url, date_str, title, content, embedding) VALUES ($1, $2, $3, $4, $5)',
                url,
                entry.date_str,
                entry.title,
                entry.content,
                embedding_json,
            )
            
            print(f"✅ Processed: {entry.title} ({entry.date_str})")


@asynccontextmanager
async def database_connect(
    create_db: bool = False,
) -> AsyncGenerator[asyncpg.Pool, None]:
    """Connect to the PostgreSQL database."""
    server_dsn = 'postgresql://postgres:postgres@localhost:54320'
    database = 'diary_rag'
    
    if create_db:
        with logfire.span('check and create DB'):
            try:
                conn = await asyncpg.connect(server_dsn)
                try:
                    db_exists = await conn.fetchval(
                        'SELECT 1 FROM pg_database WHERE datname = $1', database
                    )
                    if not db_exists:
                        await conn.execute(f'CREATE DATABASE {database}')
                        print(f"📦 Created database '{database}'")
                finally:
                    await conn.close()
            except Exception as e:
                print(f"❌ Database connection error: {e}")
                print("Make sure PostgreSQL is running:")
                print("docker run --rm -e POSTGRES_PASSWORD=postgres -p 54320:5432 -v `pwd`/postgres-data:/var/lib/postgresql/data pgvector/pgvector:pg17")
                raise
    
    pool = await asyncpg.create_pool(f'{server_dsn}/{database}')
    try:
        yield pool
    finally:
        await pool.close()


DB_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS diary_entries (
    id serial PRIMARY KEY,
    url text NOT NULL UNIQUE,
    date_str text NOT NULL,
    title text NOT NULL,
    content text NOT NULL,
    -- nomic-embed-text returns a vector of 768 floats (different from OpenAI's 1536)
    embedding vector(768) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_diary_entries_embedding 
ON diary_entries USING hnsw (embedding vector_l2_ops);

CREATE INDEX IF NOT EXISTS idx_diary_entries_date 
ON diary_entries (date_str);
"""


async def show_stats():
    """Show database statistics."""
    async with database_connect(False) as pool:
        try:
            # Count entries
            entry_count = await pool.fetchval('SELECT COUNT(*) FROM diary_entries')
            
            # Get date range
            date_range = await pool.fetchrow('''
                SELECT MIN(date_str) as earliest, MAX(date_str) as latest 
                FROM diary_entries 
                WHERE date_str != 'Unknown date'
            ''')
            
            # Get recent entries
            recent_entries = await pool.fetch('''
                SELECT title, date_str FROM diary_entries 
                ORDER BY date_str DESC 
                LIMIT 5
            ''')
            
            print(f"\n📊 DIARY DATABASE STATS")
            print(f"{'='*30}")
            print(f"📚 Total entries: {entry_count}")
            
            if date_range['earliest']:
                print(f"📅 Date range: {date_range['earliest']} to {date_range['latest']}")
            
            if recent_entries:
                print(f"\n📝 Recent entries:")
                for entry in recent_entries:
                    print(f"  • {entry['title']} ({entry['date_str']})")
                    
        except Exception as e:
            print(f"❌ Could not get stats: {e}")
            print("Make sure you've built the database first: python ollama_diary_rag.py build")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('🚀 Ollama Diary RAG - Full Featured')
        print('='*40)
        print('Usage:')
        print('  python ollama_diary_rag.py build [diary_folder]    # Build database')
        print('  python ollama_diary_rag.py search "question"       # Single search')
        print('  python ollama_diary_rag.py chat                    # Interactive chat')
        print('  python ollama_diary_rag.py stats                   # Show database stats')
        print()
        print('Requirements:')
        print('  • Ollama running: ollama serve')
        print('  • Models: ollama pull qwen3:8b && ollama pull nomic-embed-text')
        print('  • PostgreSQL: docker run... (see setup guide)')
        sys.exit(1)
    
    action = sys.argv[1]
    
    if action == 'build':
        diary_folder = sys.argv[2] if len(sys.argv) > 2 else 'diary'
        print(f"🔨 Building database from '{diary_folder}' using Ollama...")
        asyncio.run(build_search_db(diary_folder))
        
    elif action == 'search':
        if len(sys.argv) < 3:
            print('Usage: python ollama_diary_rag.py search "your question"')
            sys.exit(1)
        question = sys.argv[2]
        print(f"🔍 Searching for: {question}")
        asyncio.run(run_agent(question))
        
    elif action == 'chat':
        asyncio.run(chat_with_diary())
        
    elif action == 'stats':
        asyncio.run(show_stats())
        
    else:
        print('❌ Invalid action. Use: build|search|chat|stats')
        sys.exit(1)