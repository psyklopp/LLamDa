# 🤖 LLamDa - LLM agent for my Daily diary

Ever wanted to have meaningful conversations with yourself? You have a wide variety of documents, chats, notes, books etc. They are you. Why not talk to them?

This tutorial shows you how to build a Retrieval-Augmented Generation (RAG) system that lets you chat with your diary entries using completely local models. No cloud dependencies, no API costs.

Just you and your thoughts ☁️ 

## What You'll Build

By the end of this tutorial, you'll have:
- A local RAG system that can search through your diary entries
- An interactive chat interface to ask questions about your thoughts and patterns
- A PostgreSQL database with vector embeddings for semantic search
- Everything running locally using Ollama

## Prerequisites

- Python 3.8+
- Docker (for PostgreSQL)

## Step 1: Setting Up Ollama

First, install Ollama and download the required models:

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the required models
ollama pull qwen3:8b           # For chat completion
ollama pull nomic-embed-text   # For embeddings

# Start the Ollama server
ollama serve
```

Verify everything is working by visiting `http://127.0.0.1:11434/` in your browser. You should see "Ollama is running."

**Important Note**: I initially tried DeepSeek-R1, but it doesn't support tools, which are essential for RAG functionality. Qwen3:8b works perfectly for this use case.

You can try code snippets here: [Pydantic AI - OpenAI tools](https://ai.pydantic.dev/models/openai/#built-in-tools) for a better understanding 🙂

## Step 2: Setting Up the Python Environment

Create a virtual environment and install dependencies:

```bash
# Create and activate virtual environment
python -m venv diary-rag-venv
source diary-rag-venv/bin/activate  # On Windows: .\diary-rag-venv\Scripts\activate

# Create requirements.txt
cat > requirements.txt << EOF
pydantic-ai
asyncpg
logfire
pydantic
pydantic-core
httpx
EOF

# Install dependencies
pip install -r requirements.txt
```

## Step 3: Preparing Your Diary Entries

Organize your diary entries in a simple structure:

```
diary/
├── 2025-05-08.md
├── 2025-05-09.md
├── 2025-05-10.md
└── ...
```

Each markdown file should contain your diary entry for that day. The system will automatically extract dates from filenames and split content by headers if needed.

Example diary entry (`2025-05-08.md`):
```markdown
# Morning Reflections

Today I woke up feeling energized. I've been thinking about learning more about AI and how it could help with personal productivity.

# Evening Thoughts

Had a great conversation with a friend about machine learning. It's fascinating how we can build systems that understand our own thoughts and patterns.
```

## Step 4: Setting Up PostgreSQL with Vector Support

We'll use PostgreSQL with the pgvector extension for storing embeddings:

```bash
# Create data directory
mkdir postgres-data

# Run PostgreSQL with pgvector
docker run --rm -e POSTGRES_PASSWORD=postgres -p 54320:5432 \
  -v $(pwd)/postgres-data:/var/lib/postgresql/data \
  pgvector/pgvector:pg17
```

## Step 5: Building the Database

Now comes the exciting part! Run the build command to process your diary entries and create embeddings:

```bash
python ollama_diary_rag_v2.py build diary
```

This process will:
1. Find all markdown files in your diary folder
2. Extract entries and split them into sections
3. Generate embeddings for each entry using Ollama
4. Store everything in PostgreSQL with vector indices

## Step 6: Interacting with Your Diary

### Single Search Queries

Ask specific questions about your diary:

```bash
python ollama_diary_rag_v2.py search "What have I been thinking about AI?"
```

### Interactive Chat Mode

For a more conversational experience:

```bash
python ollama_diary_rag_v2.py chat
```

This opens an interactive session where you can ask follow-up questions and have natural conversations about your diary entries.

### Database Statistics

Check your database stats:

```bash
python ollama_diary_rag_v2.py stats
```

## How It Works

The system uses a two-step process:

1. **Retrieval**: When you ask a question, it creates an embedding of your query and finds the most similar diary entries using vector search
2. **Generation**: It then sends the relevant diary entries along with your question to the local LLM, which generates a thoughtful, personalized response

**LLM generated**: The magic happens in the semantic search—instead of just matching keywords, it understands the meaning and context of your questions and diary entries.

## Example Interactions

Here are some questions one might ask their diary:

- "What patterns do you see in my mood over time?"
- "What have I been most excited about lately?"
- "How have my goals evolved?"
- "What challenges keep coming up in my entries?"

## Areas for Improvement

This is lot of room for enhancement. If the diary entries are made with a standard structure, the overall experience could be made better with good prompts. Along with that, the following could be worked on:

1. **Performance**: Response times could be faster with optimized embeddings or different models
2. **Accuracy**: Fine-tuning the system for specific writing style could reduce hallucinations
3. **Features**: Adding date-range filtering, mood analysis, or goal tracking capabilities

## Troubleshooting

**Ollama Connection Issues**: Ensure Ollama is running with `ollama serve` and both models are pulled.

**Database Connection**: Make sure PostgreSQL container is running and accessible on port 54320.

**No Diary Entries Found**: Check that your markdown files are in the correct directory and contain text content.

## Conclusion

You now have a personal AI assistant that can help you understand patterns in your thoughts and provide insights from your diary entries, all of this running locally on your machine. This project demonstrates the power of RAG systems and how modern AI can help us better understand ourselves.

The beauty of this approach is that your data never leaves your machine, making it perfect for personal and private content like diary entries. As you continue journaling, the system becomes more valuable, building a richer understanding of your thoughts and growth over time.

## What's Next?

I plan to extend and improve this project with following:
- Mood tracking and visualization
- Goal setting and progress monitoring - Turn life into a game!
- Integration with other personal data sources
- Mobile app interface for on-the-go journaling? (Probably not)

Happy journaling and building!
