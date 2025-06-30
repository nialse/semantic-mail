# Semantically search and ask your Gmail

Lightweight CLI agent to semantically **search** and **ask** your emails. Downloads inbox, generates embeddings using local (or external) LLMs, and stores everything in a vector database on your machine. Supports incremental sync for fast updates.

![Demo](assets/demo.gif)


By default, Semantic Mail uses [Ollama](https://github.com/ollama/ollama) for local embeddings and ChromaDB for vector storage

## Install

Semantic Mail requires Python 3.11+, Ollama, and Gmail API credentials.

```bash
# Install Ollama first
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve

# Clone and install
git clone https://github.com/yahorbarkouski/semantic-mail.git
cd semantic-mail

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
uv pip install -e .
```

### Gmail Setup

1. Enable Gmail API at [console.cloud.google.com](https://console.cloud.google.com/)
2. Create OAuth 2.0 credentials (Desktop application)
3. Run setup:

```bash
smail setup
# Enter your Client ID and Secret when prompted
```

## Usage

### Initial Setup / Sync Emails

The `sync` command downloads emails and creates searchable embeddings:

```bash
# Basic sync (uses default embedding model: nomic-embed-text)
smail sync

# Sync with specific embedding model
smail sync --model nomic-embed-text    # 768d, fast (default)
smail sync --model mxbai-embed-large   # 1024d, more accurate
smail sync --model all-minilm          # 384d, lightweight

# Sync with filters
smail sync --limit 1000                # only first 1000 emails
smail sync --query "after:2024/1/1"    # Gmail search syntax
smail sync --query "from:boss@company.com is:important"

# Incremental sync (only new emails since last sync)
smail sync --incremental
smail sync -i --query "is:unread"      # combine with filters

# Use OpenAI embeddings instead of Ollama
smail sync --provider openai --model text-embedding-3-small
smail sync -p openai -m text-embedding-3-large

# Clear and resync
smail sync --clear                      # prompts before clearing
```

**Options:**
- `--provider / -p` - Embedding provider (ollama/openai, default: ollama)
- `--model / -m` - Embedding model (default: nomic-embed-text)
- `--query / -q` - Gmail search query
- `--limit / -l` - Maximum emails to sync
- `--incremental / -i` - Only sync new emails
- `--clear` - Clear existing data before sync

**Note:** Each embedding model creates a separate collection. You must search using the same model you used for indexing.

### Import from MBOX

Use the `sync-mbox` command to index a local `.mbox` file instead of Gmail:

```bash
smail sync-mbox path/to/mailbox.mbox
smail sync-mbox archive.mbox --model mxbai-embed-large
```

**Options:**
- `--provider / -p` - Embedding provider (ollama/openai)
- `--model / -m` - Embedding model
- `--clear` - Clear existing data before import

### Search Emails

The `search` command finds emails by semantic meaning:

```bash
# Basic search (uses default collection)
smail search "Emails I sent to Apple support team in April"

# Search with specific embedding model (must match indexed model)
smail search "Plump.ai project deadline" --model nomic-embed-text
smail search "Technical discussion around openmail architecture" -m mxbai-embed-large

# Search with more results, detailed
smail search "Brex transactions" --limit 20 --detailed

# Use specific provider/model combination
smail search "User/pass from slack sent from sre guy" \
  --provider ollama --model nomic-embed-text
  
# Short syntax
smail search "Blockchain password from 2015" -p ollama -m nomic-embed-text
```

**Options:**
- `--provider / -p` - Embedding provider (must match indexed data)
- `--model / -m` - Embedding model (must match indexed data)
- `--limit / -l` - Number of results (default: 10)
- `--detailed / -d` - Show full email preview

### Ask Questions About Your Emails

The `ask` command uses AI to answer questions based on your email content:

```bash
# Find specific information across multiple emails
smail ask "What was the total amount I spent on Amazon last month?"
smail ask "Summarize all emails about the production outage last week"
smail ask "What's the WiFi password the Airbnb host sent me?"

# Use specific LLM model for better answers
smail ask "Create a timeline of the hiring process for the senior engineer position" --model deepseek-r1:8b

# Use specific LLM model for answers
smail ask "Summarize the project status updates" --model deepseek-r1:8b

# Use custom models for both search and answers
smail ask "Find the AWS credentials that DevOps sent last month" \
  --embedding-provider ollama --embedding-model nomic-embed-text \
  --provider ollama --model deepseek-r1:8b

# Short syntax
smail ask "What's my United frequent flyer number?" \
  -ep ollama -em nomic-embed-text \
  -p ollama -m deepseek-r1:8b

# Search more emails and get longer answers
smail ask "Based on the email threads, what are the main blockers for the Q4 launch?" \
  --search-limit 10 --max-tokens 1000
```

**Options:**
- `--provider / -p` - LLM provider for answering (ollama/openai)
- `--model / -m` - LLM model for answering (e.g., deepseek-r1:8b, gpt-4o-mini)
- `--embedding-provider / -ep` - Provider for email search
- `--embedding-model / -em` - Model for email search (must match your indexed model)
- `--search-limit / -sl` - Number of emails to search (default: 5)
- `--max-tokens / -t` - Maximum response length (default: 500)

### View Statistics

```bash
smail stats    # shows all collections, email counts, and last sync dates
```

### List Available Models

```bash
smail models   # shows available embedding and LLM models
```

## Models

### Embedding Models

Local models (via Ollama):
- `nomic-embed-text` - 768d, fast (default)
- `mxbai-embed-large` - 1024d, more accurate
- `all-minilm` - 384d, lightweight

Cloud models (OpenAI is the only supported provider for now):
- `text-embedding-3-small` - 1536d, good balance
- `text-embedding-3-large` - 3072d, best quality

## Configuration

Semantic Mail uses environment variables stored in `.env`:

```env
GMAIL_CLIENT_ID=your_client_id
GMAIL_CLIENT_SECRET=your_client_secret

OLLAMA_HOST=http://localhost:11434
CHROMA_PERSIST_DIRECTORY=data/chroma
```

## Architecture

```
src/
├── answering/  # Ask AI agent
├── auth/       # Gmail OAuth flow
├── sync/       # Email fetching and processing (Gmail/MBOX)
├── embedding/  # Ollama/OpenAI embedding generation
├── search/     # ChromaDB vector operations
└── cli.py      # Command-line interface
```

The tool:
1. Authenticates with Gmail using OAuth 2.0
2. Fetches email metadata (not attachments)
3. Generates embeddings using the selected model
4. Stores vectors in ChromaDB with model-specific collections
5. Performs cosine similarity search on queries

## System Requirements

- Python 3.11 or higher
- 4GB RAM minimum (8GB recommended)
- ~1GB disk space per 10,000 emails
- Ollama running locally (for local embeddings)

## Troubleshooting

```bash
# Ollama connection issues
curl http://localhost:11434/api/tags    # should return JSON
ollama list                             # verify models installed

# Gmail auth issues
rm credentials/token.json               # force re-authentication
smail setup                             # reconfigure

# Memory issues  
smail sync --limit 500                  # process in smaller batches
```

## License

MIT
