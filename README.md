# RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system built with ChromaDB and LangChain, featuring multiple chunking strategies, fast embeddings, and modular design.

## Features

- **Multiple Chunking Strategies**: Recursive and Character-based chunking
- **Fast Embeddings**: Using SentenceTransformers with free, reliable models
- **Persistent Storage**: ChromaDB for efficient vector storage and retrieval
- **Dynamic Top-K Selection**: Configurable number of results
- **Modular Design**: Clean, extensible architecture following PEP8
- **CLI Interface**: Easy command-line interaction
- **Batch Processing**: Efficient handling of large document sets
- **Metadata Support**: Rich metadata filtering and search

## Installation

The RAG system dependencies are already installed in your project. If you need to install them separately:

```bash
uv add chromadb langchain-community langchain-text-splitters sentence-transformers
```

## Quick Start

### Python API

```python
from rag import create_rag_retriever, ChunkingStrategy

# Create a RAG retriever
rag = create_rag_retriever(
    collection_name="my_docs",
    embedding_model="all-MiniLM-L6-v2",
    chunk_size=1000,
    chunk_overlap=200,
    top_k=5
)

# Ingest some text
result = rag.ingest_text(
    text="Your document text here...",
    chunking_strategy=ChunkingStrategy.RECURSIVE
)

# Query the system
query_result = rag.retrieve("What is the main topic?", k=3)

# Get context for LLM
context = rag.get_context("Your query here", k=5)
```

### Command Line Interface

```bash
# Ingest documents
python -m rag.cli ingest --files document1.txt document2.txt --strategy recursive

# Query the system
python -m rag.cli query "What is machine learning?"

# Get system statistics
python -m rag.cli stats

# Interactive mode
python -m rag.cli interactive
```

## Chunking Strategies

### 1. Recursive Chunking (Default)
Best for most use cases. Splits text hierarchically using multiple separators.

```python
rag.ingest_text(text, chunking_strategy=ChunkingStrategy.RECURSIVE)
```

### 2. Character Chunking
Simple splitting based on specific separators (e.g., paragraphs).

```python
rag.ingest_text(text, chunking_strategy=ChunkingStrategy.CHARACTER)
```

Note: Semantic chunking is planned for future versions when available in LangChain.

## Configuration

### Default Configuration

```python
from rag import RAGConfig

config = RAGConfig(
    CHUNK_SIZE=1000,
    CHUNK_OVERLAP=200,
    EMBEDDING_MODEL="all-MiniLM-L6-v2",
    CHROMA_PERSIST_DIRECTORY="../data/chroma_db",
    COLLECTION_NAME="rag_documents",
    DEFAULT_TOP_K=5,
    SIMILARITY_THRESHOLD=0.7
)
```

### Available Embedding Models

- `all-MiniLM-L6-v2`: Fast and efficient (384 dimensions)
- `all-mpnet-base-v2`: Higher quality (768 dimensions)
- `distilbert-base-nli-mean-tokens`: Good balance (768 dimensions)

## API Reference

### RAGRetriever

Main class for RAG operations.

#### Methods

**`ingest_text(text, chunking_strategy, metadata)`**
- Ingest a single document
- Returns: Dictionary with ingestion results

**`ingest_documents(documents, chunking_strategy, metadatas)`**
- Ingest multiple documents
- Returns: Dictionary with batch ingestion results

**`retrieve(query, k, filter_dict, return_metadata)`**
- Retrieve relevant chunks
- Returns: RAGResult object with chunks, similarities, and metadata

**`get_context(query, k, separator, filter_dict)`**
- Get concatenated context string
- Returns: String ready for LLM consumption

**`get_system_stats()`**
- Get system statistics
- Returns: Dictionary with comprehensive stats

### RAGResult

Result object from retrieval operations.

```python
@dataclass
class RAGResult:
    query: str
    chunks: List[str]
    similarities: List[float]
    metadatas: List[Dict[str, Any]]
    total_chunks_found: int
    retrieval_time: float
```

## Usage Examples

### Basic Document Ingestion and Retrieval

```python
from rag import RAGRetriever, ChunkingStrategy

# Initialize
rag = RAGRetriever()

# Ingest documents
documents = [
    "Artificial intelligence is transforming industries...",
    "Machine learning algorithms learn from data...",
    "Natural language processing enables computers..."
]

result = rag.ingest_documents(
    documents=documents,
    chunking_strategy=ChunkingStrategy.RECURSIVE,
    metadatas=[
        {"topic": "AI", "source": "article1"},
        {"topic": "ML", "source": "article2"},
        {"topic": "NLP", "source": "article3"}
    ]
)

# Retrieve relevant information
query_result = rag.retrieve("How does machine learning work?", k=3)

for i, (chunk, similarity) in enumerate(zip(query_result.chunks, query_result.similarities)):
    print(f"Result {i+1} (similarity: {similarity:.3f}):")
    print(chunk)
    print()
```

### Advanced Usage with Filtering

```python
# Query with metadata filtering
result = rag.retrieve(
    query="artificial intelligence applications",
    k=5,
    filter_dict={"topic": "AI"}
)

# Get context for LLM with custom separator
context = rag.get_context(
    query="machine learning techniques",
    k=3,
    separator="\n---\n"
)
```

### Batch Processing Large Documents

```python
# For large document sets
large_documents = load_many_documents()  # Your document loading function

result = rag.ingest_documents(
    documents=large_documents,
    chunking_strategy=ChunkingStrategy.SEMANTIC
)

print(f"Processed {result['documents_processed']} documents")
print(f"Created {result['total_chunks_added']} chunks")
```

## Command Line Usage

### Ingestion

```bash
# Ingest text files with recursive chunking
python -m rag.cli ingest --files doc1.txt doc2.txt --strategy recursive --chunk-size 800

# Ingest direct text
python -m rag.cli ingest --text "Your text here" --strategy character

# Clear existing data before ingesting
python -m rag.cli ingest --files new_docs.txt --clear
```

### Querying

```bash
# Basic query
python -m rag.cli query "What is artificial intelligence?"

# Get more results
python -m rag.cli query "machine learning applications" -k 10

# Get context output for LLM usage
python -m rag.cli query "deep learning" --context
```

### System Management

```bash
# Check system status
python -m rag.cli stats

# Clear all data
python -m rag.cli clear

# Reset system (delete and recreate collection)
python -m rag.cli clear --reset

# Use specific collection
python -m rag.cli --collection my_collection query "test query"
```

### Interactive Mode

```bash
python -m rag.cli interactive
```

In interactive mode:
- `query <text>` - Search for chunks
- `stats` - Show statistics
- `clear` - Clear data
- `help` - Show commands
- `quit` - Exit

## Performance Tips

1. **Chunk Size**: Larger chunks (1000-1500) for comprehensive context, smaller chunks (300-500) for precise retrieval
2. **Overlap**: 10-20% of chunk size typically works well
3. **Batch Size**: Adjust based on available memory (default: 100)
4. **Embedding Model**: Use `all-MiniLM-L6-v2` for speed, `all-mpnet-base-v2` for quality
5. **Character Chunking**: Simple and fast, good for well-structured documents

## System Architecture

```
RAGRetriever (Main Interface)
├── TextChunker (Chunking Logic)
│   ├── RecursiveChunker
│   └── CharacterChunker
├── TextEmbedder (Embedding Generation)
│   └── SentenceTransformer Models
└── ChromaVectorStore (Vector Storage)
    └── ChromaDB Persistent Client
```

## Error Handling

The system includes comprehensive error handling:

- **Model Loading**: Graceful fallbacks for embedding models
- **Chunking Failures**: Automatic fallback to recursive chunking
- **Storage Issues**: Clear error messages for database problems
- **Empty Results**: Proper handling of empty queries and documents

## Demo and Examples

Run the demonstration script to see the system in action:

```bash
python -m rag.demo
```

This will demonstrate:
- Basic RAG functionality
- Different chunking strategies (recursive and character)
- Advanced features like filtering
- Performance comparisons

## Integration with Existing Codebase

The RAG system is designed to be modular and can be easily integrated:

```python
# In your existing code
from src.rag import create_rag_retriever

def enhance_with_rag(user_query):
    rag = create_rag_retriever(collection_name="knowledge_base")
    context = rag.get_context(user_query, k=5)
    
    # Use context with your existing LLM or processing logic
    return process_with_context(user_query, context)
```

## Contributing

The system follows PEP8 standards and uses type hints throughout. To extend functionality:

1. Add new chunking strategies by extending `BaseChunker`
2. Add new embedding models in `config.py`
3. Extend metadata filtering in `ChromaVectorStore`
4. Add new CLI commands in `cli.py`

## License

This RAG system is part of your QuantBot project and follows the same license terms. 