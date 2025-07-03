#!/usr/bin/env python3
"""
Command-line interface for the RAG system.

This script provides a simple CLI for interacting with the RAG system,
allowing users to ingest documents, perform queries, and manage the system.
"""

import argparse
from typing import List

from rag.core import create_rag_retriever


def load_text_file(file_path: str) -> str:
    """Load text from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return ""


def load_text_files(file_paths: List[str]) -> List[str]:
    """Load text from multiple files."""
    documents = []
    for file_path in file_paths:
        text = load_text_file(file_path)
        if text:
            documents.append(text)
    return documents


def ingest_command(args):
    """Handle the ingest command."""
    print(f"Initializing RAG system with collection: {args.collection}")
    
    rag = create_rag_retriever(
        collection_name=args.collection,
        embedding_model=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k
    )
    
    # Clear existing data if requested
    if args.clear:
        print("Clearing existing data...")
        rag.reset_system()
    
    if args.text:
        # Ingest direct text
        print(f"Ingesting text with {args.strategy} chunking...")
        result = rag.ingest_text(
            text=args.text,
            chunking_strategy=args.strategy
        )
        print(f"Ingestion result: {result}")
    
    elif args.files:
        # Ingest files
        print(f"Loading {len(args.files)} files...")
        documents = load_text_files(args.files)
        
        if not documents:
            print("No valid documents found.")
            return
        
        print(f"Ingesting {len(documents)} documents with {args.strategy} chunking...")
        result = rag.ingest_documents(
            documents=documents,
            chunking_strategy=args.strategy
        )
        print(f"Ingestion result: {result}")
    
    else:
        print("Error: Either --text or --files must be provided for ingestion.")


def query_command(args):
    """Handle the query command."""
    print(f"Querying collection: {args.collection}")
    
    rag = create_rag_retriever(
        collection_name=args.collection,
        embedding_model=args.model,
        top_k=args.top_k
    )
    
    # Check if collection has data
    stats = rag.get_system_stats()
    if stats['vector_store']['document_count'] == 0:
        print("Error: No documents found in collection. Please ingest some documents first.")
        return
    
    print(f"Query: {args.query}")
    print(f"Retrieving top {args.k} results...")
    
    result = rag.retrieve(args.query, k=args.k)
    
    print(f"\nFound {result.total_chunks_found} relevant chunks (retrieval time: {result.retrieval_time:.3f}s):")
    print("=" * 80)
    
    for i, (chunk, similarity) in enumerate(zip(result.chunks, result.similarities)):
        print(f"\nChunk {i+1} (similarity: {similarity:.3f}):")
        print("-" * 40)
        print(chunk.strip())
    
    if args.context:
        print("\n" + "=" * 80)
        print("CONTEXT (for LLM usage):")
        print("=" * 80)
        context = rag.get_context(args.query, k=args.k)
        print(context)


def stats_command(args):
    """Handle the stats command."""
    print(f"Getting statistics for collection: {args.collection}")
    
    rag = create_rag_retriever(collection_name=args.collection)
    stats = rag.get_system_stats()
    
    print("\nRAG System Statistics:")
    print("=" * 50)
    
    # Vector store stats
    vs_stats = stats.get('vector_store', {})
    print(f"Collection Name: {vs_stats.get('collection_name', 'N/A')}")
    print(f"Document Count: {vs_stats.get('document_count', 0)}")
    print(f"Persistence Path: {vs_stats.get('persistence_path', 'N/A')}")
    
    # Embedder stats
    emb_stats = stats.get('embedder', {})
    print(f"Embedding Model: {emb_stats.get('model_name', 'N/A')}")
    print(f"Embedding Dimension: {emb_stats.get('embedding_dimension', 'N/A')}")
    print(f"Model Loaded: {emb_stats.get('loaded', False)}")
    
    # Configuration
    config_stats = stats.get('configuration', {})
    print(f"Chunk Size: {config_stats.get('chunk_size', 'N/A')}")
    print(f"Chunk Overlap: {config_stats.get('chunk_overlap', 'N/A')}")
    print(f"Default Top K: {config_stats.get('default_top_k', 'N/A')}")
    print(f"Similarity Threshold: {config_stats.get('similarity_threshold', 'N/A')}")
    
    # Available strategies
    strategies = stats.get('available_chunking_strategies', [])
    print(f"Available Chunking Strategies: {', '.join(strategies)}")


def clear_command(args):
    """Handle the clear command."""
    print(f"Clearing collection: {args.collection}")
    
    rag = create_rag_retriever(collection_name=args.collection)
    
    if args.reset:
        success = rag.reset_system()
        action = "reset"
    else:
        success = rag.clear_data()
        action = "cleared"
    
    if success:
        print(f"Successfully {action} the collection.")
    else:
        print(f"Failed to {action} the collection.")


def interactive_mode(args):
    """Handle interactive mode."""
    print(f"Starting interactive mode with collection: {args.collection}")
    print("Type 'help' for available commands, 'quit' to exit.")
    
    rag = create_rag_retriever(
        collection_name=args.collection,
        embedding_model=args.model,
        top_k=args.top_k
    )
    
    while True:
        try:
            user_input = input("\nRAG> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            elif user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  query <text>     - Search for relevant chunks")
                print("  stats           - Show system statistics")
                print("  clear           - Clear all data")
                print("  help            - Show this help")
                print("  quit/exit/q     - Exit interactive mode")
            
            elif user_input.lower().startswith('query '):
                query_text = user_input[6:].strip()
                if query_text:
                    result = rag.retrieve(query_text, k=args.k)
                    print(f"\nFound {result.total_chunks_found} relevant chunks:")
                    for i, (chunk, similarity) in enumerate(zip(result.chunks, result.similarities)):
                        print(f"\n{i+1}. (similarity: {similarity:.3f})")
                        print(chunk.strip()[:300] + "..." if len(chunk) > 300 else chunk.strip())
                else:
                    print("Please provide a query text.")
            
            elif user_input.lower() == 'stats':
                stats = rag.get_system_stats()
                vs_stats = stats.get('vector_store', {})
                print(f"\nDocuments: {vs_stats.get('document_count', 0)}")
                print(f"Model: {stats.get('embedder', {}).get('model_name', 'N/A')}")
            
            elif user_input.lower() == 'clear':
                confirm = input("Are you sure you want to clear all data? (y/N): ")
                if confirm.lower() == 'y':
                    success = rag.clear_data()
                    print("Data cleared." if success else "Failed to clear data.")
            
            else:
                print(f"Unknown command: {user_input}")
                print("Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG System CLI - Retrieval-Augmented Generation with ChromaDB and LangChain"
    )
    
    # Global arguments
    parser.add_argument(
        '--collection', 
        default='default_rag',
        help='ChromaDB collection name (default: default_rag)'
    )
    parser.add_argument(
        '--model',
        default='all-MiniLM-L6-v2',
        help='Embedding model name (default: all-MiniLM-L6-v2)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Default number of chunks to retrieve (default: 5)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documents into the RAG system')
    ingest_parser.add_argument('--text', help='Text to ingest directly')
    ingest_parser.add_argument('--files', nargs='+', help='Text files to ingest')
    ingest_parser.add_argument(
        '--strategy',
        choices=['recursive', 'character'],
        default='recursive',
        help='Chunking strategy (default: recursive)'
    )
    ingest_parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Chunk size (default: 1000)'
    )
    ingest_parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=200,
        help='Chunk overlap (default: 200)'
    )
    ingest_parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear existing data before ingesting'
    )
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the RAG system')
    query_parser.add_argument('query', help='Query text')
    query_parser.add_argument(
        '-k',
        type=int,
        help='Number of chunks to retrieve (overrides --top-k)'
    )
    query_parser.add_argument(
        '--context',
        action='store_true',
        help='Also output concatenated context for LLM usage'
    )
    
    # Stats command
    subparsers.add_parser('stats', help='Show system statistics')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear or reset the collection')
    clear_parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset collection (delete and recreate) instead of just clearing'
    )
    
    # Interactive command
    subparsers.add_parser('interactive', help='Start interactive mode')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Set default k value if not provided in query command
    if hasattr(args, 'k') and args.k is None:
        args.k = args.top_k
    
    # Route to appropriate command handler
    if args.command == 'ingest':
        ingest_command(args)
    elif args.command == 'query':
        query_command(args)
    elif args.command == 'stats':
        stats_command(args)
    elif args.command == 'clear':
        clear_command(args)
    elif args.command == 'interactive':
        interactive_mode(args)


if __name__ == '__main__':
    main() 