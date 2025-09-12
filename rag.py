import os
import time
import json
import hashlib
from typing import Any, List, Tuple, Set, Dict, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession
from vertexai.language_models import TextEmbeddingModel
from google.cloud import storage
import fitz  # PyMuPDF for image detection

# community imports to avoid deprecation warnings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

# Semantic chunking dependencies
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

# ─── 1. Initialize Vertex AI ─────────────────────────────────────────────────
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "mf-crucible")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
vertexai.init(project=PROJECT_ID, location=LOCATION)

# ─── 2. Chunking parameters ─────────────────────────────────────────────────
FINE_CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "150"))
FINE_CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "50"))
COARSE_CHUNK_SIZE    = int(os.getenv("COARSE_CHUNK_SIZE", "1200"))
COARSE_CHUNK_OVERLAP = int(os.getenv("COARSE_CHUNK_OVERLAP", "200"))
SEPARATORS           = ["\n\n", "\n", " ", ""]
CONTEXT_WINDOW_SIZE  = 3  # Number of previous queries to keep chunks for

# Context expansion parameters
EXPAND_CONTEXT_BEFORE = int(os.getenv("EXPAND_CONTEXT_BEFORE", "1"))
EXPAND_CONTEXT_AFTER = int(os.getenv("EXPAND_CONTEXT_AFTER", "2"))

# GCS settings
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "attocube-rag-pdfs")
GCS_PDF_PREFIX = "pdfs/"

# Global image storage to avoid ChromaDB metadata issues
GLOBAL_IMAGE_STORAGE = {}

# ─── Semantic Chunking Implementation ────────────────────────────────────────
class SemanticChunker:
    """Semantic-aware chunking that splits documents at semantic boundaries"""
    
    def __init__(self, max_chunk_size: int, overlap_sentences: int = 2):
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences
        self.similarity_threshold = 0.5
        
        # Initialize sentence transformer model
        try:
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Warning: Could not load sentence transformer model: {e}")
            print("Falling back to simple text splitting")
            self.model = None
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, handling technical content and lists"""
        # Handle bullet points and numbered lists
        text = re.sub(r'([•\-\*])\s+', r'\1 ', text)
        text = re.sub(r'(\d+\.)\s+', r'\1 ', text)
        
        # Basic sentence splitting with technical content handling
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Clean and filter sentences
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter out very short fragments
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    def calculate_semantic_similarity(self, sentences: List[str]) -> List[float]:
        """Calculate cosine similarity between consecutive sentences"""
        if not self.model or len(sentences) < 2:
            return [0.5] * (len(sentences) - 1)  # Default similarity
        
        try:
            # Generate embeddings for all sentences
            embeddings = self.model.encode(sentences)
            
            # Calculate cosine similarity between consecutive sentences
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
                similarities.append(sim)
            
            return similarities
        except Exception as e:
            print(f"Warning: Error calculating semantic similarity: {e}")
            return [0.5] * (len(sentences) - 1)  # Default similarity
    
    def create_semantic_chunks(self, sentences: List[str]) -> List[str]:
        """Create chunks at semantic boundaries"""
        if len(sentences) <= 1:
            return sentences
        
        # Calculate semantic similarities
        similarities = self.calculate_semantic_similarity(sentences)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed size limit
            if current_length + sentence_length > self.max_chunk_size and current_chunk:
                # Create chunk from current sentences
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.overlap_sentences)
                current_chunk = current_chunk[overlap_start:]
                current_length = sum(len(s) for s in current_chunk)
            
            # Check for semantic break (low similarity to next sentence)
            semantic_break = (i < len(similarities) and 
                            similarities[i] < self.similarity_threshold and 
                            current_length > self.max_chunk_size * 0.3)  # Don't break too early
            
            current_chunk.append(sentence)
            current_length += sentence_length
            
            if semantic_break and current_length > 50:  # Minimum chunk size
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.overlap_sentences)
                current_chunk = current_chunk[overlap_start:]
                current_length = sum(len(s) for s in current_chunk)
        
        # Add remaining sentences as final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def fallback_split(self, text: str) -> List[str]:
        """Fallback to size-based splitting for oversized segments"""
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length > self.max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(word)
            current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def chunk_documents(self, pages: List[Document]) -> List[Document]:
        """Main method to chunk documents semantically"""
        chunked_docs = []
        
        # Combine text from all pages in the document
        combined_text = ""
        base_metadata = {}
        
        for page in pages:
            combined_text += page.page_content + "\n\n"
            if not base_metadata:
                base_metadata = page.metadata.copy()
        
        # Split into sentences
        sentences = self.split_into_sentences(combined_text)
        
        # Create semantic chunks
        try:
            chunks = self.create_semantic_chunks(sentences)
        except Exception as e:
            print(f"Warning: Semantic chunking failed, using fallback: {e}")
            chunks = self.fallback_split(combined_text)
        
        # Post-process: split any chunks that are still too large
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.max_chunk_size:
                # Use fallback splitting for oversized chunks
                sub_chunks = self.fallback_split(chunk)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        # Create Document objects with preserved metadata
        for i, chunk_text in enumerate(final_chunks):
            if chunk_text.strip():  # Only create non-empty chunks
                chunk_metadata = base_metadata.copy()
                # Note: chunk_index and total_chunks will be set by the calling function
                chunk_doc = Document(
                    page_content=chunk_text.strip(),
                    metadata=chunk_metadata
                )
                chunked_docs.append(chunk_doc)
        
        return chunked_docs


# ─── Caching System ──────────────────────────────────────────────────────────

@dataclass
class CachedQuery:
    """Data class for cached query entries"""
    query_text: str
    query_embedding: List[float]
    validated_chunks: List[Dict[str, Any]]
    chunk_scores: List[float]
    chunk_ids: List[str]
    timestamp: float
    answer_type: str = "mixed"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "query_text": self.query_text,
            "query_embedding": self.query_embedding,
            "validated_chunks": self.validated_chunks,
            "chunk_scores": self.chunk_scores,
            "chunk_ids": self.chunk_ids,
            "timestamp": self.timestamp,
            "answer_type": self.answer_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CachedQuery':
        """Create from dictionary"""
        return cls(**data)

class QueryCache:
    """Query caching system with embedding-based similarity matching"""
    
    def __init__(self, 
                 cache_dir: str = "cache",
                 similarity_threshold: float = 0.85,
                 max_cache_size: int = 1000,
                 expiration_hours: int = 24):
        from pathlib import Path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.expiration_seconds = expiration_hours * 3600
        
        # In-memory cache for faster access
        self.memory_cache: Dict[str, CachedQuery] = {}
        self.cache_index_file = self.cache_dir / "cache_index.json"
        
        # Load existing cache
        self._load_cache_index()
    
    def _generate_cache_key(self, query_text: str) -> str:
        """Generate a unique cache key for a query"""
        return hashlib.md5(query_text.lower().strip().encode()).hexdigest()
    
    def _load_cache_index(self):
        """Load cache index from disk"""
        if self.cache_index_file.exists():
            try:
                with open(self.cache_index_file, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                
                # Load cache entries into memory
                for cache_key, entry_data in index_data.items():
                    try:
                        cached_query = CachedQuery.from_dict(entry_data)
                        # Check if entry is not expired
                        if time.time() - cached_query.timestamp < self.expiration_seconds:
                            self.memory_cache[cache_key] = cached_query
                        else:
                            # Remove expired entry file
                            cache_file = self.cache_dir / f"{cache_key}.pkl"
                            if cache_file.exists():
                                cache_file.unlink()
                    except Exception as e:
                        print(f"Warning: Could not load cache entry {cache_key}: {e}")
                        continue
                
                print(f"Loaded {len(self.memory_cache)} cache entries")
                
            except Exception as e:
                print(f"Warning: Could not load cache index: {e}")
                self.memory_cache = {}
    
    def _save_cache_index(self):
        """Save cache index to disk"""
        try:
            index_data = {}
            for cache_key, cached_query in self.memory_cache.items():
                index_data[cache_key] = cached_query.to_dict()
            
            with open(self.cache_index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cache index: {e}")
    
    def _cleanup_expired_entries(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = []
        
        for cache_key, cached_query in self.memory_cache.items():
            if current_time - cached_query.timestamp >= self.expiration_seconds:
                expired_keys.append(cache_key)
        
        for cache_key in expired_keys:
            del self.memory_cache[cache_key]
            # Remove cache file
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
        
        if expired_keys:
            print(f"Cleaned up {len(expired_keys)} expired cache entries")
            self._save_cache_index()
    
    def _enforce_cache_size_limit(self):
        """Enforce maximum cache size by removing oldest entries"""
        if len(self.memory_cache) <= self.max_cache_size:
            return
        
        # Sort by timestamp and remove oldest entries
        sorted_entries = sorted(
            self.memory_cache.items(),
            key=lambda x: x[1].timestamp
        )
        
        entries_to_remove = len(self.memory_cache) - self.max_cache_size
        for i in range(entries_to_remove):
            cache_key = sorted_entries[i][0]
            del self.memory_cache[cache_key]
            
            # Remove cache file
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
        
        print(f"Removed {entries_to_remove} cache entries to enforce size limit")
        self._save_cache_index()
    
    def lookup(self, query_text: str, query_embedding: List[float]) -> Optional[CachedQuery]:
        """Look up a query in the cache using embedding similarity"""
        # Clean up expired entries periodically
        if len(self.memory_cache) > 0 and len(self.memory_cache) % 50 == 0:
            self._cleanup_expired_entries()
        
        if not self.memory_cache:
            return None
        
        # Find best matching cached query by embedding similarity
        best_similarity = 0.0
        best_match = None
        
        query_embedding_array = np.array(query_embedding).reshape(1, -1)
        
        for cached_query in self.memory_cache.values():
            try:
                cached_embedding_array = np.array(cached_query.query_embedding).reshape(1, -1)
                similarity = cosine_similarity(query_embedding_array, cached_embedding_array)[0][0]
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match = cached_query
            except Exception as e:
                print(f"Warning: Error calculating similarity: {e}")
                continue
        
        if best_match:
            print(f"Cache hit! Similarity: {best_similarity:.3f} for query: '{query_text[:50]}...'")
            return best_match
        
        return None
    
    def store(self, 
              query_text: str,
              query_embedding: List[float],
              validated_chunks: List[Document],
              chunk_scores: List[float],
              answer_type: str = "mixed") -> str:
        """Store a query and its validated chunks in the cache"""
        # Convert Documents to serializable format
        serializable_chunks = []
        chunk_ids = []
        
        for doc in validated_chunks:
            chunk_data = {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "doc_id": doc.metadata.get("doc_id", ""),
                "chunk_index": doc.metadata.get("chunk_index", 0)
            }
            serializable_chunks.append(chunk_data)
            
            # Generate unique chunk ID
            chunk_id = f"{doc.metadata.get('doc_id', 'unknown')}_{doc.metadata.get('chunk_index', 0)}"
            chunk_ids.append(chunk_id)
        
        # Create cache entry
        cached_query = CachedQuery(
            query_text=query_text,
            query_embedding=query_embedding,
            validated_chunks=serializable_chunks,
            chunk_scores=chunk_scores,
            chunk_ids=chunk_ids,
            timestamp=time.time(),
            answer_type=answer_type
        )
        
        # Generate cache key and store
        cache_key = self._generate_cache_key(query_text)
        self.memory_cache[cache_key] = cached_query
        
        # Enforce cache size limit
        self._enforce_cache_size_limit()
        
        # Save to disk
        self._save_cache_index()
        
        print(f"Cached query with {len(validated_chunks)} chunks: '{query_text[:50]}...'")
        return cache_key
    
    def get_cached_documents(self, cached_query: CachedQuery) -> List[Document]:
        """Convert cached chunk data back to Document objects"""
        documents = []
        
        for chunk_data in cached_query.validated_chunks:
            doc = Document(
                page_content=chunk_data["page_content"],
                metadata=chunk_data["metadata"]
            )
            documents.append(doc)
        
        return documents
    
    def invalidate_by_document(self, doc_id: str):
        """Invalidate cache entries that contain chunks from a specific document"""
        keys_to_remove = []
        
        for cache_key, cached_query in self.memory_cache.items():
            # Check if any chunk in this cache entry is from the specified document
            if any(chunk_id.startswith(doc_id) for chunk_id in cached_query.chunk_ids):
                keys_to_remove.append(cache_key)
        
        for cache_key in keys_to_remove:
            del self.memory_cache[cache_key]
            # Remove cache file
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
        
        if keys_to_remove:
            print(f"Invalidated {len(keys_to_remove)} cache entries for document: {doc_id}")
            self._save_cache_index()
    
    def clear_cache(self):
        """Clear all cache entries"""
        # Remove all cache files
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        
        # Clear memory cache
        self.memory_cache.clear()
        
        # Remove index file
        if self.cache_index_file.exists():
            self.cache_index_file.unlink()
        
        print("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        current_time = time.time()
        
        total_entries = len(self.memory_cache)
        expired_entries = sum(
            1 for cached_query in self.memory_cache.values()
            if current_time - cached_query.timestamp >= self.expiration_seconds
        )
        
        return {
            "total_entries": total_entries,
            "active_entries": total_entries - expired_entries,
            "expired_entries": expired_entries,
            "cache_size_mb": sum(
                len(json.dumps(cached_query.to_dict()).encode())
                for cached_query in self.memory_cache.values()
            ) / (1024 * 1024),
            "similarity_threshold": self.similarity_threshold,
            "expiration_hours": self.expiration_seconds / 3600
        }


# ─── Multi-Stage Retrieval ───────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Result of chunk validation"""
    chunk_index: int
    contains_answer: bool
    relevance_score: float
    answer_type: str
    confidence: float = 0.0
    reasoning: str = ""

class MultiStageRetriever:
    """Multi-stage retrieval system with validation and caching"""
    
    def __init__(self, 
                 fine_db, 
                 coarse_db,
                 cache_dir: str = "cache",
                 broad_retrieval_k: int = 15,
                 final_chunks_k: int = 4,
                 validation_timeout: int = 30,
                 max_context_expansions: int = 2):
        
        self.fine_db = fine_db
        self.coarse_db = coarse_db
        self.broad_retrieval_k = broad_retrieval_k
        self.final_chunks_k = final_chunks_k
        self.validation_timeout = validation_timeout
        self.max_context_expansions = max_context_expansions
        
        # Initialize models
        self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        self.validation_model = GenerativeModel("gemini-2.5-flash")
        self.answer_model = GenerativeModel("gemini-2.5-pro")
        
        # Initialize cache
        self.cache = QueryCache(cache_dir=cache_dir)
        
        # Status callback for UI updates
        self.status_callback = None
        
        # Improved validation prompt template
        
        # Improved validation prompt template
        self.validation_prompt_template = """
Query: "{query}"

Text Chunk:
{chunk_content}

Does this chunk help answer the query? Respond with JSON:
{{
  "contains_answer": true/false,
  "relevance_score": 0-10,
  "answer_type": "direct_specification|procedural_context|related_mention|background_context|irrelevant",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation (max 30 words)"
}}
"""
    
    def set_status_callback(self, callback):
        """Set callback function for status updates"""
        self.status_callback = callback
    
    def _update_status(self, message: str):
        """Update status via callback if available"""
        if self.status_callback:
            self.status_callback(message)
    
    def _expand_context(self, query: str, expansion_round: int = 1) -> List[Document]:
        """Expand context by retrieving more chunks with broader search and expanding their contexts"""
        self._update_status(f"Answer not found, expanding context (attempt {expansion_round})")
        
        # Increase retrieval count for expansion
        expanded_k = self.broad_retrieval_k + (expansion_round * 10)
        
        # Try both databases with expanded search
        expanded_chunks = []
        
        try:
            # Search fine database with expanded parameters
            fine_docs = self.fine_db.similarity_search(query, k=expanded_k // 2)
            expanded_chunks.extend(fine_docs)
            
            # Search coarse database with expanded parameters  
            coarse_docs = self.coarse_db.similarity_search(query, k=expanded_k // 2)
            expanded_chunks.extend(coarse_docs)
            
            # Remove duplicates based on content similarity
            unique_chunks = []
            seen_content = set()
            
            for chunk in expanded_chunks:
                # Use first 100 characters as a simple deduplication key
                content_key = chunk.page_content[:100].strip()
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    unique_chunks.append(chunk)
            
            # Apply context expansion to all unique chunks
            context_expanded_chunks = self._expand_chunk_contexts(unique_chunks[:expanded_k])
            
            print(f"Context expansion {expansion_round}: Retrieved {len(unique_chunks)} unique chunks, expanded to {len(context_expanded_chunks)} total chunks")
            return context_expanded_chunks
            
        except Exception as e:
            print(f"Error in context expansion {expansion_round}: {e}")
            return []
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query using text-embedding-004"""
        try:
            embeddings = self.embedding_model.get_embeddings([query])
            return embeddings[0].values
        except Exception as e:
            print(f"Error getting query embedding: {e}")
            raise
    
    def _expand_chunk_contexts(self, docs: List[Document]) -> List[Document]:
        """Expand context for all chunks using neighboring chunks"""
        expanded_docs = []
        seen_chunks = set()
        
        for doc in docs:
            doc_id = doc.metadata.get("doc_id", "")
            chunk_index = doc.metadata.get("chunk_index", 0)
            
            # Add original chunk
            chunk_key = f"{doc_id}_{chunk_index}"
            if chunk_key not in seen_chunks:
                expanded_docs.append(doc)
                seen_chunks.add(chunk_key)
            
            # Determine which database this chunk came from based on chunk size
            source_db = self.fine_db if len(doc.page_content) < 500 else self.coarse_db
            
            # Add preceding chunks
            for i in range(1, EXPAND_CONTEXT_BEFORE + 1):
                prev_chunk = self._get_chunk_by_index(source_db, doc_id, chunk_index - i)
                if prev_chunk:
                    prev_key = f"{doc_id}_{chunk_index - i}"
                    if prev_key not in seen_chunks:
                        expanded_docs.append(prev_chunk)
                        seen_chunks.add(prev_key)
            
            # Add following chunks
            for i in range(1, EXPAND_CONTEXT_AFTER + 1):
                next_chunk = self._get_chunk_by_index(source_db, doc_id, chunk_index + i)
                if next_chunk:
                    next_key = f"{doc_id}_{chunk_index + i}"
                    if next_key not in seen_chunks:
                        expanded_docs.append(next_chunk)
                        seen_chunks.add(next_key)
        
        return expanded_docs
    
    def _get_chunk_by_index(self, db: Chroma, doc_id: str, chunk_index: int) -> Optional[Document]:
        """Retrieve a specific chunk by document ID and chunk index"""
        try:
            results = db.get(
                where={"$and": [
                    {"doc_id": {"$eq": doc_id}},
                    {"chunk_index": {"$eq": chunk_index}}
                ]},
                limit=1
            )
            
            if results and results['documents']:
                metadata = results['metadatas'][0] if results['metadatas'] else {}
                return Document(
                    page_content=results['documents'][0],
                    metadata=metadata
                )
            return None
        except Exception as e:
            print(f"Error retrieving chunk {doc_id}_{chunk_index}: {e}")
            return None

    def _broad_retrieval(self, query: str, use_procedural: bool = False) -> List[Document]:
        """Perform broad candidate retrieval from vector store"""
        try:
            if use_procedural:
                # Use coarse database for procedural queries
                docs = self.coarse_db.similarity_search(
                    query, 
                    k=self.broad_retrieval_k
                )
            else:
                # Use fine database for factual queries
                docs = self.fine_db.similarity_search(
                    query, 
                    k=self.broad_retrieval_k
                )
            
            print(f"Broad retrieval found {len(docs)} candidate chunks")
            return docs
            
        except Exception as e:
            print(f"Error in broad retrieval: {e}")
            return []
    
    def _validate_chunk_sync(self, query: str, chunk: Document, chunk_index: int) -> ValidationResult:
        """Validate a single chunk synchronously with proper token limits"""
        try:
            # More reasonable limits - chunks can be 1200-1500 chars, so we need adequate space
            max_chunk_length = 1000  # Allow for substantial chunk content
            chunk_content = chunk.page_content[:max_chunk_length]
            if len(chunk.page_content) > max_chunk_length:
                chunk_content += "... [truncated]"
            
            # Keep query reasonable but not overly restrictive
            query_limited = query[:200] if len(query) > 200 else query
            
            # Use the proper validation prompt template
            prompt = self.validation_prompt_template.format(
                query=query_limited,
                chunk_content=chunk_content
            )
            
            # Set appropriate token limits - input can be ~1500 tokens, output needs ~500 for JSON
            response = self.validation_model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.0,
                    "max_output_tokens": 800,  # Adequate for JSON response with reasoning
                    "candidate_count": 1,
                }
            )
            
            # Check if we got a response
            if not response or not response.text:
                print(f"Empty response for chunk {chunk_index}")
                return ValidationResult(
                    chunk_index=chunk_index,
                    contains_answer=False,
                    relevance_score=0.0,
                    answer_type="irrelevant",
                    confidence=0.0,
                    reasoning="Empty response from validation model"
                )
            
            # Parse JSON response, handling ```json wrapper
            response_text = response.text.strip()
            
            # Remove ```json wrapper if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]  # Remove ```json
            if response_text.startswith("```"):
                response_text = response_text[3:]  # Remove ```
            if response_text.endswith("```"):
                response_text = response_text[:-3]  # Remove ```
            
            response_text = response_text.strip()
            
            try:
                result_data = json.loads(response_text)
                
                return ValidationResult(
                    chunk_index=chunk_index,
                    contains_answer=result_data.get("contains_answer", False),
                    relevance_score=float(result_data.get("relevance_score", 0)),
                    answer_type=result_data.get("answer_type", "irrelevant"),
                    confidence=float(result_data.get("confidence", 0)),
                    reasoning=result_data.get("reasoning", "")
                )
                
            except json.JSONDecodeError as e:
                print(f"Failed to parse validation JSON for chunk {chunk_index}: {e}")
                print(f"Raw response: {response_text[:200]}...")
                
                # Fallback: try to extract basic info from partial response
                contains_answer = "true" in response_text.lower() and ("contains_answer" in response_text.lower() or "answer" in response_text.lower())
                return ValidationResult(
                    chunk_index=chunk_index,
                    contains_answer=contains_answer,
                    relevance_score=5.0 if contains_answer else 2.0,
                    answer_type="related_mention" if contains_answer else "irrelevant",
                    confidence=0.5,
                    reasoning="Partial parsing due to JSON error"
                )
                
        except Exception as e:
            print(f"Error validating chunk {chunk_index}: {e}")
            return ValidationResult(
                chunk_index=chunk_index,
                contains_answer=False,
                relevance_score=0.0,
                answer_type="irrelevant",
                confidence=0.0,
                reasoning=f"Validation error: {str(e)[:50]}"
            )
    
    def validate_chunks_parallel(self, query: str, chunks: List[Document]) -> List[ValidationResult]:
        """Validate multiple chunks in parallel using ThreadPoolExecutor"""
        if not chunks:
            return []
        
        print(f"Starting parallel validation of {len(chunks)} chunks...")
        start_time = time.time()
        
        validation_results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=min(8, len(chunks))) as executor:
            # Submit all validation tasks
            future_to_index = {
                executor.submit(self._validate_chunk_sync, query, chunk, i): i
                for i, chunk in enumerate(chunks)
            }
            
            # Collect results with timeout
            try:
                for future in as_completed(future_to_index, timeout=self.validation_timeout):
                    try:
                        result = future.result(timeout=5)  # Individual chunk timeout
                        validation_results.append(result)
                    except Exception as e:
                        chunk_index = future_to_index[future]
                        print(f"Validation failed for chunk {chunk_index}: {e}")
                        # Add failed result
                        validation_results.append(ValidationResult(
                            chunk_index=chunk_index,
                            contains_answer=False,
                            relevance_score=0.0,
                            answer_type="irrelevant",
                            confidence=0.0,
                            reasoning="Validation timeout"
                        ))
                        
            except TimeoutError:
                print(f"Validation timeout after {self.validation_timeout} seconds")
                # Add timeout results for remaining chunks
                completed_indices = {result.chunk_index for result in validation_results}
                for i in range(len(chunks)):
                    if i not in completed_indices:
                        validation_results.append(ValidationResult(
                            chunk_index=i,
                            contains_answer=False,
                            relevance_score=0.0,
                            answer_type="irrelevant",
                            confidence=0.0,
                            reasoning="Validation timeout"
                        ))
        
        # Sort results by chunk index to maintain order
        validation_results.sort(key=lambda x: x.chunk_index)
        
        elapsed_time = time.time() - start_time
        print(f"Parallel validation completed in {elapsed_time:.2f} seconds")
        
        # Log validation summary
        positive_results = [r for r in validation_results if r.contains_answer]
        avg_score = np.mean([r.relevance_score for r in validation_results]) if validation_results else 0
        print(f"Validation summary: {len(positive_results)}/{len(validation_results)} chunks contain answers, avg score: {avg_score:.2f}")
        
        return validation_results
    
    def _rerank_chunks(self, 
                      chunks: List[Document], 
                      validation_results: List[ValidationResult]) -> Tuple[List[Document], List[float]]:
        """Re-rank chunks based on validation results"""
        if not chunks or not validation_results:
            return [], []
        
        # Create list of (chunk, validation_result, composite_score) tuples
        chunk_scores = []
        
        for i, (chunk, validation) in enumerate(zip(chunks, validation_results)):
            # Calculate composite score
            base_score = validation.relevance_score
            
            # Boost score based on answer type
            type_multipliers = {
                "direct_specification": 1.3,
                "procedural_context": 1.2,
                "related_mention": 1.0,
                "background_context": 0.9,
                "irrelevant": 0.3
            }
            type_multiplier = type_multipliers.get(validation.answer_type, 1.0)
            
            # Boost score if it contains answer
            answer_multiplier = 1.4 if validation.contains_answer else 1.0
            
            # Apply confidence weighting
            confidence_weight = max(0.3, validation.confidence)  # Minimum confidence
            
            composite_score = base_score * type_multiplier * answer_multiplier * confidence_weight
            
            chunk_scores.append((chunk, validation, composite_score))
        
        # Sort by composite score (descending)
        chunk_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Extract top chunks and scores
        top_chunks = [item[0] for item in chunk_scores[:self.final_chunks_k]]
        top_scores = [item[2] for item in chunk_scores[:self.final_chunks_k]]
        
        print(f"Re-ranking selected top {len(top_chunks)} chunks with scores: {[f'{s:.2f}' for s in top_scores]}")
        
        return top_chunks, top_scores
    
    def _rerank_chunks_fallback(self, 
                               chunks: List[Document], 
                               validation_results: List[ValidationResult]) -> Tuple[List[Document], List[float]]:
        """Fallback re-ranking when no chunks are marked as containing answers"""
        if not chunks or not validation_results:
            return [], []
        
        print("Using fallback ranking strategy (no chunks marked as containing answers)")
        
        # Create list of (chunk, validation_result, composite_score) tuples
        chunk_scores = []
        
        for i, (chunk, validation) in enumerate(zip(chunks, validation_results)):
            # Use relevance score as primary ranking factor
            base_score = validation.relevance_score
            
            # More lenient type multipliers
            type_multipliers = {
                "direct_specification": 1.3,
                "procedural_context": 1.2,
                "related_mention": 1.1,
                "background_context": 1.05,
                "irrelevant": 0.5  # Don't completely eliminate
            }
            type_multiplier = type_multipliers.get(validation.answer_type, 1.0)
            
            # Don't penalize for contains_answer=False in fallback mode
            confidence_weight = max(0.3, validation.confidence)
            
            composite_score = base_score * type_multiplier * confidence_weight
            
            chunk_scores.append((chunk, validation, composite_score))
        
        # Sort by composite score (descending)
        chunk_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Extract top chunks and scores
        top_chunks = [item[0] for item in chunk_scores[:self.final_chunks_k]]
        top_scores = [item[2] for item in chunk_scores[:self.final_chunks_k]]
        
        print(f"Fallback ranking selected top {len(top_chunks)} chunks with scores: {[f'{s:.2f}' for s in top_scores]}")
        
        return top_chunks, top_scores
    
    def retrieve_and_validate(self, query: str) -> Tuple[List[Document], List[float], bool]:
        """Main multi-stage retrieval method with caching and context expansion"""
        print(f"\n=== Multi-Stage Retrieval for: '{query[:50]}...' ===")
        self._update_status("Searching knowledge base")
        
        # Step 1: Get query embedding
        try:
            query_embedding = self._get_query_embedding(query)
        except Exception as e:
            print(f"Failed to get query embedding: {e}")
            return [], [], False
        
        # Step 2: Check cache
        cached_result = self.cache.lookup(query, query_embedding)
        if cached_result:
            print("Using cached result!")
            self._update_status("Found cached result")
            cached_docs = self.cache.get_cached_documents(cached_result)
            return cached_docs, cached_result.chunk_scores, True
        
        print("No cache hit, proceeding with full retrieval...")
        
        # Step 3: Determine query type for retrieval strategy
        lower_query = query.lower()
        is_procedural = any(keyword in lower_query for keyword in 
                           ["how", "procedure", "steps", "process", "install", "setup", "calibrate"])
        
        # Step 4: Initial broad retrieval - get original 15 chunks
        self._update_status("Finding relevant information")
        original_chunks = self._broad_retrieval(query, use_procedural=is_procedural)
        
        if not original_chunks:
            print("No candidate chunks found in initial retrieval")
            return [], [], False
        
        # Track validated chunks with their results using chunk_key -> (chunk, validation_result)
        validated_chunk_map = {}
        
        # Step 5: Process chunks with potential context expansion
        validated_chunks = []
        scores = []
        expansion_count = 0
        
        while expansion_count <= self.max_context_expansions:
            if expansion_count == 0:
                # Round 0: Validate original 15 chunks
                candidate_chunks = original_chunks
                print(f"Round {expansion_count}: Validating {len(candidate_chunks)} original chunks")
            else:
                # Round 1+: Expand context and validate only new chunks
                self._update_status(f"Answer not found, expanding context (attempt {expansion_count})")
                
                # Get all chunks with context expansion (original + neighbors)
                expanded_chunks = self._expand_chunk_contexts(original_chunks)
                
                # Filter out already validated chunks
                candidate_chunks = []
                for chunk in expanded_chunks:
                    chunk_key = f"{chunk.metadata.get('doc_id', '')}_{chunk.metadata.get('chunk_index', 0)}"
                    if chunk_key not in validated_chunk_map:
                        candidate_chunks.append(chunk)
                
                print(f"Context expansion {expansion_count}: Retrieved {len(expanded_chunks)} total chunks, {len(candidate_chunks)} new chunks to validate")
            
            if not candidate_chunks:
                print(f"No new candidate chunks found in round {expansion_count}")
                expansion_count += 1
                continue
            
            # Step 6: Parallel validation of new chunks only
            self._update_status("Analyzing content relevance")
            validation_results = self.validate_chunks_parallel(query, candidate_chunks)
            
            if not validation_results:
                print(f"Validation failed in round {expansion_count}")
                expansion_count += 1
                continue
            
            # Store validated chunks with their results
            for chunk, validation_result in zip(candidate_chunks, validation_results):
                chunk_key = f"{chunk.metadata.get('doc_id', '')}_{chunk.metadata.get('chunk_index', 0)}"
                validated_chunk_map[chunk_key] = (chunk, validation_result)
            
            # Check if we found any chunks that contain answers in this round
            positive_results = [r for r in validation_results if r.contains_answer]
            
            if len(positive_results) > 0:
                # We found relevant chunks! Now re-rank all validated chunks so far
                self._update_status("Ranking best matches")
                
                # Get all validated chunks and their results in order
                all_validated_chunks = [item[0] for item in validated_chunk_map.values()]
                all_validation_results = [item[1] for item in validated_chunk_map.values()]
                
                print(f"Re-ranking {len(all_validated_chunks)} total validated chunks")
                validated_chunks, scores = self._rerank_chunks(all_validated_chunks, all_validation_results)
                break
            else:
                print(f"No positive validation results in round {expansion_count} ({len(positive_results)}/{len(validation_results)} chunks marked as containing answers)")
                expansion_count += 1
                
                # If this is the last expansion attempt, use fallback ranking on all chunks
                if expansion_count > self.max_context_expansions:
                    print("Maximum context expansions reached, using fallback ranking")
                    self._update_status("Using best available matches")
                    
                    # Get all validated chunks and their results in order
                    all_validated_chunks = [item[0] for item in validated_chunk_map.values()]
                    all_validation_results = [item[1] for item in validated_chunk_map.values()]
                    
                    print(f"Fallback ranking {len(all_validated_chunks)} total validated chunks")
                    validated_chunks, scores = self._rerank_chunks_fallback(all_validated_chunks, all_validation_results)
                    break
        
        if not validated_chunks:
            print("No chunks passed validation after all expansion attempts")
            return [], [], False
        
        # Step 7: Cache the results
        answer_types = [validation_result.answer_type for chunk, validation_result in validated_chunk_map.values() if validation_result.contains_answer]
        dominant_answer_type = max(set(answer_types), key=answer_types.count) if answer_types else "mixed"
        
        self.cache.store(
            query_text=query,
            query_embedding=query_embedding,
            validated_chunks=validated_chunks,
            chunk_scores=scores,
            answer_type=dominant_answer_type
        )
        
        print(f"Multi-stage retrieval completed: {len(validated_chunks)} validated chunks")
        return validated_chunks, scores, False
    
    def generate_answer(self, query: str, validated_chunks: List[Document]) -> str:
        """Generate final answer using validated chunks"""
        if not validated_chunks:
            return "I couldn't find relevant information to answer your question."
        
        self._update_status("Generating answer")
        
        # Prepare context from validated chunks
        context_parts = []
        for i, chunk in enumerate(validated_chunks, 1):
            source = chunk.metadata.get('source', 'Unknown')
            page = chunk.metadata.get('page', 'Unknown')
            doc_type = chunk.metadata.get('doc_type', 'document')
            
            context_parts.append(f"[Source {i}: {source} (page {page}, {doc_type})]")
            context_parts.append(chunk.page_content)
            context_parts.append("")  # Empty line for separation
        
        context = "\n".join(context_parts)
        
        # Generate answer prompt
        answer_prompt = f"""You are a helpful technical assistant. Use the following validated and ranked context to answer the user's question accurately and comprehensively.

The context has been pre-validated to contain relevant information for this query.

Question: {query}

Validated Context:
{context}

Instructions:
1. Provide a clear, accurate answer based on the validated context
2. Reference specific sources when mentioning facts or procedures
3. If the context contains multiple relevant pieces of information, synthesize them coherently
4. If any information is incomplete, mention what additional details might be needed
5. Maintain technical accuracy while being accessible

Answer:"""
        
        try:
            response = self.answer_model.generate_content(
                answer_prompt,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 2048,
                }
            )
            return response.text
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"I found relevant information but encountered an error generating the response: {str(e)}"
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_cache_stats()
    
    def clear_cache(self):
        """Clear the query cache"""
        self.cache.clear_cache()
    
    def invalidate_cache_for_document(self, doc_id: str):
        """Invalidate cache entries for a specific document"""
        self.cache.invalidate_by_document(doc_id)

# ─── 3. LLM wrapper for Vertex AI Gemini ────────────────────────────────────
class VertexAIGeminiLLM(LLM):
    model: Optional[GenerativeModel] = None  # Add default value
    model_name: str = "gemini-2.5-pro"
    
    def __init__(self, model_name: str = None):
        super().__init__()
        if model_name:
            self.model_name = model_name
        self.model = GenerativeModel(self.model_name)
    
    @property
    def _llm_type(self) -> str:
        return "vertex-ai-gemini"
    
    def _call(self, prompt: str, stop=None) -> str:
        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 2048,
            }
        )
        return response.text

# ─── 4. Embeddings wrapper for Vertex AI ─────────────────────────────────────
class VertexAIEmbeddings(Embeddings):
    model: TextEmbeddingModel
    model_name: str = "text-embedding-005"
    
    def __init__(self, model_name: str = None):
        if model_name:
            self.model_name = model_name
        self.model = TextEmbeddingModel.from_pretrained(self.model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = self.model.get_embeddings(batch)
            embeddings.extend([emb.values for emb in batch_embeddings])
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        embeddings = self.model.get_embeddings([text])
        return embeddings[0].values

# ─── 5. GCS PDF Loading ──────────────────────────────────────────────────────
def download_pdfs_from_gcs(bucket_name: str, prefix: str, local_dir: str = "pdfs"):
    """Download PDFs from GCS to local directory"""
    os.makedirs(local_dir, exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # List and download all PDFs
    blobs = bucket.list_blobs(prefix=prefix)
    pdf_count = 0
    
    for blob in blobs:
        if blob.name.endswith('.pdf'):
            local_path = os.path.join(local_dir, os.path.basename(blob.name))
            blob.download_to_filename(local_path)
            pdf_count += 1
            print(f"Downloaded: {blob.name}")
    
    print(f"Downloaded {pdf_count} PDFs from GCS")
    return local_dir

# Keep all the original chunking and retriever logic...
# (I'll include the rest of your original functions with minor modifications)

def load_and_split_pdfs(folder: str) -> Tuple[List[Document], List[Document]]:
    # Initialize semantic chunkers instead of RecursiveCharacterTextSplitter
    semantic_chunker_fine = SemanticChunker(max_chunk_size=FINE_CHUNK_SIZE)
    semantic_chunker_coarse = SemanticChunker(max_chunk_size=COARSE_CHUNK_SIZE)
    
    fine_docs, coarse_docs = [], []

    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".pdf"): continue
        path = os.path.join(folder, fname)

        # Detect and extract image information
        doc_pdf = fitz.open(path)
        pages_with_images = {}
        for i, pg in enumerate(doc_pdf):
            page_num = i + 1
            images = pg.get_images()
            if images:
                pages_with_images[page_num] = []
                for img_index, img in enumerate(images):
                    try:
                        # Extract image
                        xref = img[0]
                        pix = fitz.Pixmap(doc_pdf, xref)
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            pages_with_images[page_num].append({
                                "image_index": img_index,
                                "image_data": img_data,
                                "width": pix.width,
                                "height": pix.height
                            })
                        if pix:
                            pix = None
                    except Exception as e:
                        print(f"Failed to extract image {img_index} from page {page_num} of {fname}: {e}")
                        continue
        doc_pdf.close()
        
        # Print image extraction summary
        total_images = sum(len(imgs) for imgs in pages_with_images.values())
        if total_images > 0:
            print(f"Extracted {total_images} images from {fname} across {len(pages_with_images)} pages")

        # Classify document type based on filename
        doc_type = "email" if fname.startswith("UHD") else "manual"
        
        pages = PyPDFLoader(path).load()
        for i, doc in enumerate(pages, start=1):
            doc.metadata.update({
                "source": fname,
                "page": i,
                "has_image": i in pages_with_images,
                "doc_type": doc_type,
                "image_count": len(pages_with_images.get(i, []))
            })
            # Store images separately to avoid ChromaDB metadata issues
            if i in pages_with_images:
                image_key = f"{fname}_page_{i}"
                GLOBAL_IMAGE_STORAGE[image_key] = pages_with_images[i]
        
        # Use semantic chunking instead of RecursiveCharacterTextSplitter
        fine_chunks = semantic_chunker_fine.chunk_documents(pages)
        coarse_chunks = semantic_chunker_coarse.chunk_documents(pages)
        
        # Add chunk index within the document
        for idx, chunk in enumerate(fine_chunks):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["total_chunks"] = len(fine_chunks)
            chunk.metadata["doc_id"] = fname
        
        for idx, chunk in enumerate(coarse_chunks):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["total_chunks"] = len(coarse_chunks)
            chunk.metadata["doc_id"] = fname
        
        fine_docs.extend(fine_chunks)
        coarse_docs.extend(coarse_chunks)

    return fine_docs, coarse_docs

# ─── 6. ChromaDB setup ───────────────────────────────────────────────────────
CHROMA_BASE_DIR   = "/tmp/chroma_db"  # Use /tmp for Cloud Run
CHROMA_FINE_DIR   = os.path.join(CHROMA_BASE_DIR, "fine")
CHROMA_COARSE_DIR = os.path.join(CHROMA_BASE_DIR, "coarse")

def get_vectorstores(fine_docs: List[Document], coarse_docs: List[Document]):
    os.makedirs(CHROMA_FINE_DIR, exist_ok=True)
    os.makedirs(CHROMA_COARSE_DIR, exist_ok=True)
    emb = VertexAIEmbeddings()
    
    # Always rebuild for Cloud Run
    print("Building vector stores...")
    fine_db = Chroma.from_documents(fine_docs, emb, persist_directory=CHROMA_FINE_DIR)
    coarse_db = Chroma.from_documents(coarse_docs, emb, persist_directory=CHROMA_COARSE_DIR)
    print("Vector stores built successfully!")
    
    return fine_db, coarse_db

# Include all your original retriever and conversation classes here...
# (ContextExpandingHybridRetriever, ConversationHistory, etc.)
# I'll skip them for brevity but they remain the same

class ContextExpandingHybridRetriever:
    """Hybrid retriever that expands context by including neighboring chunks"""
    def __init__(self, fine_db, coarse_db):
        self.fine_db = fine_db
        self.coarse_db = coarse_db
        self.fine_retriever = fine_db.as_retriever(search_kwargs={"k": 3})
        self.coarse_retriever = coarse_db.as_retriever(search_kwargs={"k": 2})
    
    def expand_context(self, docs: List[Document], db: Chroma, before: int = 1, after: int = 2) -> List[Document]:
        """Expand context by including neighboring chunks"""
        expanded_docs = []
        seen_chunks = set()
        
        for doc in docs:
            doc_id = doc.metadata.get("doc_id")
            chunk_idx = doc.metadata.get("chunk_index")
            
            if doc_id is None or chunk_idx is None:
                expanded_docs.append(doc)
                continue
            
            # Add chunks before
            for i in range(before, 0, -1):
                target_idx = chunk_idx - i
                if target_idx >= 0:
                    neighbor = self._get_chunk_by_index(db, doc_id, target_idx)
                    if neighbor and (doc_id, target_idx) not in seen_chunks:
                        expanded_docs.append(neighbor)
                        seen_chunks.add((doc_id, target_idx))
            
            # Add the original chunk
            if (doc_id, chunk_idx) not in seen_chunks:
                expanded_docs.append(doc)
                seen_chunks.add((doc_id, chunk_idx))
            
            # Add chunks after
            total_chunks = doc.metadata.get("total_chunks", float('inf'))
            for i in range(1, after + 1):
                target_idx = chunk_idx + i
                if target_idx < total_chunks:
                    neighbor = self._get_chunk_by_index(db, doc_id, target_idx)
                    if neighbor and (doc_id, target_idx) not in seen_chunks:
                        expanded_docs.append(neighbor)
                        seen_chunks.add((doc_id, target_idx))
        
        return expanded_docs
    
    def _get_chunk_by_index(self, db: Chroma, doc_id: str, chunk_index: int) -> Optional[Document]:
        """Retrieve a specific chunk by document ID and chunk index"""
        results = db.get(
            where={"$and": [
                {"doc_id": {"$eq": doc_id}},
                {"chunk_index": {"$eq": chunk_index}}
            ]},
            limit=1
        )
        
        if results and results['documents']:
            return Document(
                page_content=results['documents'][0],
                metadata=results['metadatas'][0]
            )
        return None
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Main retrieval method with context expansion"""
        lower_q = query.lower()
        
        procedural = any(k in lower_q for k in ["how", "procedure", "steps", "process", "install", "setup"])
        
        if procedural:
            docs = self.coarse_retriever.get_relevant_documents(query)
            expanded = self.expand_context(
                docs, 
                self.coarse_db, 
                before=EXPAND_CONTEXT_BEFORE, 
                after=EXPAND_CONTEXT_AFTER + 1
            )
        else:
            docs = self.fine_retriever.get_relevant_documents(query)
            expanded = self.expand_context(
                docs, 
                self.fine_db, 
                before=EXPAND_CONTEXT_BEFORE, 
                after=EXPAND_CONTEXT_AFTER
            )
        
        return expanded
    
    def get_relevant_documents_by_type(self, query: str, doc_type: str = None) -> List[Document]:
        """Retrieval method with optional document type filtering"""
        lower_q = query.lower()
        
        procedural = any(k in lower_q for k in ["how", "procedure", "steps", "process", "install", "setup"])
        
        if procedural:
            if doc_type:
                # Use filtered retriever for specific document types
                docs = self._get_filtered_documents(query, self.coarse_db, doc_type, k=2)
            else:
                docs = self.coarse_retriever.get_relevant_documents(query)
            expanded = self.expand_context(
                docs, 
                self.coarse_db, 
                before=EXPAND_CONTEXT_BEFORE, 
                after=EXPAND_CONTEXT_AFTER + 1
            )
        else:
            if doc_type:
                # Use filtered retriever for specific document types
                docs = self._get_filtered_documents(query, self.fine_db, doc_type, k=3)
            else:
                docs = self.fine_retriever.get_relevant_documents(query)
            expanded = self.expand_context(
                docs, 
                self.fine_db, 
                before=EXPAND_CONTEXT_BEFORE, 
                after=EXPAND_CONTEXT_AFTER
            )
        
        return expanded
    
    def get_relevant_documents_with_expansion(self, query: str, before: int = None, after: int = None) -> List[Document]:
        """Main retrieval method with custom context expansion parameters"""
        if before is None:
            before = EXPAND_CONTEXT_BEFORE
        if after is None:
            after = EXPAND_CONTEXT_AFTER
            
        lower_q = query.lower()
        
        procedural = any(k in lower_q for k in ["how", "procedure", "steps", "process", "install", "setup"])
        
        if procedural:
            docs = self.coarse_retriever.get_relevant_documents(query)
            expanded = self.expand_context(docs, self.coarse_db, before=before, after=after + 1)
        else:
            docs = self.fine_retriever.get_relevant_documents(query)
            expanded = self.expand_context(docs, self.fine_db, before=before, after=after)
        
        return expanded
    
    def get_relevant_documents_by_type_with_expansion(self, query: str, doc_type: str = None, before: int = None, after: int = None) -> List[Document]:
        """Retrieval method with optional document type filtering and custom context expansion"""
        if before is None:
            before = EXPAND_CONTEXT_BEFORE
        if after is None:
            after = EXPAND_CONTEXT_AFTER
            
        lower_q = query.lower()
        
        procedural = any(k in lower_q for k in ["how", "procedure", "steps", "process", "install", "setup"])
        
        if procedural:
            if doc_type:
                docs = self._get_filtered_documents(query, self.coarse_db, doc_type, k=2)
            else:
                docs = self.coarse_retriever.get_relevant_documents(query)
            expanded = self.expand_context(docs, self.coarse_db, before=before, after=after + 1)
        else:
            if doc_type:
                docs = self._get_filtered_documents(query, self.fine_db, doc_type, k=3)
            else:
                docs = self.fine_retriever.get_relevant_documents(query)
            expanded = self.expand_context(docs, self.fine_db, before=before, after=after)
        
        return expanded
    
    def _get_filtered_documents(self, query: str, db: Chroma, doc_type: str, k: int = 3) -> List[Document]:
        """Get documents filtered by document type"""
        # Perform similarity search with metadata filter
        docs = db.similarity_search(
            query, 
            k=k*3,  # Get more docs initially to account for filtering
            filter={"doc_type": doc_type}
        )
        # Return top k after filtering
        return docs[:k]
    
    def invoke(self, input: str, config=None) -> List[Document]:
        return self.get_relevant_documents(input)
    
    async def ainvoke(self, input: str, config=None) -> List[Document]:
        return self.get_relevant_documents(input)

class ConversationHistory:
    def __init__(self):
        self.messages = []
        self.awaiting_clarification = False
        self.original_question = None
        self.chunk_history = []
        self.max_chunk_history = CONTEXT_WINDOW_SIZE
    
    def add_user_message(self, message: str):
        self.messages.append(HumanMessage(content=message))
    
    def add_ai_message(self, message: str):
        self.messages.append(AIMessage(content=message))
    
    def add_chunks_to_history(self, question: str, chunks: List[Document]):
        self.chunk_history.append((question, chunks))
        if len(self.chunk_history) > self.max_chunk_history:
            self.chunk_history.pop(0)
    
    def get_recent_chunks(self, n: int = 2) -> List[Document]:
        all_chunks = []
        for _, chunks in self.chunk_history[-n:]:
            all_chunks.extend(chunks)
        seen = set()
        unique_chunks = []
        for chunk in all_chunks:
            if chunk.page_content not in seen:
                seen.add(chunk.page_content)
                unique_chunks.append(chunk)
        return unique_chunks
    
    def get_messages(self):
        return self.messages
    
    def set_clarification_mode(self, question: str):
        self.awaiting_clarification = True
        self.original_question = question
    
    def clear_clarification_mode(self):
        self.awaiting_clarification = False
        self.original_question = None
    
    def get_conversation_context(self, n: int = 3) -> str:
        recent = self.messages[-n*2:] if len(self.messages) >= n*2 else self.messages
        context = []
        for msg in recent:
            if isinstance(msg, HumanMessage):
                context.append(f"User: {msg.content}")
            else:
                context.append(f"Assistant: {msg.content[:200]}...")
        return "\n".join(context)
    
    def clear(self):
        self.messages = []
        self.awaiting_clarification = False
        self.original_question = None
        self.chunk_history = []

def is_follow_up_query(query: str) -> bool:
    follow_up_phrases = [
        "tell me more", "more about", "what else", "anything else",
        "more details", "more information", "explain that",
        "elaborate", "go on", "continue", "what about",
        "how about", "and", "also", "furthermore", "additionally"
    ]
    lower_q = query.lower().strip()
    return any(phrase in lower_q for phrase in follow_up_phrases) or len(lower_q.split()) <= 3

def needs_clarification(query: str) -> bool:
    vague_queries = [
        "what", "how", "why", "when", "where", "who",
        "tell me", "explain", "help", "info", "information",
        "details", "specs", "?"
    ]
    
    query_lower = query.lower().strip()
    words = query_lower.split()
    
    if len(words) <= 2 and any(vague == query_lower for vague in vague_queries):
        return True
    
    if any(char.isdigit() for char in query):
        return False
    if len(words) > 5:
        return False
    if any(word in query_lower for word in ["model", "serial", "temperature", "pressure", "voltage", "current", "power", "size", "dimension"]):
        return False
    
    return False

def extract_images_from_chunks(docs: List[Document]) -> List[Dict]:
    """Extract images from document chunks that contain images"""
    images = []
    seen_images = set()
    
    for doc in docs:
        # Check if this chunk has images using metadata
        if doc.metadata.get("has_image", False):
            source = doc.metadata.get('source', '')
            page = doc.metadata.get('page', 0)
            
            # Get image key for this page
            image_key = f"{source}_page_{page}"
            
            # Check if this page has images in global storage
            if image_key in GLOBAL_IMAGE_STORAGE:
                page_images = GLOBAL_IMAGE_STORAGE[image_key]
                for img_info in page_images:
                    # Create unique identifier for image
                    img_id = f"{source}_page{page}_img{img_info['image_index']}"
                    
                    if img_id not in seen_images:
                        images.append({
                            "id": img_id,
                            "source": source,
                            "page": page,
                            "doc_type": doc.metadata.get("doc_type"),
                            "image_data": img_info["image_data"],
                            "width": img_info["width"],
                            "height": img_info["height"],
                            "image_index": img_info["image_index"]
                        })
                        seen_images.add(img_id)
    
    return images

# ─── 7. Initialize RAG system ────────────────────────────────────────────────
def initialize_rag_system():
    """Initialize the RAG system by downloading PDFs and building vector stores"""
    print("Initializing RAG system...")
    
    # Download PDFs from GCS
    pdf_folder = download_pdfs_from_gcs(GCS_BUCKET_NAME, GCS_PDF_PREFIX)
    
    # Load and split PDFs
    fine_docs, coarse_docs = load_and_split_pdfs(pdf_folder)
    
    # Create vector stores
    fine_db, coarse_db = get_vectorstores(fine_docs, coarse_docs)
    
    # Create traditional retriever (keeping for backward compatibility)
    traditional_retriever = ContextExpandingHybridRetriever(fine_db, coarse_db)
    
    # Create multi-stage retriever
    multi_stage_retriever = MultiStageRetriever(fine_db, coarse_db)
    
    # Create LLM
    llm = VertexAIGeminiLLM()
    
    print("RAG system initialized successfully!")
    return traditional_retriever, multi_stage_retriever, llm

# ─── 8. Query processing function for API ────────────────────────────────────
def process_query(query: str, retriever, llm, conversation_history, debug_mode: bool = False, use_multi_stage: bool = True, status_callback=None):
    """Process a single query and return response with optional debug info"""
    
    # Use multi-stage retrieval if available and enabled
    if use_multi_stage and hasattr(retriever, 'retrieve_and_validate'):
        return process_query_multi_stage(query, retriever, conversation_history, debug_mode, status_callback)
    else:
        return process_query_traditional(query, retriever, llm, conversation_history, debug_mode)

def process_query_multi_stage(query: str, multi_stage_retriever, conversation_history, debug_mode: bool = False, status_callback=None):
    """Process query using multi-stage retrieval with validation and caching"""
    
    print(f"\n=== Processing Query with Multi-Stage Retrieval ===")
    print(f"Query: {query}")
    
    # Set up status callback for live updates
    if status_callback:
        multi_stage_retriever.set_status_callback(status_callback)
    
    # Query reformulation logic for follow-up queries
    effective_query = query
    if is_follow_up_query(query) and len(conversation_history.messages) > 0:
        # For multi-stage retrieval, we'll include conversation context in the query
        history_context = conversation_history.get_conversation_context(n=2)
        effective_query = f"Based on our conversation about: {history_context}\n\nQuestion: {query}"
    
    # Step 1: Multi-stage retrieval and validation
    start_time = time.time()
    validated_chunks, scores, cache_hit = multi_stage_retriever.retrieve_and_validate(effective_query)
    retrieval_time = time.time() - start_time
    
    if not validated_chunks:
        # Fallback response
        conversation_history.add_user_message(query)
        fallback_answer = "I couldn't find relevant information to answer your question. Please try rephrasing your query or provide more specific details."
        conversation_history.add_ai_message(fallback_answer)
        
        return {
            "answer": fallback_answer,
            "sources": [],
            "images": [],
            "debug_info": {
                "retrieval_method": "multi_stage",
                "cache_hit": cache_hit,
                "retrieval_time": retrieval_time,
                "chunks_found": 0,
                "validation_scores": []
            } if debug_mode else None
        }
    
    # Step 2: Generate answer using validated chunks
    start_answer_time = time.time()
    answer = multi_stage_retriever.generate_answer(effective_query, validated_chunks)
    answer_time = time.time() - start_answer_time
    
    # Extract images from validated chunks
    images = extract_images_from_chunks(validated_chunks)
    
    # Get sources with document type information
    sources = []
    for chunk in validated_chunks:
        source_info = {
            "filename": chunk.metadata.get("source"),
            "doc_type": chunk.metadata.get("doc_type"),
            "page": chunk.metadata.get("page")
        }
        if source_info not in sources:
            sources.append(source_info)
    
    # Prepare debug info
    debug_info = None
    if debug_mode:
        debug_info = {
            "retrieval_method": "multi_stage",
            "cache_hit": cache_hit,
            "retrieval_time": retrieval_time,
            "answer_generation_time": answer_time,
            "total_time": retrieval_time + answer_time,
            "chunks_found": len(validated_chunks),
            "validation_scores": scores,
            "chunks": [
                {
                    "source": chunk.metadata.get("source"),
                    "page": chunk.metadata.get("page"),
                    "doc_type": chunk.metadata.get("doc_type"),
                    "score": scores[i] if i < len(scores) else 0,
                    "preview": chunk.page_content[:200].replace('\n', ' ')
                }
                for i, chunk in enumerate(validated_chunks)
            ],
            "cache_stats": multi_stage_retriever.get_cache_stats()
        }
    
    # Update conversation history
    conversation_history.add_user_message(query)
    conversation_history.add_ai_message(answer)
    
    print(f"Multi-stage query processing completed in {retrieval_time + answer_time:.2f}s")
    
    return {
        "answer": answer,
        "sources": sources,
        "images": images,
        "debug_info": debug_info
    }

def process_query_traditional(query: str, retriever, llm, conversation_history, debug_mode: bool = False):
    """Process a single query using traditional retrieval method"""
    
    # Query reformulation logic
    if is_follow_up_query(query) and len(conversation_history.messages) > 0:
        reformulation_prompt = PromptTemplate.from_template(
            """Given the conversation history and a follow-up question, reformulate the question to be self-contained and specific.

Conversation history:
{history}

Follow-up question: {question}

Reformulated question (be specific and include context from the conversation):"""
        )
        reformulation_chain = reformulation_prompt | llm | StrOutputParser()
        
        history_context = conversation_history.get_conversation_context()
        reformulated_q = reformulation_chain.invoke({
            "history": history_context,
            "question": query
        })
        effective_query = reformulated_q
    else:
        effective_query = query
    
    # Detect if user is asking for specific document type
    def detect_document_type(query_text: str) -> str:
        """Detect if user is asking for emails or manuals specifically"""
        lower_q = query_text.lower()
        
        email_keywords = ["email", "emails", "uhd", "correspondence", "message", "communication"]
        manual_keywords = ["manual", "manuals", "documentation", "guide", "handbook", "instruction"]
        
        email_score = sum(1 for keyword in email_keywords if keyword in lower_q)
        manual_score = sum(1 for keyword in manual_keywords if keyword in lower_q)
        
        if email_score > 0 and email_score > manual_score:
            return "email"
        elif manual_score > 0 and manual_score > email_score:
            return "manual"
        else:
            return None
    
    # Check for document type filtering
    doc_type_filter = detect_document_type(effective_query)
    
    # Determine if this is a follow-up query for context expansion
    is_followup = is_follow_up_query(query)
    
    # Get documents (with optional filtering and expanded context for follow-ups)
    if doc_type_filter:
        if is_followup:
            # Temporarily increase context expansion for follow-up queries
            original_before = retriever.fine_db._expand_before if hasattr(retriever.fine_db, '_expand_before') else EXPAND_CONTEXT_BEFORE
            original_after = retriever.fine_db._expand_after if hasattr(retriever.fine_db, '_expand_after') else EXPAND_CONTEXT_AFTER
            
            # Get documents with expanded context
            docs = retriever.get_relevant_documents_by_type_with_expansion(
                effective_query, doc_type_filter, 
                before=EXPAND_CONTEXT_BEFORE + 1, 
                after=EXPAND_CONTEXT_AFTER + 1
            )
        else:
            docs = retriever.get_relevant_documents_by_type(effective_query, doc_type_filter)
    else:
        if is_followup:
            # Get documents with expanded context for follow-up queries
            docs = retriever.get_relevant_documents_with_expansion(
                effective_query,
                before=EXPAND_CONTEXT_BEFORE + 1,
                after=EXPAND_CONTEXT_AFTER + 1
            )
        else:
            docs = retriever.get_relevant_documents(effective_query)
    
    print(f"DEBUG: Query: {effective_query}")
    print(f"DEBUG: Is follow-up query: {is_followup}")
    print(f"DEBUG: Doc type filter: {doc_type_filter}")
    print(f"DEBUG: Context expansion - Before: {EXPAND_CONTEXT_BEFORE + (1 if is_followup else 0)}, After: {EXPAND_CONTEXT_AFTER + (1 if is_followup else 0)}")
    print(f"DEBUG: Retrieved {len(docs)} documents")
    for i, doc in enumerate(docs[:3]):  # Show first 3 docs
        print(f"DEBUG: Doc {i+1} - Source: {doc.metadata.get('source')}, Page: {doc.metadata.get('page')}")
        print(f"DEBUG: Doc {i+1} - Content preview: {doc.page_content[:100]}...")
    
    conversation_history.add_chunks_to_history(effective_query, docs)
    
    # Get previous chunks if this is a follow-up query
    if is_followup:
        previous_chunks = conversation_history.get_recent_chunks(n=2)
    else:
        previous_chunks = []
    
    # Extract images from retrieved chunks (for separate output, not LLM)
    all_chunks_for_images = docs[:]
    if previous_chunks:
        all_chunks_for_images.extend(previous_chunks)
    images = extract_images_from_chunks(all_chunks_for_images)
    
    # Prepare debug info
    debug_info = None
    if debug_mode:
        debug_info = {
            "retrieval_method": "traditional",
            "current_chunks": [],
            "previous_chunks": [],
            "doc_type_filter": doc_type_filter,
            "images_found": len(images)
        }
        
        for i, d in enumerate(docs, 1):
            debug_info["current_chunks"].append({
                "index": i,
                "chunk_index": d.metadata.get("chunk_index"),
                "total_chunks": d.metadata.get("total_chunks"),
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "doc_type": d.metadata.get("doc_type"),
                "has_image": d.metadata.get("has_image", False),
                "image_count": d.metadata.get("image_count", 0),
                "preview": d.page_content[:200].replace('\n', ' ')
            })
        
        for i, d in enumerate(previous_chunks, 1):
            debug_info["previous_chunks"].append({
                "index": i,
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "doc_type": d.metadata.get("doc_type"),
                "has_image": d.metadata.get("has_image", False),
                "image_count": d.metadata.get("image_count", 0),
                "preview": d.page_content[:100].replace('\n', ' ')
            })
    
    def format_docs(docs):
        if not docs:
            return "No relevant context found."
        return "\n\n---\n\n".join(doc.page_content for doc in docs)
    
    # Format the context strings
    context_str = format_docs(docs)
    previous_context_str = format_docs(previous_chunks) if previous_chunks else "None"
    
    # Debug logging to check context
    print(f"DEBUG: Retrieved {len(docs)} documents")
    print(f"DEBUG: Context length: {len(context_str)} characters")
    print(f"DEBUG: Context preview: {context_str[:200]}...")
    print(f"DEBUG: Previous context length: {len(previous_context_str)} characters")
    
    # Additional debug: Check if context is actually populated
    if not docs:
        print("WARNING: No documents retrieved for query!")
    if context_str == "No relevant context found.":
        print("WARNING: Context string indicates no relevant context found!")
    else:
        print(f"DEBUG: Context contains data from {len(docs)} documents")
    
    # Create prompt template without f-string formatting to preserve template variables
    system_message = """You are a helpful assistant for answering questions about documents. 
Use the following pieces of retrieved context to answer the question. 
If relevant, you may also reference the previous context chunks.

The documents are classified as either 'email' (files starting with 'UHD') or 'manual' (all other files).
When referencing sources, please mention the document type when relevant.

Current Context: {context}

Previous Context (if relevant): {previous_context}"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])
    
    # Create the chain with direct input
    rag_chain = prompt | llm | StrOutputParser()
    
    # Debug: Show what's being sent to the LLM
    input_data = {
        "context": context_str,
        "previous_context": previous_context_str,
        "question": effective_query,
        "history": conversation_history.get_messages()
    }
    
    print(f"DEBUG: Input to LLM chain:")
    print(f"  - Question: {effective_query}")
    print(f"  - Context length: {len(input_data['context'])} chars")
    print(f"  - Previous context length: {len(input_data['previous_context'])} chars")
    print(f"  - History messages: {len(input_data['history'])}")
    print(f"DEBUG: Actual context being sent: {context_str[:300]}...")
    
    # Get answer
    answer = rag_chain.invoke(input_data)
    
    print(f"DEBUG: LLM Response: {answer[:200]}...")
    
    # Get sources with document type information
    sources = []
    for d in docs:
        source_info = {
            "filename": d.metadata.get("source"),
            "doc_type": d.metadata.get("doc_type"),
            "page": d.metadata.get("page")
        }
        if source_info not in sources:
            sources.append(source_info)
    
    if previous_chunks:
        for d in previous_chunks:
            source_info = {
                "filename": d.metadata.get("source"),
                "doc_type": d.metadata.get("doc_type"),
                "page": d.metadata.get("page")
            }
            if source_info not in sources:
                sources.append(source_info)
    
    # Update conversation history
    conversation_history.add_user_message(query)
    conversation_history.add_ai_message(answer)
    
    return {
        "answer": answer,
        "sources": sources,
        "images": images,
        "debug_info": debug_info
    }