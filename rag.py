import os
import time
from typing import Any, List, Tuple, Set, Dict, Optional
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
                "max_output_tokens": 4096,
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
    fine_splitter = RecursiveCharacterTextSplitter(
        chunk_size=FINE_CHUNK_SIZE,
        chunk_overlap=FINE_CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )
    coarse_splitter = RecursiveCharacterTextSplitter(
        chunk_size=COARSE_CHUNK_SIZE,
        chunk_overlap=COARSE_CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )
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
        
        # Split and add chunk indices
        fine_chunks = fine_splitter.split_documents(pages)
        coarse_chunks = coarse_splitter.split_documents(pages)
        
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
        self.fine_retriever = fine_db.as_retriever(search_kwargs={"k": 10})
        self.coarse_retriever = coarse_db.as_retriever(search_kwargs={"k": 5})
    
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
    
    # Create retriever
    retriever = ContextExpandingHybridRetriever(fine_db, coarse_db)
    
    # Create LLM
    llm = VertexAIGeminiLLM()
    
    print("RAG system initialized successfully!")
    return retriever, llm

# ─── 8. Query processing function for API ────────────────────────────────────
def process_query(query: str, retriever, llm, conversation_history, debug_mode: bool = False, status_callback=None):
    """Process a single query and return response with optional debug info"""
    
    if status_callback:
        status_callback("searching", "Searching knowledge base...")
    
    # Query reformulation logic
    if is_follow_up_query(query) and len(conversation_history.messages) > 0:
        if status_callback:
            status_callback("reformulating", "Understanding context...")
            
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
    
    if status_callback:
        if is_followup:
            status_callback("expanding", "Expanding context...")
        else:
            status_callback("searching", "Searching knowledge base...")
    
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
    
    if status_callback:
        status_callback("generating", "Generating answer...")
    
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