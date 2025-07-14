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

from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

# ─── 1. Initialize Vertex AI ─────────────────────────────────────────────────
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-project-id")
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

# ─── 3. LLM wrapper for Vertex AI Gemini ────────────────────────────────────
class VertexAIGeminiLLM(LLM):
    model: Optional[GenerativeModel] = None  # Add default value
    model_name: str = "gemini-2.0-flash-001"
    
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

        # Detect image pages
        doc_pdf = fitz.open(path)
        pages_with_images = {i+1 for i, pg in enumerate(doc_pdf) if pg.get_images()}
        doc_pdf.close()

        pages = PyPDFLoader(path).load()
        for i, doc in enumerate(pages, start=1):
            doc.metadata.update({
                "source": fname,
                "page": i,
                "has_image": i in pages_with_images
            })
        
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
def process_query(query: str, retriever, llm, conversation_history, debug_mode: bool = False):
    """Process a single query and return response with optional debug info"""
    
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
    
    # Get documents
    docs = retriever.get_relevant_documents(effective_query)
    conversation_history.add_chunks_to_history(effective_query, docs)
    
    # Get previous chunks for follow-ups
    previous_chunks = []
    if is_follow_up_query(query):
        previous_chunks = conversation_history.get_recent_chunks(n=2)
    
    # Prepare debug info
    debug_info = None
    if debug_mode:
        debug_info = {
            "current_chunks": [],
            "previous_chunks": []
        }
        
        for i, d in enumerate(docs, 1):
            debug_info["current_chunks"].append({
                "index": i,
                "chunk_index": d.metadata.get("chunk_index"),
                "total_chunks": d.metadata.get("total_chunks"),
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "preview": d.page_content[:200].replace('\n', ' ')
            })
        
        for i, d in enumerate(previous_chunks, 1):
            debug_info["previous_chunks"].append({
                "index": i,
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "preview": d.page_content[:100].replace('\n', ' ')
            })
    
    # Create prompt and get response
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant for answering questions about documents. 
        Use the following pieces of retrieved context to answer the question. 
        If relevant, you may also reference the previous context chunks.
        
        Current Context: {context}
        
        Previous Context (if relevant): {previous_context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])
    
    def format_docs(docs):
        if not docs:
            return "No relevant context found."
        return "\n\n---\n\n".join(doc.page_content for doc in docs)
    
    def get_combined_context(data):
        current_docs = data["current_docs"]
        previous_docs = data.get("previous_docs", [])
        return {
            "context": format_docs(current_docs),
            "previous_context": format_docs(previous_docs) if previous_docs else "None",
            "question": data["question"],
            "history": data.get("history", [])
        }
    
    rag_chain = (
        RunnablePassthrough()
        | get_combined_context
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Get answer
    answer = rag_chain.invoke({
        "current_docs": docs,
        "previous_docs": previous_chunks,
        "question": effective_query,
        "history": conversation_history.get_messages()
    })
    
    # Get sources
    sources = list({d.metadata.get("source") for d in docs})
    if previous_chunks:
        sources.extend({d.metadata.get("source") for d in previous_chunks})
    sources = list(set(sources))
    
    # Update conversation history
    conversation_history.add_user_message(query)
    conversation_history.add_ai_message(answer)
    
    return {
        "answer": answer,
        "sources": sources,
        "debug_info": debug_info
    }