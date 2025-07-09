import os
import time
from typing import Any, List, Tuple, Set, Dict
from dotenv import load_dotenv
import fitz  # PyMuPDF for image detection

# community imports to avoid deprecation warnings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever, Document
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever as CoreBaseRetriever

from openai import OpenAI, AuthenticationError

# ─── 1. Load config ─────────────────────────────────────────────────────────
load_dotenv()
API_KEY  = os.getenv("CBORG_API_KEY")
BASE_URL = os.getenv("CBORG_BASE_URL", "https://api.cborg.lbl.gov")
if not API_KEY:
    raise ValueError("Set CBORG_API_KEY in your .env file")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ─── 2. Chunking parameters ─────────────────────────────────────────────────
FINE_CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "150"))  # Changed from 300 to 150
FINE_CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "50"))
COARSE_CHUNK_SIZE    = int(os.getenv("COARSE_CHUNK_SIZE", "1200"))
COARSE_CHUNK_OVERLAP = int(os.getenv("COARSE_CHUNK_OVERLAP", "200"))
SEPARATORS           = ["\n\n", "\n", " ", ""]
CONTEXT_WINDOW_SIZE  = 3  # Number of previous queries to keep chunks for

# ─── 3. LLM wrapper for Google Gemini-Flash via CBorg ────────────────────────
class CBorgLLM(LLM):
    client: OpenAI
    model_name: str = "google/gemini-flash"
    callbacks: Any = None
    verbose: bool = False

    def __init__(self, client: OpenAI, model_name: str = None, callbacks: Any = None, verbose: bool = False):
        super().__init__(client=client, callbacks=callbacks, verbose=verbose)
        self.client = client
        if model_name:
            self.model_name = model_name

    @property
    def _llm_type(self) -> str:
        return "cborg-gemini"

    def _call(self, prompt: str, stop=None) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return resp.choices[-1].message.content

# ─── 4. Embeddings wrapper for text-embedding-004 via CBorg with retry ─────
class CBorgEmbeddings(Embeddings):
    client: OpenAI
    model_name: str = "google/text-embedding-004"

    def __init__(self, client: OpenAI, model_name: str = None):
        self.client = client
        if model_name:
            self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            attempts = 3
            while True:
                try:
                    resp = self.client.embeddings.create(
                        model=self.model_name,
                        input=batch
                    )
                    embeddings.extend([d.embedding for d in resp.data])
                    break
                except AuthenticationError as e:
                    if attempts > 0 and "remaining connection slots" in str(e):
                        attempts -= 1
                        time.sleep(5)
                        continue
                    raise
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        try:
            resp = self.client.embeddings.create(model=self.model_name, input=text)
        except AuthenticationError:
            time.sleep(2)
            resp = self.client.embeddings.create(model=self.model_name, input=text)
        return resp.data[0].embedding

# ─── 5. PDF → Documents → Hierarchical Chunks with image flag ──────────────
PDF_FOLDER        = "pdfs"
CHROMA_BASE_DIR   = "chroma_db"
CHROMA_FINE_DIR   = os.path.join(CHROMA_BASE_DIR, "fine")
CHROMA_COARSE_DIR = os.path.join(CHROMA_BASE_DIR, "coarse")

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
        fine_docs.extend(fine_splitter.split_documents(pages))
        coarse_docs.extend(coarse_splitter.split_documents(pages))

    return fine_docs, coarse_docs

# ─── 6. Build or reload two Chroma vector stores ─────────────────────────────
def get_vectorstores(fine_docs: List[Document], coarse_docs: List[Document]):
    os.makedirs(CHROMA_FINE_DIR, exist_ok=True)
    os.makedirs(CHROMA_COARSE_DIR, exist_ok=True)
    emb = CBorgEmbeddings(client=client)
    if not os.listdir(CHROMA_FINE_DIR):
        fine_db = Chroma.from_documents(fine_docs, emb, persist_directory=CHROMA_FINE_DIR)
    else:
        fine_db = Chroma(persist_directory=CHROMA_FINE_DIR, embedding_function=emb)
    if not os.listdir(CHROMA_COARSE_DIR):
        coarse_db = Chroma.from_documents(coarse_docs, emb, persist_directory=CHROMA_COARSE_DIR)
    else:
        coarse_db = Chroma(persist_directory=CHROMA_COARSE_DIR, embedding_function=emb)
    return fine_db, coarse_db

# ─── 7. Hybrid retriever with proper implementation ─────────────────────────
class HybridRetriever:
    """Custom hybrid retriever that doesn't inherit from BaseRetriever"""
    def __init__(self, fine_db, coarse_db):
        self.fine_retriever = fine_db.as_retriever(search_kwargs={"k": 3})
        self.coarse_retriever = coarse_db.as_retriever(search_kwargs={"k": 2})
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Main retrieval method"""
        lower_q = query.lower()
        procedural = any(k in lower_q for k in ["how", "procedure", "steps", "process"])
        retriever = self.coarse_retriever if procedural else self.fine_retriever
        return retriever.get_relevant_documents(query)
    
    def invoke(self, input: str, config=None) -> List[Document]:
        """Invoke method for compatibility with chains"""
        return self.get_relevant_documents(input)
    
    async def ainvoke(self, input: str, config=None) -> List[Document]:
        """Async invoke method for compatibility"""
        return self.get_relevant_documents(input)

# ─── 8. Enhanced conversation history with chunk storage ────────────────────
class ConversationHistory:
    def __init__(self):
        self.messages = []
        self.awaiting_clarification = False
        self.original_question = None
        self.chunk_history = []  # List of (question, List[Document]) tuples
        self.max_chunk_history = CONTEXT_WINDOW_SIZE
    
    def add_user_message(self, message: str):
        self.messages.append(HumanMessage(content=message))
    
    def add_ai_message(self, message: str):
        self.messages.append(AIMessage(content=message))
    
    def add_chunks_to_history(self, question: str, chunks: List[Document]):
        """Store chunks associated with a question"""
        self.chunk_history.append((question, chunks))
        # Keep only the last N entries
        if len(self.chunk_history) > self.max_chunk_history:
            self.chunk_history.pop(0)
    
    def get_recent_chunks(self, n: int = 2) -> List[Document]:
        """Get chunks from the last n queries"""
        all_chunks = []
        for _, chunks in self.chunk_history[-n:]:
            all_chunks.extend(chunks)
        # Remove duplicates based on content
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
        """Get recent conversation as context string"""
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

# ─── 9. Query reformulation for vague follow-ups ────────────────────────────
def is_follow_up_query(query: str) -> bool:
    """Check if query is a vague follow-up"""
    follow_up_phrases = [
        "tell me more", "more about", "what else", "anything else",
        "more details", "more information", "explain that",
        "elaborate", "go on", "continue", "what about",
        "how about", "and", "also", "furthermore", "additionally"
    ]
    lower_q = query.lower().strip()
    return any(phrase in lower_q for phrase in follow_up_phrases) or len(lower_q.split()) <= 3

# ─── 10. Build conversational QA chain with improvements ────────────────────
def build_qa_chain():
    fine_docs, coarse_docs = load_and_split_pdfs(PDF_FOLDER)
    fine_db, coarse_db = get_vectorstores(fine_docs, coarse_docs)
    llm = CBorgLLM(client=client)
    retriever = HybridRetriever(fine_db, coarse_db)
    
    # Query reformulation chain
    reformulation_prompt = PromptTemplate.from_template(
        """Given the conversation history and a follow-up question, reformulate the question to be self-contained and specific.

Conversation history:
{history}

Follow-up question: {question}

Reformulated question (be specific and include context from the conversation):"""
    )
    reformulation_chain = reformulation_prompt | llm | StrOutputParser()
    
    # Enhanced conversational prompt with previous chunks
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant for answering questions about documents. 
        Use the following pieces of retrieved context to answer the question. 
        If relevant, you may also reference the previous context chunks.
        
        Current Context: {context}
        
        Previous Context (if relevant): {previous_context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])
    
    # Create the chain using the modern approach
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
    
    # Enhanced clarifier chain
    clarify_check_prompt = PromptTemplate.from_template(
        """Analyze if this question needs clarification to provide a good answer:
        Question: "{question}"
        
        Conversation context:
        {history}
        
        Does this question need clarification? Consider:
        - Is it too vague or ambiguous?
        - Are there multiple possible interpretations?
        - Is critical information missing (like model numbers, specific components, etc.)?
        
        If clarification is needed, respond with:
        CLARIFY: [your specific clarifying question]
        
        If no clarification is needed, respond with:
        PROCEED
        
        Be selective - only ask for clarification when truly necessary."""
    )
    
    clarify_chain = clarify_check_prompt | llm | StrOutputParser()
    
    return rag_chain, retriever, clarify_chain, reformulation_chain

# ─── 11. Interactive CLI loop with all improvements ─────────────────────────
if __name__ == "__main__":
    rag_chain, retriever, clarify_chain, reformulation_chain = build_qa_chain()
    conversation_history = ConversationHistory()
    
    print("🎉 Attocube support is ready! Type 'exit' to quit.")
    
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("exit", "quit"): 
            break
        
        try:
            # Check if we're in clarification mode
            if conversation_history.awaiting_clarification:
                # Combine original question with clarification
                enhanced_question = f"{conversation_history.original_question} Specifically, {q}"
                conversation_history.clear_clarification_mode()
                
                # Add the clarification to history
                conversation_history.add_user_message(q)
                
                # Process the enhanced question
                docs = retriever.get_relevant_documents(enhanced_question)
                conversation_history.add_chunks_to_history(enhanced_question, docs)
                
                # Get previous relevant chunks
                previous_chunks = conversation_history.get_recent_chunks(n=1)
                
                # DEBUG: Print retrieved chunks
                print("\n📋 DEBUG - Retrieved chunks:")
                print("-" * 50)
                print("CURRENT CHUNKS:")
                for i, d in enumerate(docs, 1):
                    print(f"Chunk {i}:")
                    print(f"  — Source: {d.metadata['source']}, Page: {d.metadata.get('page', '?')}")
                    print(f"  — Content preview: {d.page_content[:200].replace(chr(10), ' ')}...")
                
                if previous_chunks:
                    print("\nPREVIOUS CONTEXT CHUNKS:")
                    for i, d in enumerate(previous_chunks, 1):
                        print(f"Previous Chunk {i}:")
                        print(f"  — Source: {d.metadata['source']}, Page: {d.metadata.get('page', '?')}")
                        print(f"  — Content preview: {d.page_content[:100].replace(chr(10), ' ')}...")
                print("-" * 50)
                
                # Get answer with both current and previous context
                answer = rag_chain.invoke({
                    "current_docs": docs,
                    "previous_docs": previous_chunks,
                    "question": enhanced_question,
                    "history": conversation_history.get_messages()[:-1]
                })
                
                sources = {d.metadata.get("source") for d in docs}
                conversation_history.add_ai_message(answer)
                
                print(f"\nBot: {answer}\n")
                print(f"Sources: {', '.join(sorted(sources))}\n")
                
            else:
                # Check if this is a vague follow-up query
                if is_follow_up_query(q) and len(conversation_history.messages) > 0:
                    
                    # Reformulate the query
                    history_context = conversation_history.get_conversation_context()
                    reformulated_q = reformulation_chain.invoke({
                        "history": history_context,
                        "question": q
                    })
                    
                    print(f"📝 Reformulated as: {reformulated_q}")
                    
                    # Use reformulated query for retrieval
                    effective_query = reformulated_q
                else:
                    effective_query = q
                
                # Check if clarification is needed
                clarify_response = clarify_chain.invoke({
                    "question": effective_query,
                    "history": conversation_history.get_conversation_context()
                })
                
                if clarify_response.strip().startswith("CLARIFY:"):
                    # Extract the clarifying question
                    clarifying_question = clarify_response.replace("CLARIFY:", "").strip()
                    conversation_history.set_clarification_mode(q)
                    conversation_history.add_user_message(q)
                    conversation_history.add_ai_message(clarifying_question)
                    
                    print(f"\nBot: {clarifying_question}")
                    print("(I need more information to give you the best answer)")
                    
                else:
                    # No clarification needed, proceed normally
                    conversation_history.add_user_message(q)
                    
                    # Get current documents
                    docs = retriever.get_relevant_documents(effective_query)
                    conversation_history.add_chunks_to_history(effective_query, docs)
                    
                    # Get previous relevant chunks for follow-ups
                    previous_chunks = []
                    if is_follow_up_query(q):
                        previous_chunks = conversation_history.get_recent_chunks(n=2)
                    
                    # DEBUG: Print retrieved chunks
                    print("\n📋 DEBUG - Retrieved chunks:")
                    print("-" * 50)
                    print("CURRENT CHUNKS:")
                    for i, d in enumerate(docs, 1):
                        print(f"Chunk {i}:")
                        print(f"  — Source: {d.metadata['source']}, Page: {d.metadata.get('page', '?')}")
                        print(f"  — Content preview: {d.page_content[:200].replace(chr(10), ' ')}...")
                    
                    if previous_chunks:
                        print("\nPREVIOUS CONTEXT CHUNKS (from recent queries):")
                        for i, d in enumerate(previous_chunks, 1):
                            print(f"Previous Chunk {i}:")
                            print(f"  — Source: {d.metadata['source']}, Page: {d.metadata.get('page', '?')}")
                            print(f"  — Content preview: {d.page_content[:100].replace(chr(10), ' ')}...")
                    print("-" * 50)
                    
                    # Get answer with context
                    answer = rag_chain.invoke({
                        "current_docs": docs,
                        "previous_docs": previous_chunks,
                        "question": effective_query,
                        "history": conversation_history.get_messages()[:-1]
                    })
                    
                    sources = {d.metadata.get("source") for d in docs}
                    if previous_chunks:
                        sources.update(d.metadata.get("source") for d in previous_chunks)
                    
                    conversation_history.add_ai_message(answer)
                    
                    print(f"\nBot: {answer}\n")
                    print(f"Sources: {', '.join(sorted(sources))}\n")
                    
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            # Reset clarification mode on error
            conversation_history.clear_clarification_mode()
            # Remove the failed message from history if it exists
            if conversation_history.messages:
                conversation_history.messages.pop()