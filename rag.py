import os
import time
from typing import Any, List, Tuple
from dotenv import load_dotenv

# community imports to avoid deprecation warnings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever, Document

from openai import OpenAI, AuthenticationError

# ─── 1. Load config ─────────────────────────────────────────────────────────
load_dotenv()
API_KEY  = os.getenv("CBORG_API_KEY")
BASE_URL = os.getenv("CBORG_BASE_URL", "https://api.cborg.lbl.gov")
if not API_KEY:
    raise ValueError("Set CBORG_API_KEY in your .env file")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ─── 2. Chunking parameters ─────────────────────────────────────────────────
FINE_CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "300"))
FINE_CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "50"))
COARSE_CHUNK_SIZE    = int(os.getenv("COARSE_CHUNK_SIZE", "1200"))
COARSE_CHUNK_OVERLAP = int(os.getenv("COARSE_CHUNK_OVERLAP", "200"))
SEPARATORS           = ["\n\n", "\n", " ", ""]

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
        batch_size = 50  # reduce batch size to ease DB load
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
                    msg = str(e)
                    if "remaining connection slots" in msg and attempts > 0:
                        attempts -= 1
                        time.sleep(5)
                        continue
                    raise
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        try:
            resp = self.client.embeddings.create(model=self.model_name, input=text)
        except AuthenticationError as e:
            # single-item retry
            time.sleep(2)
            resp = self.client.embeddings.create(model=self.model_name, input=text)
        return resp.data[0].embedding

# ─── 5. PDF → Documents → Hierarchical Chunks ───────────────────────────────
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
        pages = PyPDFLoader(path).load()
        for i, doc in enumerate(pages, start=1):
            doc.metadata.update({"source": fname, "page": i})
        fine_docs.extend(fine_splitter.split_documents(pages))
        coarse_docs.extend(coarse_splitter.split_documents(pages))
    return fine_docs, coarse_docs

# ─── 6. Build or reload two Chroma vector stores ─────────────────────────────
def get_vectorstores(fine_docs: List[Document], coarse_docs: List[Document]):
    os.makedirs(CHROMA_FINE_DIR, exist_ok=True)
    os.makedirs(CHROMA_COARSE_DIR, exist_ok=True)
    emb = CBorgEmbeddings(client=client)
    # fine index
    if not os.listdir(CHROMA_FINE_DIR):
        fine_db = Chroma.from_documents(
            documents=fine_docs,
            embedding=emb,
            persist_directory=CHROMA_FINE_DIR
        )
    else:
        fine_db = Chroma(
            persist_directory=CHROMA_FINE_DIR,
            embedding_function=emb
        )
    # coarse index
    if not os.listdir(CHROMA_COARSE_DIR):
        coarse_db = Chroma.from_documents(
            documents=coarse_docs,
            embedding=emb,
            persist_directory=CHROMA_COARSE_DIR
        )
    else:
        coarse_db = Chroma(
            persist_directory=CHROMA_COARSE_DIR,
            embedding_function=emb
        )
    return fine_db, coarse_db

# ─── 7. Hybrid retriever with required fields ───────────────────────────────
class HybridRetriever(BaseRetriever):
    """
    Routes queries: short facts to fine retriever and procedures to coarse retriever.
    """
    fine_retriever: BaseRetriever
    coarse_retriever: BaseRetriever

    def __init__(self, fine_db, coarse_db):
        # build the actual retrievers first
        fine = fine_db.as_retriever(search_kwargs={"k": 3})
        coarse = coarse_db.as_retriever(search_kwargs={"k": 2})
        # initialize BaseRetriever with these fields
        super().__init__(fine_retriever=fine, coarse_retriever=coarse)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        lower_q = query.lower()
        procedural = any(keyword in lower_q for keyword in ["how", "procedure", "steps", "process"])
        retriever = self.coarse_retriever if procedural else self.fine_retriever
        return retriever.get_relevant_documents(query)

# ─── 8. Wire up the RetrievalQA chain ──────────────────────────────────────── Wire up the RetrievalQA chain ────────────────────────────────────────
def build_qa_chain():
    fine_docs, coarse_docs = load_and_split_pdfs(PDF_FOLDER)
    fine_db, coarse_db = get_vectorstores(fine_docs, coarse_docs)
    llm = CBorgLLM(client=client)
    retriever = HybridRetriever(fine_db, coarse_db)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

# ─── 9. Interactive CLI loop ─────────────────────────────────────────────────
if __name__ == "__main__":
    qa = build_qa_chain()
    print("🎉 RAG bot ready! Type ‘exit’ to quit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("exit", "quit"): break
        result = qa.invoke({"query": q})
        print(f"\nBot: {result['result']}")
