import os
from typing import Any, List, Optional
from dotenv import load_dotenv

# community imports to avoid deprecation warnings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # updated import from langchain-chroma package

from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA

from openai import OpenAI

# ─── 1. Load config ─────────────────────────────────────────────────────────
load_dotenv()
API_KEY  = os.getenv("CBORG_API_KEY")
BASE_URL = os.getenv("CBORG_BASE_URL", "https://api.cborg.lbl.gov")
if not API_KEY:
    raise ValueError("Set CBORG_API_KEY in your .env file")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ─── 2. Chunking parameters (tweak these!) ─────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "500"))       # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))   # characters to overlap
SEPARATORS    = ["\n\n", "\n", " ", ""]             # preferred split separators

# ─── 3. LLM wrapper for Google Gemini-Flash via CBorg ────────────────────────
class CBorgLLM(LLM):
    client:     OpenAI
    model_name: str      = "google/gemini-flash"
    callbacks:  Any      = None
    verbose:    bool     = False

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

# ─── 4. Embeddings wrapper for text-embedding-004 via CBorg ────────────────
class CBorgEmbeddings(Embeddings):
    client:     OpenAI
    model_name: str = "google/text-embedding-004"

    def __init__(self, client: OpenAI, model_name: str = None):
        self.client = client
        if model_name:
            self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self.client.embeddings.create(
                model=self.model_name,
                input=batch
            )
            embeddings.extend([d.embedding for d in resp.data])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        return resp.data[0].embedding

# ─── 5. PDF → Documents → Chunks ─────────────────────────────────────────────
PDF_FOLDER = "pdfs"
CHROMA_DIR = "chroma_db"

def load_and_split_pdfs(folder: str) -> List[Any]:
    """
    Load all PDFs from `folder`, split into chunks with overlap using RecursiveCharacterTextSplitter.
    Metadata includes source filename and page number for better retrieval accuracy.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len
    )
    all_docs = []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(folder, fname)
        loader = PyPDFLoader(path)
        pages = loader.load()
        # tag each page with filename and page index
        for i, doc in enumerate(pages, start=1):
            doc.metadata.update({"source": fname, "page": i})
        # split and keep metadata
        chunks = splitter.split_documents(pages)
        print(f"Loaded {len(chunks)} chunks from {fname}")
        all_docs.extend(chunks)
    print(f"Total chunks loaded: {len(all_docs)}")
    return all_docs

# ─── 6. Build or reload your Chroma vector store ─────────────────────────────
def get_vectorstore(docs: List[Any]) -> Chroma:
    emb = CBorgEmbeddings(client=client)
    if not os.path.isdir(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        db = Chroma.from_documents(
            documents=docs,
            embedding=emb,
            persist_directory=CHROMA_DIR
        )
    else:
        db = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=emb
        )
    return db

# ─── 7. Wire up the RetrievalQA chain ────────────────────────────────────────
def build_qa_chain():
    docs      = load_and_split_pdfs(PDF_FOLDER)
    vector_db = get_vectorstore(docs)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    llm       = CBorgLLM(client=client)
    qa_chain  = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    return qa_chain

# ─── 8. Interactive CLI loop ─────────────────────────────────────────────────
if __name__ == "__main__":
    qa = build_qa_chain()
    print("🎉 RAG bot ready! Type ‘exit’ to quit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        result = qa.invoke({"query": q})
        print(f"\nBot: {result['result']}")
