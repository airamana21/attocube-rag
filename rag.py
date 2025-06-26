import os
from dotenv import load_dotenv

# LangChain imports
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Google GenAI SDK
import PyPDF2
from google import genai

# ──────────────────────────────────────────────────────────────────────────────
# 1. Load API key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Set GOOGLE_API_KEY in your .env")

# 2. Gemini LLM wrapper for LangChain
class GeminiLLM(LLM):
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-001"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(self, prompt: str, stop=None) -> str:
        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return resp.text

# 3. Gemini embeddings wrapper
class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str, model_name: str = "text-embedding-004"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.models.embed_content(
            model=self.model_name,
            contents=texts
        )
        return resp.embeddings

    def embed_query(self, text: str) -> list[float]:
        resp = self.client.models.embed_content(
            model=self.model_name,
            contents=text
        )
        return resp.embeddings

# ──────────────────────────────────────────────────────────────────────────────
# 4. Load & split PDFs
PDF_FOLDER = "pdfs"

def load_and_split_pdfs(folder: str) -> list[Document]:
    docs: list[Document] = []
    loader = PyPDFLoader  # uses pypdf under the hood
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(folder, fname)
        # load pages
        raw_docs = loader(path).load()
        # split into chunks
        chunks = splitter.split_documents(raw_docs)
        docs.extend(chunks)
    return docs

# 5. Build or load your Chroma vector store
def get_vectorstore(docs: list[Document]) -> Chroma:
    persist_dir = "chroma_db"
    embedding_fn = GeminiEmbeddings(api_key=API_KEY)
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        # first-time ingest
        db = Chroma.from_documents(
            documents=docs,
            embedding=embedding_fn,
            persist_directory=persist_dir
        )
        db.persist()
    else:
        # reload existing
        db = Chroma(persist_directory=persist_dir, embedding_function=embedding_fn)
    return db

# 6. Build the RetrievalQA chain
def build_qa_chain() -> RetrievalQA:
    docs = load_and_split_pdfs(PDF_FOLDER)
    db = get_vectorstore(docs)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = GeminiLLM(api_key=API_KEY)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",      # simple “stuff” chain
        retriever=retriever
    )
    return qa_chain

# ──────────────────────────────────────────────────────────────────────────────
# 7. Run as a script
if __name__ == "__main__":
    qa = build_qa_chain()
    while True:
        q = input("\nEnter your question (or 'exit'): ").strip()
        if q.lower() in ("exit", "quit"):
            break
        ans = qa.run(q)
        print("\n📝 Answer:\n", ans)
