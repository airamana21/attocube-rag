import os
from typing import Any, List
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

# ─── 2. LLM wrapper for Google Gemini-Flash via CBorg ────────────────────────
class CBorgLLM(LLM):
    client:     OpenAI
    model_name: str      = "google/gemini-flash"
    callbacks:  Any      = None
    verbose:    bool     = False

    def __init__(self, client: OpenAI, model_name: str = None, callbacks: Any = None, verbose: bool = False):
        # pass through to BaseModel/LLM initializer for pydantic
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

# ─── 3. Embeddings wrapper for text-embedding-004 via CBorg ────────────────
class CBorgEmbeddings(Embeddings):
    client:     OpenAI
    model_name: str = "google/text-embedding-004"

    def __init__(self, client: OpenAI, model_name: str = None):
        # Embeddings base has no init args, so just assign
        self.client = client
        if model_name:
            self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed documents in batches of up to 100 to avoid API limits.
        """
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

# ─── 4. PDF → Documents → Chunks ─────────────────────────────────────────────
PDF_FOLDER = "pdfs"
CHROMA_DIR = "chroma_db"

def load_and_split_pdfs(folder: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = []
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".pdf"):
            continue
        pages = PyPDFLoader(os.path.join(folder, fname)).load()
        docs += splitter.split_documents(pages)
    return docs

# ─── 5. Build or reload your Chroma vector store ─────────────────────────────
def get_vectorstore(docs):
    emb = CBorgEmbeddings(client=client)
    if not os.path.isdir(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        db = Chroma.from_documents(
            documents=docs,
            embedding=emb,
            persist_directory=CHROMA_DIR
        )
        # No need to call db.persist(); from_documents handles persistence.
    else:
        db = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=emb
        )
    return db

# ─── 6. Wire up the RetrievalQA chain ────────────────────────────────────────
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

# ─── 7. Interactive CLI loop ─────────────────────────────────────────────────
if __name__ == "__main__":
    qa = build_qa_chain()
    print("🎉 RAG bot ready! Type ‘exit’ to quit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        result = qa.invoke({"query": q})
        print("\nBot:", result["result"])
