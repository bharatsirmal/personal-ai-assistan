import os
import hashlib
import pdfplumber
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document



# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
load_dotenv(os.path.join(BASE_DIR, ".env"), override=False)


# ============================================================
# SETTINGS
# ============================================================

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


# ============================================================
# EXTRACT TEXT FROM PDF
# ============================================================

def extract_pdf_text(pdf_path):

    text = ""

    try:

        with pdfplumber.open(pdf_path) as pdf:

            for page_number, page in enumerate(pdf.pages, start=1):

                page_text = page.extract_text()

                if page_text:

                    text += page_text + "\n\n"

    except Exception as e:

        print(f"Error reading {pdf_path}: {e}")

        return ""

    return text.strip()


# ============================================================
# LOAD ALL PDF FILES
# ============================================================

def load_all_pdfs():

    documents = []

    if not os.path.exists(DATA_DIR):

        raise FileNotFoundError(
            f"Data directory not found: {DATA_DIR}"
        )

    pdf_files = [
        file
        for file in os.listdir(DATA_DIR)
        if file.lower().endswith(".pdf")
    ]

    if not pdf_files:

        raise FileNotFoundError(
            f"No PDF files found inside: {DATA_DIR}"
        )

    print()
    print("=" * 60)
    print("BHARAT AI - PDF DATA INGESTION")
    print("=" * 60)

    print(f"Data directory: {DATA_DIR}")
    print(f"Found {len(pdf_files)} PDF files.")
    print()

    for filename in sorted(pdf_files):

        pdf_path = os.path.join(
            DATA_DIR,
            filename
        )

        print(f"Reading: {filename}")

        text = extract_pdf_text(pdf_path)

        if not text:

            print("  WARNING: No text extracted.")
            raise ValueError(f"No readable text in {filename}; keeping the existing index.")

        print(
            f"  Extracted {len(text)} characters."
        )

        document = Document(
            page_content=text,
            metadata={
                "source": filename,
                "file_type": "pdf",
            }
        )

        documents.append(document)

    return documents


# ============================================================
# SPLIT DOCUMENTS
# ============================================================

def split_documents(documents):

    print()
    print("Splitting documents...")

    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = text_splitter.split_documents(
        documents
    )

    print(
        f"Generated {len(chunks)} chunks."
    )

    return chunks


# ============================================================
# CREATE CHROMA DATABASE
# ============================================================

def synchronize_vector_database(vector_store, chunks=None):
    """Add current PDF chunks before removing obsolete ones; never delete the DB."""
    if chunks is None:
        chunks = split_documents(load_all_pdfs())
    if not chunks:
        raise ValueError("No PDF chunks available; keeping the existing index.")
    ids = []
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        content = chunk.metadata["source"] + "\0" + str(index) + "\0" + chunk.page_content
        ids.append(hashlib.sha256(content.encode("utf-8")).hexdigest())
    existing = set(vector_store.get()["ids"])
    missing = [(chunk_id, chunk) for chunk_id, chunk in zip(ids, chunks) if chunk_id not in existing]
    if missing:
        vector_store.add_documents(
            [chunk for _, chunk in missing], ids=[chunk_id for chunk_id, _ in missing]
        )
    obsolete = existing - set(ids)
    if obsolete:
        vector_store.delete(ids=list(obsolete))
    return vector_store


def create_vector_database(chunks):

    print()
    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )

    print("Embedding model loaded.")

    # --------------------------------------------------------
    # Open the existing database and synchronize its documents
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Create new database
    # --------------------------------------------------------

    print()
    print("Creating ChromaDB...")

    vector_store = Chroma(
        collection_name=os.getenv("CHROMA_COLLECTION", "langchain"),
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )
    synchronize_vector_database(vector_store, chunks)

    print()
    print("ChromaDB created successfully.")

    return vector_store


# ============================================================
# MAIN INGESTION
# ============================================================

def ingest_data():

    documents = load_all_pdfs()

    if not documents:

        print()
        print("ERROR: No documents were loaded.")

        return

    print()
    print(
        f"Loaded {len(documents)} PDF documents."
    )

    chunks = split_documents(
        documents
    )

    if not chunks:

        print()
        print("ERROR: No chunks generated.")

        return

    create_vector_database(
        chunks
    )

    print()
    print("=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)

    print()
    print(f"PDF documents : {len(documents)}")
    print(f"Chunks        : {len(chunks)}")
    print(f"Database      : {DB_DIR}")

    print()
    print("Your Bharat AI RAG database is ready.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    ingest_data()
