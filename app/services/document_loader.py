from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List


def load_and_chunk(pdf_path: str) -> List[Document]:
    # Load PDF
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    # Chunk
    # 500/80 chosen because policy clauses average ~400 chars
    # 80 overlap prevents key terms from being split across boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )
    chunks = splitter.split_documents(documents)

    return chunks