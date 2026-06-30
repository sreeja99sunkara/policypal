import os
import uuid
import time
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.document_loader import load_and_chunk
from app.services.vectorstore import ingest_documents
from app.core.logger import get_logger, log_with_fields

router = APIRouter()
logger = get_logger(__name__)

MAX_FILE_SIZE = 20 * 1024 * 1024
RAW_DATA_DIR = "data/raw"


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    log_with_fields(logger, "info", "Upload received", filename=file.filename)

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds 20MB limit")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    job_id = str(uuid.uuid4())
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    save_path = os.path.join(RAW_DATA_DIR, f"{job_id}_{file.filename}")

    try:
        with open(save_path, "wb") as f:
            f.write(content)
    except Exception:
        logger.exception(f"Failed to save uploaded file: {file.filename}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    start_time = time.time()
    try:
        chunks = load_and_chunk(save_path)
        log_with_fields(logger, "info", "Chunking complete", filename=file.filename, chunk_count=len(chunks))
    except Exception:
        logger.exception(f"PDF loading/chunking failed: {file.filename}")
        raise HTTPException(status_code=500, detail="Failed to read PDF. The file may be corrupted or password-protected.")

    if len(chunks) == 0:
        raise HTTPException(status_code=400, detail="No extractable text found in PDF")

    try:
        ingest_documents(chunks)
        latency = round(time.time() - start_time, 2)
        log_with_fields(
            logger, "info", "Ingestion complete",
            filename=file.filename,
            job_id=job_id,
            chunk_count=len(chunks),
            latency_seconds=latency
        )
    except Exception:
        logger.exception(f"Vector store ingestion failed: {file.filename}")
        raise HTTPException(status_code=500, detail="Failed to index document. Please try again.")

    return {
        "job_id": job_id,
        "status": "completed",
        "filename": file.filename,
        "chunks_created": len(chunks)
    }