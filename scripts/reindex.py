"""Re-index a policy PDF through the same path /api/upload uses (local only).

Usage:
    python scripts/reindex.py [pdf_path]
Default pdf: data/raw/StarHealthAssureInsurancePolicy-Policy-wording.pdf
"""
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from app.services.document_loader import load_and_chunk  # noqa: E402
from app.services.vectorstore import get_vectorstore, ingest_documents  # noqa: E402

DEFAULT_PDF = "data/raw/StarHealthAssureInsurancePolicy-Policy-wording.pdf"


def main() -> None:
    pdf = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF
    vs = get_vectorstore()
    before = vs._collection.count()
    before_types = Counter((m or {}).get("chunk_type", "legacy")
                           for m in vs.get(include=["metadatas"])["metadatas"])

    chunks = load_and_chunk(pdf)
    ingest_documents(chunks)

    after = vs._collection.count()
    after_types = Counter((m or {}).get("chunk_type", "legacy")
                          for m in vs.get(include=["metadatas"])["metadatas"])
    print(f"pdf: {pdf}")
    print(f"chunks before: {before}  {dict(before_types)}")
    print(f"chunks after:  {after}  {dict(after_types)}")
    print(f"table chunks:  {after_types.get('table', 0)}")


if __name__ == "__main__":
    main()
