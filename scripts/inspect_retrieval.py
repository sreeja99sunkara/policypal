import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import os
os.chdir(ROOT)

from app.services.vectorstore import get_vectorstore

question = " ".join(sys.argv[1:]) or "What are the sum insured options available?"
docs = get_vectorstore().as_retriever(search_kwargs={"k": 3}).invoke(question)

for i, d in enumerate(docs, 1):
    print(f"\n===== chunk {i} | page {d.metadata.get('page')} | {len(d.page_content)} chars =====")
    print(d.page_content)
    hits = [ln for ln in d.page_content.splitlines() if "lakh" in ln.lower() or "crore" in ln.lower()]
    print(f"--- lines mentioning lakh/crore: {hits}")