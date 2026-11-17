"""Structured sources + citation rewriting. The LLM is always mocked (no Groq calls).

Run:  python -m pytest -q tests/test_citations.py
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402

from app.prompts.qa_prompt import get_qa_prompt  # noqa: E402
from app.services.citations import build_sources, rewrite_citations  # noqa: E402
from app.services.rag_chain import build_chain  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def legacy_format_docs(docs):
    """Frozen copy of rag_chain.format_docs as it was before this change.
    The prompt context must stay byte-identical to this."""
    return "\n\n".join(f"[Page {doc.metadata['page']}]\n{doc.page_content}" for doc in docs)


class RecordingLLM:
    """Stands in for ChatGroq: records the prompt it receives, returns a canned answer."""

    def __init__(self, answer: str):
        self.answer = answer
        self.calls = []
        self.runnable = RunnableLambda(self._invoke)

    def _invoke(self, prompt_value):
        self.calls.append(prompt_value.to_messages())
        return AIMessage(content=self.answer)


SAMPLE_DOCS = [
    Document(page_content="Pre-existing diseases are covered after 36 months.\nSecond line.",
             metadata={"page": 4, "pages": "4", "chunk_type": "text"}),
    Document(page_content="[SCHEDULE OF BENEFITS] S.No: 29 | Benefit: Co-payment | Sum Insured (INR) -> "
                          "5 lacs / 10 lacs: subject to co-payment of 10% of each and every claim amount",
             metadata={"page": 18, "pages": "18,19", "chunk_type": "table", "table_part": "rows"}),
    Document(page_content="Table SCHEDULE OF BENEFITS gives Benefit for each Sum Insured (INR): 5 lacs, 10 lacs.",
             metadata={"page": 18, "pages": "18,19", "chunk_type": "table", "table_part": "header",
                       "table_title": "SCHEDULE OF BENEFITS"}),
    Document(page_content="x" * 500, metadata={"page": 0}),  # legacy chunk: no pages / chunk_type
]


@pytest.fixture
def client(monkeypatch):
    from app import main
    from app.routers import query as query_router

    llm = RecordingLLM("Pre-existing diseases: 36 months (Page 4). Co-payment 10% (Pages 18-19).")
    chain = build_chain(RunnableLambda(lambda q: SAMPLE_DOCS), llm.runnable)
    main.app.dependency_overrides[query_router.get_chain] = lambda: chain
    monkeypatch.setattr(query_router, "has_documents", lambda: True)
    yield TestClient(main.app), llm  # no `with`: lifespan (model warm-up) not run
    main.app.dependency_overrides.clear()


# --------------------------------------------------------------------------- #
# (a) + (b) /query response
# --------------------------------------------------------------------------- #
def test_query_returns_structured_sources_with_viewer_pages(client):
    tc, _ = client
    r = tc.post("/api/query", json={"question": "What is the waiting period?"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"answer", "sources", "cited_pages"}
    src = body["sources"]
    assert [s["page"] for s in src] == [5, 19, 19, 1]
    assert src[0]["pages"] == [5] and src[0]["chunk_type"] == "text" and src[0]["title"] is None
    assert src[1]["pages"] == [19, 20] and src[1]["chunk_type"] == "table"
    assert src[1]["title"] == "SCHEDULE OF BENEFITS"      # parsed from row text
    assert src[2]["title"] == "SCHEDULE OF BENEFITS"      # from table_title metadata
    assert src[3]["pages"] == [1] and src[3]["chunk_type"] == "text"
    assert src[0]["text"] == SAMPLE_DOCS[0].page_content
    assert len(src[3]["snippet"]) <= 302 and src[3]["text"] == "x" * 500


def test_answer_citations_rewritten_to_viewer_pages(client):
    tc, _ = client
    body = tc.post("/api/query", json={"question": "q"}).json()
    assert body["answer"] == "Pre-existing diseases: 36 months (Page 5). Co-payment 10% (Pages 19-20)."
    assert body["cited_pages"] == [5, 19, 20]


@pytest.mark.parametrize("answer,expected,pages", [
    ("Covered (Page 4).", "Covered (Page 5).", [5]),
    ("No citation in this answer.", "No citation in this answer.", []),
])
def test_single_and_no_citation(answer, expected, pages):
    assert rewrite_citations(answer) == (expected, pages)


# --------------------------------------------------------------------------- #
# (c) multi-page forms, and numbers that are not pages
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("answer,expected,pages", [
    ("(Page 4, 5)", "(Page 5, 6)", [5, 6]),
    ("(Pages 4-5)", "(Pages 5-6)", [5, 6]),
    ("(Pages 4 – 5)", "(Pages 5 – 6)", [5, 6]),
    ("(Pages 4 to 6)", "(Pages 5 to 7)", [5, 6, 7]),
    ("(Pages 4 and 18)", "(Pages 5 and 19)", [5, 19]),
    ("(page 18,19)", "(page 19,20)", [19, 20]),
    ("(Page 4) and later (Page 7)", "(Page 5) and later (Page 8)", [5, 8]),
    ("(Pages 5, 6, 9) and (Page 6)", "(Pages 6, 7, 10) and (Page 7)", [6, 7, 10]),
    ("(PAGE 0)", "(PAGE 1)", [1]),
    # amounts / units right after a citation are not pages
    ("(Page 5, 1,500 for individual)", "(Page 6, 1,500 for individual)", [6]),
    ("(Page 4 and 5 lakh)", "(Page 5 and 5 lakh)", [5]),
    ("(Page 4, 10,00,000)", "(Page 5, 10,00,000)", [5]),
])
def test_multi_page_forms(answer, expected, pages):
    assert rewrite_citations(answer) == (expected, pages)


# --------------------------------------------------------------------------- #
# (d) prompt context unchanged
# --------------------------------------------------------------------------- #
def _assert_prompt_unchanged(llm: RecordingLLM, docs, question: str):
    expected = get_qa_prompt().format_messages(context=legacy_format_docs(docs), question=question)
    assert len(llm.calls) == 1
    got = llm.calls[0]
    assert [m.type for m in got] == [m.type for m in expected]
    for g, e in zip(got, expected):
        assert g.content.encode("utf-8") == e.content.encode("utf-8")


def test_prompt_context_byte_identical_sample_docs():
    llm = RecordingLLM("ok (Page 4)")
    chain = build_chain(RunnableLambda(lambda q: SAMPLE_DOCS), llm.runnable)
    out = chain.invoke("What is the co-payment?")
    _assert_prompt_unchanged(llm, SAMPLE_DOCS, "What is the co-payment?")
    assert out["answer"] == "ok (Page 4)"            # chain itself does not rewrite
    assert out["sources"] == build_sources(SAMPLE_DOCS)


def test_prompt_context_byte_identical_real_retrieval():
    """One real retrieval against the local index (LLM still mocked)."""
    from app.services.vectorstore import has_documents
    if not has_documents():
        pytest.skip("no local index")
    try:
        from app.services.rag_chain import retrieve_and_rerank
        q = "What is the co-payment?"
        docs = retrieve_and_rerank(q)
    except Exception as e:  # models not in the local HF cache, etc.
        pytest.skip(f"real retrieval unavailable: {e}")
    llm = RecordingLLM("stub")
    build_chain(RunnableLambda(lambda _: docs), llm.runnable).invoke(q)
    _assert_prompt_unchanged(llm, docs, q)
    assert "[Page " in llm.calls[0][-1].content       # 0-indexed labels still in the prompt
