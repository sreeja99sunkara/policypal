from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from typing import List
import os

from app.services.vectorstore import get_vectorstore
from app.prompts.qa_prompt import get_qa_prompt

_chain = None


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[Page {doc.metadata['page']}]\n{doc.page_content}"
        for doc in docs
    )


def get_rag_chain():
    global _chain
    if _chain is not None:
        return _chain

    # Initialize components
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    prompt = get_qa_prompt()
    parser = StrOutputParser()

    # Base chain
    base_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | parser
    )

    # Chain with sources — API will use this to return citations
    _chain = RunnableParallel(
        answer=base_chain,
        sources=(retriever | format_docs)
    )

    return _chain