from langchain_core.prompts import ChatPromptTemplate

template = """You are a helpful insurance policy assistant. 
Your job is to answer questions about insurance policies accurately.

Use ONLY the following context to answer the question.
If the answer is not found in the context, say "I don't have enough information in the provided policy document to answer this question."
Do NOT make up information or use outside knowledge.
Always mention the page number where you found the answer.

Context:
{context}

Question:
{question}

Answer:"""

def get_qa_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(template)