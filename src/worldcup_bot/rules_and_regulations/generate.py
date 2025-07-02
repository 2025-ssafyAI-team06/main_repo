### generate.py (rules_and_regulations)

import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage
from langchain_core.output_parsers import StrOutputParser
from .search import retrieve

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def generate(state: dict, rag_chain) -> dict:
    logging.info("답변 생성 중...")
    question = state["question"]
    docs = state["documents"]
    context = "\n\n".join([doc.page_content for doc in docs])
    answer = rag_chain.invoke({"context": context, "question": question})
    return {"question": question, "documents": docs, "generation": answer}

def run_rules_pipeline(user_question: str, retriever, rag_chain):
    state = {"question": user_question, "documents": [], "generation": ""}
    state = retrieve(state, retriever)
    state = generate(state, rag_chain)
    print(state["generation"])
    return state["generation"]
