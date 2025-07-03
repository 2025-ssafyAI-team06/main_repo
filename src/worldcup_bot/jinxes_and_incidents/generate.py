# generate.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_upstage import ChatUpstage
from langchain_core.runnables import Runnable

from worldcup_bot.jinxes_and_incidents.search import retrieve

import os


def build_jinxes_rag_chain() -> Runnable:
    template = (
        "당신은 제공된 정보를 바탕으로 질문에 답하는 도움이 되는 한국어 어시스턴트입니다.\n"
        "항상 주어진 컨텍스트만 활용하여 질문에 답해줘야 합니다.\n\n"
        "컨텍스트:\n{context}\n\n"
        "질문: {question}\n"
        "답변:"
    )

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatUpstage(model="solar-pro", temperature=0)
    rag_chain = prompt | llm | StrOutputParser()
    return rag_chain


def generate(state: dict, rag_chain) -> dict:
    print("---GENERATE ANSWER---")
    question = state["question"]
    docs = state["documents"]
    answer = rag_chain.invoke({"context": docs, "question": question})
    return {"question": question, "documents": docs, "generation": answer}


def run_jinxes_and_incidents_pipeline(user_query: str, retriever, rag_chain) -> str:
    state = {"question": user_query}
    state = retrieve(state, retriever)
    state = generate(state, rag_chain)

    print("\n🗣️ 최종 응답:\n")
    print(state["generation"])
    return state["generation"]
