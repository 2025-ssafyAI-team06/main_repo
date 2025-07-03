# search.py
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_upstage import UpstageEmbeddings
from langchain.schema import Document
from typing import List
import os


def retrieve(state: dict, retriever) -> dict:
    print("---RETRIEVE---")
    question = state["question"]
    docs = retriever.get_relevant_documents(question)
    return {"documents": docs, "question": question}

def load_jinxes_retriever(persist_dir: str = "./chroma_jinxes"):
    """
    이미 구축된 벡터스토어를 불러와 retriever 생성
    """
    embedding_model = UpstageEmbeddings(model="solar-embedding-1-large")
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    print(f"🔍 벡터스토어 로드 완료: {persist_dir}")
    return vectorstore.as_retriever(search_kwargs={"k": 4})