import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings

load_dotenv()

# API_KEY 호출
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
DB_NAME = os.getenv("DB_NAME")
DB_PATH = os.getenv("DB_PATH")

# 임베딩 모델 및 벡터스토어 초기화
def load_embeddings():
    passage_embedder = UpstageEmbeddings(model="embedding-passage")
    query_embedder = UpstageEmbeddings(model="embedding-query")
    return passage_embedder, query_embedder

def load_vectorstore(passage_embedder):
    spot_store = Chroma(
        collection_name=DB_NAME,
        embedding_function=passage_embedder,
        persist_directory=DB_PATH
    )
    return spot_store

def run_attractions_embedding():
    passage_embedder, query_embedder = load_embeddings()
    vectorstore = load_vectorstore(passage_embedder)
    return passage_embedder, query_embedder, vectorstore