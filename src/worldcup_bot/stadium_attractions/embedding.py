import os
import pandas as pd
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_upstage import UpstageEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

CSV_PATH = os.getenv("CSV_PATH")
DB_NAME = os.getenv("DB_NAME")
DB_PATH = os.getenv("DB_PATH")


# Upstage 임베딩 설정
embedding = UpstageEmbeddings(model="embedding-passage")

## 생성된 docs 리스트 사용해서 chromaDB에 저장하기
def chunking_data():
    # csv 불러오기
    embedding_df = pd.read_csv(CSV_PATH)

    # dict로 변환
    embedding_dict = embedding_df.to_dict(orient="records")

    # Document 객체 리스트 생성하기
    documents = []

    for item in embedding_dict:
        # 'summary' 필드의 텍스트 가져오기
        summary = item.get("summary")

        # summary가 문자열인지 확인하기, 그렇지 않다면 변환하기
        if not isinstance(summary, str):
            summary = str(summary)
        
            # 메타데이터 생성
        metadata = {
            "name": item.get("name"),
            "near_field": item.get("near_field"),
            "website": item.get("website"),
            "tripadvisor_url": item.get("tripadvisor_url"),
            "address": item.get("address"),
            "about_rank": item.get("about_rank"),
            "summary": item.get("summary")
        }

        # Document 객체 생성
        document = Document(page_content=summary, metadata=metadata)
        documents.append(document)
    
    # 텍스트 스플리터 설정
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

    # 문서를 나눠서 처리한다.
    split_documents = text_splitter.split_documents(documents)
    
    return split_documents
    

# ChromaDB 생성 함수
def save_to_chroma(doc_chunks, collection_name = DB_NAME, persist_directory = DB_PATH):
    chunked_documents = []
    for chunk in doc_chunks:
        chunked_documents.append(chunk)
        if len(chunked_documents) >= 100:
            try:
                Chroma.from_documents(
                    documents=chunked_documents,
                    embedding=embedding,
                    collection_name=collection_name,
                    persist_directory=persist_directory,
                )
            except Exception as e:
                print(f"Error saving documents to Chroma: {e}")
            finally:
                chunked_documents = []  # 저장 후 리스트 초기화

    # 마지막 남은 문서들 저장
    if chunked_documents:
        try:
            Chroma.from_documents(
                documents=chunked_documents,
                embedding=embedding,
                collection_name=collection_name,
                persist_directory=persist_directory,
            )
        except Exception as e:
            print(f"Error saving remaining documents to Chroma: {e}")
    print("attraction data saved!!!")


# split_documents = chunking_data()

# # 생성
# save_to_chroma(chunking_data())