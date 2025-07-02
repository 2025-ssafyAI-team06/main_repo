# main.py
"""
월드컵 데이터 분석 시스템 메인 실행 파이프라인
"""

from utils import (
    set_api_key,
    print_pipeline_step
)
from services import (
    classify_query_category
)

import os
import sys

# 현재 파일(main.py)의 위치를 기준으로 src 경로 추가
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from serpapi import GoogleSearch

from worldcup_bot.stadium_attractions.generate import run_spot_pipeline
from worldcup_bot.stadium_attractions.embedding import load_embeddings, load_vectorstore

from worldcup_bot.country_statistics.generate import run_worldcup_analysis_pipeline

from src.worldcup_bot.rules_and_regulations.search import load_retriever, build_rag_chain
from worldcup_bot.rules_and_regulations.generate import run_rules_pipeline

load_dotenv()
passage_embedder, query_embedder = load_embeddings()
spot_store = load_vectorstore(passage_embedder)


def create_next_query(category,user_query):
    print("category : ",category)
    if category == "1":
      print("유형1")
      return run_spot_pipeline(user_query, query_embedder, spot_store)
    elif category == "2":
      print("유형2")
      return run_rules_pipeline(user_query, load_retriever(), build_rag_chain())
    elif category == "3":
      return run_worldcup_analysis_pipeline(user_query,"./matches_1930_2022.csv")
    elif category == "4":
      print("유형4")
    elif category == "5":
      print("유형5")
    else:
        return "죄송합니다. 질문을 이해하지 못했습니다."

def main():
    """메인 실행 함수"""
    # API 키 설정
    set_api_key()
    
    # 예시 실행
    user_query = input("질문을 입력하세요: ")
    # csv_path = "./matches_1930_2022.csv"

    category = classify_query_category(user_query)
    print_pipeline_step("📂 분류된 카테고리:", category)

    result = create_next_query(category,user_query)
    
    return result

if __name__ == "__main__":
    main()