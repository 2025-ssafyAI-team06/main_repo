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
import asyncio

# 현재 파일(main.py)의 위치를 기준으로 src 경로 추가
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

import threading

from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from worldcup_bot.stadium_attractions.embedding import *
from worldcup_bot.stadium_attractions.generate import run_spot_pipeline 

from worldcup_bot.rules_and_regulations.embedding import *
from worldcup_bot.rules_and_regulations.search import load_retriever, build_rag_chain
from worldcup_bot.rules_and_regulations.generate import run_rules_pipeline

from worldcup_bot.country_statistics.generate import run_worldcup_analysis_pipeline

from worldcup_bot.jinxes_and_incidents.embedding import *
from worldcup_bot.jinxes_and_incidents.search import load_jinxes_retriever
from worldcup_bot.jinxes_and_incidents.generate import run_jinxes_and_incidents_pipeline, build_jinxes_rag_chain

from worldcup_bot.formations_and_tactics.generate import run_formations_and_tactics_pipeline

import os


load_dotenv()

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 선언
class MessageRequest(BaseModel):
    message: str
    
rules_retriever = load_retriever()
rules_rag_chain = build_rag_chain()

jinxes_retriever = load_jinxes_retriever()
jinxes_rag_chain = build_jinxes_rag_chain()

# @app.on_event("startup")
# async def startup_event():
#     # retriever 초기화는 별도 스레드에서 수행
#     threading.Thread(target=init_vector_store).start()

# def init_vector_store():
#     global rules_retriever, rules_rag_chain, jinxes_retriever, jinxes_rag_chain
#     try:
#         save_to_chroma(chunking_data())
#         run_rules_embedding()
#         load_jinxes_embedding(load_wikipedia_docs(), load_namuwiki_docs())
        
#         rules_retriever = load_retriever()
#         rules_rag_chain = build_rag_chain()

#         jinxes_retriever = load_jinxes_retriever()
#         jinxes_rag_chain = build_jinxes_rag_chain()
#         print("✅ Retriever initialized in background")
#     except Exception as e:
#         print("❌ Retriever initialization failed:", e)


def create_next_query(category,user_query):
    print("category : ",category)
    if category == "1":
      print("유형1")
      return run_spot_pipeline(user_query)
    elif category == "2":
      print("유형2")
      return run_rules_pipeline(user_query, rules_retriever, rules_rag_chain)
    elif category == "3":
      print("유형3")
      return run_worldcup_analysis_pipeline(user_query,"./matches_1930_2022.csv")
    elif category == "4":
      print("유형4")
      return run_jinxes_and_incidents_pipeline(user_query, jinxes_retriever, jinxes_rag_chain)
    elif category == "5":
      print("유형5")
      return run_formations_and_tactics_pipeline(user_query, "./formation_per_nation.csv")  
    else:
        return "죄송합니다. 질문을 이해하지 못했습니다."

@app.post("/chat")
async def chat(req: MessageRequest):
    global retriever
    category = classify_query_category(req.message)
    print_pipeline_step("📂 분류된 카테고리:", category)
    
    return await asyncio.to_thread(create_next_query, category, req.message)

# def main():
#     """메인 실행 함수"""
#     # API 키 설정
#     set_api_key()
    
#     # 예시 실행
#     user_query = input("질문을 입력하세요: ")
#     # csv_path = "./matches_1930_2022.csv"

#     category = classify_query_category(user_query)
#     print_pipeline_step("📂 분류된 카테고리:", category)

#     result = create_next_query(category,user_query)
    
#     return result
  
@app.get("/")
async def root():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)