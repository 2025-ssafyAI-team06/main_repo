# main.py
"""
월드컵 데이터 분석 시스템 메인 실행 파이프라인
"""
from src.util.utils import (
    print_pipeline_step
)

import os
import sys
import asyncio
import chromadb
# 현재 파일(main.py)의 위치를 기준으로 src 경로 추가
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)
from chromadb.config import Settings

settings = Settings(anonymized_telemetry=False)
client = chromadb.Client(settings)
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

from prompts import (
    CATEGORY_CLASSIFICATION_SYSTEM_PROMPT,
    CATEGORY_CLASSIFICATION_EXAMPLES
)
from src.util.utils import (
    create_llm_chain
)

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
      return run_worldcup_analysis_pipeline(user_query)
    elif category == "4":
      print("유형4")
      return run_jinxes_and_incidents_pipeline(user_query, jinxes_retriever, jinxes_rag_chain)
    elif category == "5":
      print("유형5")
      return run_formations_and_tactics_pipeline(user_query)
    else:
        return "죄송합니다. 질문을 이해하지 못했습니다."

@app.post("/chat")
async def chat(req: MessageRequest):
    global retriever
    category = classify_query_category(req.message)
    print_pipeline_step("📂 분류된 카테고리:", category)
    
    return await asyncio.to_thread(create_next_query, category, req.message)


@app.get("/")
async def root():
    return {"status": "ok"}

def classify_query_category(user_query: str) -> str:
    """
    사용자 질문을 카테고리로 분류
    
    Args:
        user_query (str): 사용자 질문
        
    Returns:
        str: 분류된 카테고리 번호
    """
    chain = create_llm_chain(
        system_prompt=CATEGORY_CLASSIFICATION_SYSTEM_PROMPT,
        examples=CATEGORY_CLASSIFICATION_EXAMPLES,
        user_query=user_query
    )

    category = chain.invoke({})
    print("category: ", category)
    return category.strip()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)