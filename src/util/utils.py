# utils.py
"""
월드컵 데이터 분석 시스템의 유틸리티 함수들
"""

import os
import getpass
import warnings
import pandas as pd
import pandasql as ps
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from prompts import CATEGORY_MESSAGES

warnings.filterwarnings("ignore")


def set_api_key():
    """Upstage API 키를 환경변수로 설정"""
    load_dotenv()  # .env 파일 로드
    os.environ["UPSTAGE_API_KEY"] =  os.getenv("UPSTAGE_API_KEY")
    print(CATEGORY_MESSAGES["api_key_success"])


def load_csv_data(csv_path: str) -> pd.DataFrame:
    """CSV 파일을 로드하여 DataFrame으로 반환"""
    return pd.read_csv(csv_path)


def get_schema_description(df: pd.DataFrame) -> str:
    """DataFrame의 스키마를 문자열로 반환"""
    return "\n".join([f"- {col}: {df[col].dtype}" for col in df.columns])


def run_sql_query(sql: str, df: pd.DataFrame) -> pd.DataFrame:
    """SQL 쿼리를 실행하여 결과를 DataFrame으로 반환"""
    return ps.sqldf(sql, {'df': df})


def format_dataframe_result(df_result: pd.DataFrame) -> str:
    """DataFrame 결과를 문자열로 포맷팅"""
    return df_result.to_string(index=False)


def create_llm_chain(system_prompt: str, examples: list = None, user_query: str = None):
    """LLM 체인을 생성하는 헬퍼 함수"""
    messages = [("system", system_prompt)]
    
    # Few-shot 예시가 있는 경우 추가
    if examples:
        for human_msg, ai_msg in examples:
            messages.extend([("human", human_msg), ("ai", ai_msg)])
    
    # 사용자 쿼리가 있는 경우 추가
    if user_query:
        messages.append(("human", user_query))
    print(messages)
    prompt = ChatPromptTemplate.from_messages(messages)
    llm = ChatUpstage(model="solar-pro")
    
    return prompt | llm | StrOutputParser()


def print_pipeline_step(step_name: str, content: str):
    """파이프라인 실행 과정을 출력하는 헬퍼 함수"""
    print(f"\n{step_name} {content}")