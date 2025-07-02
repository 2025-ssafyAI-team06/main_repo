"""
LLM 기반 응답 생성, SQL 생성, 자연어 응답 생성 등 처리
"""

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage
from langchain_core.output_parsers import StrOutputParser

from prompts import (
    SQL_GENERATION_SYSTEM_PROMPT_TEMPLATE,
    SQL_GENERATION_EXAMPLES,
    FINAL_ANSWER_SYSTEM_PROMPT,
    FINAL_ANSWER_PROMPT_TEMPLATE
)
from utils import (
    get_schema_description,
    create_llm_chain,
    format_dataframe_result,
    load_csv_data,
    print_pipeline_step,
    run_sql_query
)

def is_supported_category(category: str) -> bool:
    """
    지원되는 카테고리인지 확인
    
    Args:
        category (str): 카테고리 번호
        
    Returns:
        bool: 지원 여부
    """
    return category == "3"

def generate_sql_from_query(user_query: str, df: pd.DataFrame) -> str:
    """사용자 질문과 DataFrame 스키마를 기반으로 SQL 쿼리 생성"""
    schema_description = get_schema_description(df)
    system_prompt = SQL_GENERATION_SYSTEM_PROMPT_TEMPLATE.format(
        schema_description=schema_description
    )

    chain = create_llm_chain(
        system_prompt=system_prompt,
        examples=SQL_GENERATION_EXAMPLES,
        user_query=user_query
    )

    sql_result = chain.invoke({})
    return sql_result.strip()


def generate_natural_answer(user_query: str, sql_result: str) -> str:
    """SQL 실행 결과를 바탕으로 자연스러운 응답 생성"""
    prompt_content = FINAL_ANSWER_PROMPT_TEMPLATE.format(
        user_query=user_query,
        sql_result=sql_result
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", FINAL_ANSWER_SYSTEM_PROMPT),
        ("human", prompt_content)
    ])

    llm = ChatUpstage(model="solar-pro")
    chain = prompt | llm | StrOutputParser()

    return chain.invoke({})

def run_worldcup_analysis_pipeline(user_query: str, csv_path: str) -> str:
    """
    월드컵 데이터 분석 파이프라인 실행
    
    Args:
        user_query (str): 사용자 질문
        csv_path (str): CSV 파일 경로
        
    Returns:
        str: 최종 응답 또는 None
    """
    # 1. CSV 데이터 로드
    df = load_csv_data(csv_path)
    
    # 2. SQL 쿼리 생성
    sql_query = generate_sql_from_query(user_query, df)
    print_pipeline_step("📄 생성된 SQL:", f"\n{sql_query}")
    
    # 3. SQL 실행
    df_result = run_sql_query(sql_query, df)
    print_pipeline_step("📊 SQL 실행 결과:", f"\n{df_result}")
    
    # 4. 최종 자연어 응답 생성
    sql_result_str = format_dataframe_result(df_result)
    final_answer = generate_natural_answer(user_query, sql_result_str)
    print_pipeline_step("🗣️ 최종 응답:", f"\n{final_answer}")
    
    return final_answer