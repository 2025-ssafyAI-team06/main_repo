# services.py
"""
월드컵 데이터 분석 시스템의 핵심 비즈니스 로직 함수들
"""

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from prompts import (
    CATEGORY_CLASSIFICATION_SYSTEM_PROMPT,
    CATEGORY_CLASSIFICATION_EXAMPLES,
    SQL_GENERATION_SYSTEM_PROMPT_TEMPLATE,
    SQL_GENERATION_EXAMPLES,
    FINAL_ANSWER_SYSTEM_PROMPT,
    FINAL_ANSWER_PROMPT_TEMPLATE
)
from utils import (
    get_schema_description,
    create_llm_chain,
    format_dataframe_result
)


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
    print("categgory: ",category)
    return category.strip()


def generate_sql_from_query(user_query: str, df: pd.DataFrame) -> str:
    """
    사용자 질문과 DataFrame 스키마를 기반으로 SQL 쿼리 생성
    
    Args:
        user_query (str): 사용자 질문
        df (pd.DataFrame): 데이터프레임
        
    Returns:
        str: 생성된 SQL 쿼리
    """
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
    """
    SQL 실행 결과를 바탕으로 자연스러운 응답 생성
    
    Args:
        user_query (str): 원본 사용자 질문
        sql_result (str): SQL 실행 결과 (문자열 형태)
        
    Returns:
        str: 자연스러운 응답
    """
    prompt_content = FINAL_ANSWER_PROMPT_TEMPLATE.format(
        user_query=user_query,
        sql_result=sql_result
    )
    
    chain = create_llm_chain(
        system_prompt=FINAL_ANSWER_SYSTEM_PROMPT
    )
    
    # ChatPromptTemplate을 직접 생성하여 사용
    prompt = ChatPromptTemplate.from_messages([
        ("system", FINAL_ANSWER_SYSTEM_PROMPT),
        ("human", prompt_content)
    ])
    
    from langchain_upstage import ChatUpstage
    from langchain_core.output_parsers import StrOutputParser
    
    llm = ChatUpstage(model="solar-pro")
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({})


def is_supported_category(category: str) -> bool:
    """
    지원되는 카테고리인지 확인
    
    Args:
        category (str): 카테고리 번호
        
    Returns:
        bool: 지원 여부
    """
    return category == "3"