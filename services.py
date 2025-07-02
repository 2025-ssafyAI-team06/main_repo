# services.py
"""
월드컵 데이터 분석 시스템의 핵심 비즈니스 로직 함수들
"""

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from prompts import (
    CATEGORY_CLASSIFICATION_SYSTEM_PROMPT,
    CATEGORY_CLASSIFICATION_EXAMPLES
)
from utils import (
    create_llm_chain
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
    print("category: ", category)
    return category.strip()