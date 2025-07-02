# main.py
"""
월드컵 데이터 분석 시스템 메인 실행 파이프라인
"""

from utils import (
    set_api_key,
    load_csv_data,
    run_sql_query,
    format_dataframe_result,
    print_pipeline_step
)
from services import (
    classify_query_category,
    generate_sql_from_query,
    generate_natural_answer,
    is_supported_category
)
from prompts import CATEGORY_MESSAGES

import os
import json
import random

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from serpapi import GoogleSearch

load_dotenv()

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# ChromaDB 불러오기
DB_NAME = os.getenv("DB_NAME")
DB_PATH = os.getenv("DB_PATH")

# 임베딩
passage_embedder = UpstageEmbeddings(model="embedding-passage")
query_embedder = UpstageEmbeddings(model="embedding-query")

spot_store = Chroma(
                collection_name=DB_NAME,
                embedding_function=passage_embedder,
                persist_directory=DB_PATH
            )


# 만약 ~에 대해서 알려줘. 라고 한다면? -> 분기 처리 필요!
def intent_classification(user_question):
    system_prompt = \
    """
    
    당신은 사용자의 질문을 아래 두 가지의 의도 중 하나로 분류하고, 키워드를 추출해야 합니다.

    의도:
    1. 특정 장소의 위치(주소, 위치 설명 등)를 묻는 질문
    2. 특정 장소 근처의 맛집/관광지/숙소 등 주변 장소를 추천해달라는 질문
    
    경기장 이름 모음:
    {{
        루멘 필드: Lumen Field,
        리바이스 스타디움: Levi’s Stadium,
        소피 스타디움: SoFi Stadium,
        에이티앤티 스타디움: AT&T Stadium,
        엔알지 스타디움: NRG Stadium,
        애로우헤드 스타디움: Arrowhead Stadium,
        링컨 파이낸셜 스타디움: Lincoln Financial Field,
        메르세데스 벤츠 스타디움: Mercedes Benz Stadium,
        하드록 스타디움: Hard Rock Stadium,
        멧라이프 스타디움: MetLife Stadium,
        질레트 스타디움: Gillette Stadium,
        아크론 스타디움: Estadio Akron,
        BC 플레이스: BC Place,
        BMO 필드: BMO Field
    }}

    사용자 질문을 읽고,
    1) intent는 반드시 1, 2 중 하나만 답하세요.
    2) intent가 1이라면, keywords로 특정 장소만 영어로 바꿔서 추출하고, field_name은 삭제해주세요.
    3) intent가 2이고, 사용자 질문에 경기장 이름이 있다면, field_name으로 추출해주세요. field_name이 한국어라면 경기장 이름 모음을 참고해서 field_name만 영어로 변경해주세요.
    4) intent가 2이고, 사용자 질문에 경기장 이름이 없다면, field_name은 None입니다.
    4) intent가 2라면, keywords는 최대 3개만 존재합니다.
    5) JSON 형식으로 반환해주세요.

    """
    user_prompt = \
    """
    사용자 질문: {user_question}

    <<<Output Format>>>
    `{{intent: <1 or 2>,
     keywords: <comma-separated keywords>,
     field_name: <field_name or None>}}`
    """

    # llm
    llm = ChatUpstage(model="solar-mini",
                      temperature=0)

    intent_prompt = ChatPromptTemplate.from_messages(
        messages=[
            (
                "system",
                system_prompt
            ),
            (
                "user",
                user_prompt
            )
        ],
    )

    chain = intent_prompt | llm | StrOutputParser()
    intent_response = chain.invoke({"user_question": user_question})
    print(intent_response)
    return json.loads(intent_response)


# documents모음에서 원하는 docs 찾기
def search_answer(response):
    field_name = response["field_name"]

    # 쿼리용 Upstage 임베딩 사용하기 -> 위키독스
    keywords_vector = query_embedder.embed_query(response["keywords"])

    # 벡터 유사도 검색을 수행해서 가장 유사한 5개의 문서 반환하기
    results = spot_store.similarity_search_by_vector(
        keywords_vector,
        k=5,
        filter={"near_field": field_name}
    )
    return results


# 응답 생성하기
def generate_recommendation_response(filtered_document, user_question):
    context = {
        "name": filtered_document.metadata.get("name"),
        "website": filtered_document.metadata.get("website"),
        "tripadvisor": filtered_document.metadata.get("tripadvisor_url"),
        "address": filtered_document.metadata.get("address"),
        "summary": filtered_document.metadata.get("summary")
    }

    website = context["website"]
    tripadvisor = context["tripadvisor"]

    # LLM에게 주어진 문맥을 바탕으로 응답을 생성하도록 요청
    system_prompt = \
    """
    관광지 정보를 바탕으로 각 관광지에 대한 추천 내용을 만들어주세요.
    """

    user_prompt = \
    """
    아래 조건을 참조해서 형식에 따라 답변을 생성하세요
    조건: [CONDITION]

    관광지 정보가 JSON 형식으로 제공됩니다.
    {context}

    사용자 질문: {user_question}

    [CONDITION]
    1. 관광지 정보의 summary를 확인하고 최대 3문장으로 요약해서 summary로 바꿔서 보여주세요.
    2. website에 대한 내용이 없다면 보여주지 않습니다.
    3. tripadvisor에 대한 내용이 없다면 보여주지 않습니다.

    <<<Output Format>>>
    ```
    ### name

    - **웹사이트**: {website}
    - **[트립어드바이저]({tripadvisor})**
    - **주소**: address
    - **요약**:
        summary
    ```
    """
    # llm
    llm = ChatUpstage(model="solar-mini",
                      temperature=0.56)

    answer_generation_prompt = ChatPromptTemplate.from_messages(
        messages=[
            (
                "system",
                system_prompt
            ),
            (
                "user",
                user_prompt
            )
        ],
    )

    chain = answer_generation_prompt | llm | StrOutputParser()
    raw_answer = chain.invoke({"context": context, "website": website, "tripadvisor": tripadvisor, "user_question": user_question})
    return raw_answer

# Web search
def use_web_search(user_question):
    params = {
        "engine": "google",
        "q": user_question,
        "api_key": SERPAPI_API_KEY
    }

    search = GoogleSearch(params)
    search_result = search.get_dict()
    print(search_result)

    if search_result["knowledge_graph"]:
        description = search_result["knowledge_graph"].get("description")
        address = search_result["knowledge_graph"].get("address")

        result = {"description": description,
                  "address": address}
        return result

    else:
        return "결과가 없습니다."


# 특정 관광지에 대한 답변
def generate_location_response(search_result, response, user_question):

    spot_name = response["keywords"]

    # LLM에게 주어진 문맥을 바탕으로 응답을 생성하도록 요청
    system_prompt = \
    """
    사용자 질문을 읽고 관광지 설명을 바탕으로 관광지에 대한 내용을 만들어주세요.
    """

    user_prompt = \
    """
    아래 조건을 참조해서 형식에 따라 답변을 생성하세요
    조건: [CONDITION]

    관광지명: {spot_name}
    구글 검색 결과: {search_result}

    사용자 질문: {user_question}

    [CONDITION]
    1. 답변은 해당 관광지에 대해 설명해주겠다는 분위기로 시작합니다.
    2. 구글 검색 결과 안의 description을 한국어로 번역해서 summary로 보여주세요.
    3. 주소는 구글 검색 결과 안의 address를 사용합니다.
    4. name은 관광지명을 사용합니다.

    <<<Output Format>>>
    ```
    ### name
    - 주소: address

    summary
    ```
    """

    # llm
    llm = ChatUpstage(model="solar-mini",
                      temperature=0.56)

    answer_generation_prompt = ChatPromptTemplate.from_messages(
        messages=[
            (
                "system",
                system_prompt
            ),
            (
                "user",
                user_prompt
            )
        ],
    )

    chain = answer_generation_prompt | llm | StrOutputParser()
    raw_answer = chain.invoke({"spot_name": spot_name, "search_result": search_result, "user_question": user_question})
    return raw_answer



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
    
    # 2. 쿼리 카테고리 분류
    category = classify_query_category(user_query)
    print_pipeline_step("📂 분류된 카테고리:", category)
    
    # 3. 지원되는 카테고리인지 확인
    if not is_supported_category(category):
        print(CATEGORY_MESSAGES["unsupported"])
        return None
    
    # 4. SQL 쿼리 생성
    sql_query = generate_sql_from_query(user_query, df)
    print_pipeline_step("📄 생성된 SQL:", f"\n{sql_query}")
    
    # 5. SQL 실행
    df_result = run_sql_query(sql_query, df)
    print_pipeline_step("📊 SQL 실행 결과:", f"\n{df_result}")
    
    # 6. 최종 자연어 응답 생성
    sql_result_str = format_dataframe_result(df_result)
    final_answer = generate_natural_answer(user_query, sql_result_str)
    print_pipeline_step("🗣️ 최종 응답:", f"\n{final_answer}")
    
    return final_answer

def categoryOne(user_question):
    user_question = "루멘 필드 근처 맛집 알려줘"
    response = intent_classification(user_question)

    if response["intent"] == 1:
        # keyword로 웹서치하기
        print("웹 서치가 필요합니다.")
        search_result = use_web_search(response["keywords"])

        raw_response = generate_location_response(search_result, response, user_question)
        ai_answer = raw_response.strip("```")
        print(ai_answer)
        return ai_answer


    elif response["intent"] == 2:
        print("DB를 검색합니다.")
        documents = search_answer(response)
        # 메타 데이터 이용해서 조금 더 정확한 답변을 받기
        keyword_list = response["keywords"].split(", ")

        # 한 번 더 거르기
        filtered_docs = {}

        for document in documents:
            for keyword in keyword_list:
                if keyword in document.metadata.get("about_rank", ""):
                    name = document.metadata.get("name")
                    if name not in filtered_docs:
                        filtered_docs[name] = document

        # 한번 더 걸러진 것이 없다면
        if not filtered_docs:
            print("키워드로 정확한 문서 필터링 실패")
            for document in documents:
                name = document.metadata.get("name")
                if name not in filtered_docs:
                    filtered_docs[name] = document

        filtered_docs_list = [filtered_docs[name] for name in filtered_docs]

        # 거른 것 중에 랜덤으로 2개 뽑기
        selected_docs = random.sample(filtered_docs_list, min(2, len(filtered_docs_list)))

        responses = []

        for doc in selected_docs:
            raw_response = generate_recommendation_response(doc, user_question)
            response = raw_response.strip("```")
            responses.append(response)

        # 답변 생성하기
        ai_answer = "\n\n".join(responses)
        print(ai_answer)
        return ai_answer

def create_next_query(category,user_query):
    print("category : ",category)
    if category == "1":
      print("유형1")
      categoryOne(user_query)
    elif category == "2":
      print("유형2")
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