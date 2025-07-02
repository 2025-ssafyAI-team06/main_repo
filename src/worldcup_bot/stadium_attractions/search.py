import json
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from serpapi import GoogleSearch
import os

from dotenv import load_dotenv
load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

def intent_classification(user_question):
    system_prompt = """
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
    user_prompt = """
    사용자 질문: {user_question}

    <<<Output Format>>>
    `{{intent: <1 or 2>,
     keywords: <comma-separated keywords>,
     field_name: <field_name or None>}}`
    """

    llm = ChatUpstage(model="solar-mini", temperature=0)
    intent_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt)
    ])
    chain = intent_prompt | llm | StrOutputParser()
    intent_response = chain.invoke({"user_question": user_question})
    print(intent_response)
    return json.loads(intent_response)

def search_answer(response, query_embedder, spot_store):
    field_name = response["field_name"]
    keywords_vector = query_embedder.embed_query(response["keywords"])
    results = spot_store.similarity_search_by_vector(
        keywords_vector,
        k=5,
        filter={"near_field": field_name}
    )
    return results

def use_web_search(user_question):
    params = {
        "engine": "google",
        "q": user_question,
        "api_key": SERPAPI_API_KEY
    }
    search = GoogleSearch(params)
    search_result = search.get_dict()
    print(search_result)

    if search_result.get("knowledge_graph"):
        return {
            "description": search_result["knowledge_graph"].get("description"),
            "address": search_result["knowledge_graph"].get("address")
        }
    else:
        return "결과가 없습니다."
