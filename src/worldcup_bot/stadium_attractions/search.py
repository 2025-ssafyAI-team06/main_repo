# search.py

import os
import json
import urllib.parse
import urllib.request

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tavily import TavilyClient

load_dotenv()

# API_KEY 호출
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

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

# 파파고 번역
TEXT_TRANSLATION_URL = os.getenv("TEXT_TRANSLATION_URL")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")


# 만약 ~에 대해서 알려줘. 라고 한다면? -> 분기 처리 필요!
def intent_classification(user_question):
    system_prompt = \
    """
    당신은 사용자의 질문을 아래 두 가지의 의도 중 하나로 분류하고, 키워드를 추출해야 합니다.

    의도:
    1. 특정 장소의 위치(주소, 위치 설명 등)를 묻는 질문
    2. 특정 축구 경기장 근처의 맛집/관광지/숙소 등 주변 장소를 추천해달라는 질문

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
    1) intent가 1이라면, keywords로 특정 장소만 영어로 바꿔서 추출하고, field_name은 삭제해주세요.
    2) intent가 2이고, 사용자 질문에 경기장 이름이 있다면, field_name으로 추출해주세요. field_name이 한국어라면 경기장 이름 모음을 참고해서 field_name만 영어로 변경해주세요.
    3) intent가 2이고, 사용자 질문에 경기장 이름이 없다면, field_name은 None입니다.
    4) intent가 2라면, keywords는 최대 3개만 존재합니다.
    5) intent가 1, 2라면, JSON 형식으로 반환해주세요.

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

    if intent_response:
        return json.loads(intent_response)
    
    else:
        return intent_response


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

    if results:
        return results
    else:
        return None


# 사용자 질문으로 웹 서치 하기
def search_spots_in_web(user_question):
    client = TavilyClient(TAVILY_API_KEY)

    # 사용자 질문 영어로 변환
    encText = urllib.parse.quote(user_question)
    data = "source=ko&target=en&text=" + encText
    request = urllib.request.Request(TEXT_TRANSLATION_URL)
    request.add_header("X-NCP-APIGW-API-KEY-ID", CLIENT_ID)
    request.add_header("X-NCP-APIGW-API-KEY", CLIENT_SECRET)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()

    if rescode == 200:
        response_body = response.read()
        all_response = response_body.decode("utf-8")
        json_response = json.loads(all_response)
        translated_question = json_response["message"]["result"]["translatedText"]
        
        search_result = client.search(
                            query=translated_question,
                            include_answer="advanced",
                        )
        return search_result
    
    else:
        return None