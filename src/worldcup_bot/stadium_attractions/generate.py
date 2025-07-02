from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

    system_prompt = "관광지 정보를 바탕으로 각 관광지에 대한 추천 내용을 만들어주세요."
    user_prompt = """
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

    llm = ChatUpstage(model="solar-mini", temperature=0.56)
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", user_prompt)])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "website": website, "tripadvisor": tripadvisor, "user_question": user_question})

def generate_location_response(search_result, response, user_question):
    spot_name = response["keywords"]

    system_prompt = "사용자 질문을 읽고 관광지 설명을 바탕으로 관광지에 대한 내용을 만들어주세요."
    user_prompt = """
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

    llm = ChatUpstage(model="solar-mini", temperature=0.56)
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", user_prompt)])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"spot_name": spot_name, "search_result": search_result, "user_question": user_question})

import random
from worldcup_bot.stadium_attractions.search import intent_classification, search_answer, use_web_search
from worldcup_bot.stadium_attractions.generate import generate_recommendation_response, generate_location_response


def run_spot_pipeline(user_question, query_embedder, spot_store):
    # user_question = "루멘 필드 근처 맛집 알려줘"
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
        documents = search_answer(response, query_embedder, spot_store)
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