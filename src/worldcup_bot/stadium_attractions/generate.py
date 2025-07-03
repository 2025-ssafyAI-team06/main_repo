# generate.py

from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# 응답 생성하기
def generate_recommendation_response(filtered_document, user_question, search_result):
    # llm
    llm = ChatUpstage(model="solar-mini",
                      temperature=0.56)
    
    if filtered_document:
        context = {
            "name": filtered_document.metadata.get("name"),
            "website": filtered_document.metadata.get("website"),
            "address": filtered_document.metadata.get("address"),
            "summary": filtered_document.metadata.get("summary")
        }

        website = context["website"]

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
        2. website에 대한 내용이 없다면 해당 문장은 보여주지 않습니다.

        <<<Output Format>>>
        ```
        ### name

        - **웹사이트**: {website}
        - **주소**: address
        - **요약**:
            summary
        ```
        """
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
        raw_answer = chain.invoke({"context": context, "website": website, "user_question": user_question})
        return raw_answer

    else:
        # LLM에게 주어진 문맥을 바탕으로 응답을 생성하도록 요청
        system_prompt = \
        """
        관광지 정보를 바탕으로 각 관광지에 대한 추천 내용을 만들어주세요.
        """

        user_prompt = \
        """
        아래 조건을 참조해서 형식에 따라 답변을 생성하세요

        관광지 추천에 대한 내용이 제공됩니다.
        {search_result}

        사용자 질문: {user_question}

        [CONDITION]
        1. search_result 내용을 사용자가 보기 편하게 정리해서 summary로 보여줍니다.

        <<<Output Format>>>
        ```
        ### name

        - summary
        ```
        """
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
        raw_answer = chain.invoke({"search_result": search_result, "user_question": user_question})
        return raw_answer


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
    Tavily 검색 결과: {search_result}

    사용자 질문: {user_question}

    [CONDITION]
    1. 답변은 해당 관광지에 대해 설명해주겠다는 분위기로 시작합니다.
    2. Tavily 검색 결과를 한국어로 번역해서 summary로 보여주세요.
    3. name은 관광지명을 사용합니다.

    <<<Output Format>>>
    ```
    ### name

    - summary
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

import random
import random
from worldcup_bot.stadium_attractions.search import *
from worldcup_bot.stadium_attractions.generate import *


def run_spot_pipeline(user_question):
    # user_question = input("북중미 월드컵 경기장 근처 관광지에 대해서 질문해주세요.\n")
    response = intent_classification(user_question)

    if response["intent"] == 1:
        # keyword로 웹 서치하기
        print("웹 서치가 필요합니다.")
        search_result = search_spots_in_web(user_question)
        if search_result:
            raw_response = generate_location_response(search_result, response, user_question)
            ai_answer = raw_response.strip("```")
            print(ai_answer)
            return ai_answer
        else:
            print("죄송합니다. 다른 관광지에 대해서 물어봐주세요.")
            return "죄송합니다. 다른 관광지에 대해서 물어봐주세요."


    elif response["intent"] == 2:
        print("DB를 검색합니다.")
        documents = search_answer(response)
        if documents:
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

            # 정확한 문서들이 걸러졌다면
            if filtered_docs:
                filtered_docs_list = [filtered_docs[name] for name in filtered_docs]

                # 거른 것 중에 랜덤으로 2개 뽑기
                selected_docs = random.sample(filtered_docs_list, min(2, len(filtered_docs_list)))

                responses = []

                for doc in selected_docs:
                    raw_response = generate_recommendation_response(doc, user_question,"없음")
                    response = raw_response.strip("```")
                    responses.append(response)

                # 답변 생성하기
                ai_answer = "\n\n".join(responses)
                print(ai_answer)
                return ai_answer
            
            # 정확한 답변이 걸러지지 않았다면
            else:
                search_result = search_spots_in_web(user_question)
                if search_result:
                    raw_response = generate_recommendation_response(filtered_docs, user_question, search_result["answer"])
                    # 답변 생성하기
                    response = raw_response.strip("```")
                    print(response)
                    return response
                else:
                    print("해당 경기장 근처 관광지에 대한 내용이 업데이트 되지 않았습니다.")
                    return "해당 경기장 근처 관광지에 대한 내용이 업데이트 되지 않았습니다."
    return "죄송합니다. 해당 정보에 대해서 찾을 수 없습니다."

