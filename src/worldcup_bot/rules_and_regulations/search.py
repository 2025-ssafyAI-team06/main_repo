### search.py (rules_and_regulations)

import os
import logging
import warnings
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

load_dotenv()
os.environ["UPSTAGE_API_KEY"]

def build_rag_chain() -> Runnable:
    template = (
        "당신은 제공된 정보를 바탕으로 질문에 답하는 한국어 또는 영어 어시스턴트입니다.\n"
        "질문이 어떤 언어로 들어오든, 답변은 반드시 질문과 동일한 언어로 해주세요.\n"
        "한국어 질문이 들어오면 영어로 번역을 한 후, 다시 한국어로 번역해 사용해주세요.\n"
        "답이 문서 내에 직접적으로 없더라도 유사한 내용이 있다면 바탕으로 최대한 답변을 시도하세요.\n"
        # "아래 컨텍스트(문서)에서 답을 찾을 수 없으면, '죄송합니다. 해당 질문에 대한 답을 찾을 수 없습니다.'라고만 답하세요.\n"
        "\n[컨텍스트]\n{context}\n\n"
        "[질문] {question}\n"
        "[답변]:"
    )
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatUpstage(model="solar-pro", temperature=0.5)
    rag_chain = prompt | llm | StrOutputParser()
    return rag_chain

def load_retriever(persist_dir: str = "./chroma_rules"):
    embedding = UpstageEmbeddings(model="solar-embedding-1-large")
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

def retrieve(state: dict, retriever) -> dict:
    logging.info("문서 검색 중...")
    question = state["question"]
    docs = retriever.get_relevant_documents(question)
    logging.info(f"{len(docs)}개의 문서가 검색되었습니다.")
    print(docs)
    # print(docs)
    return {"documents": docs, "question": question}