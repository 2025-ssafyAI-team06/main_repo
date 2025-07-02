### embedding.py (rules_and_regulations)

import os
import logging
import warnings
from typing import List
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

URLS = [
    "https://ko.wikipedia.org/wiki/2026%EB%85%84_FIFA_%EC%9B%94%EB%93%9C%EC%BB%B5",
    "https://www.britannica.com/sports/World-Cup-football",
    "https://www.maniareport.com/view.php?ud=2025060608385343656cf2d78c68_19"
]

DEFAULT_VECTOR_DB_DIR = "./chroma_rules"

load_dotenv()
if "UPSTAGE_API_KEY" not in os.environ or not os.environ["UPSTAGE_API_KEY"]:
    import getpass
    os.environ["UPSTAGE_API_KEY"] = getpass.getpass("Enter your Upstage API key: ")

def load_documents(urls: List[str], pdf_path: str) -> List[Document]:
    try:
        logging.info("웹페이지 문서 로딩 중...")
        url_loader = WebBaseLoader(urls)
        url_docs = url_loader.load()
        logging.info(f"웹 문서 {len(url_docs)}개 로딩 완료")
    except Exception as e:
        logging.error(f"웹 문서 로딩 실패: {e}")
        url_docs = []
    try:
        logging.info("PDF 문서 로딩 중...")
        pdf_loader = PyMuPDFLoader(pdf_path)
        pdf_docs = pdf_loader.load()
        logging.info(f"PDF 문서 {len(pdf_docs)}개 로딩 완료")
    except Exception as e:
        logging.error(f"PDF 문서 로딩 실패: {e}")
        pdf_docs = []
    return url_docs + pdf_docs

def build_vector_db(docs: List[Document], persist_dir: str):
    try:
        logging.info("문서 분할 중...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = splitter.split_documents(docs)
        logging.info(f"총 {len(split_docs)}개의 청크 생성 완료")
    except Exception as e:
        logging.error(f"문서 분할 실패: {e}")
        return
    try:
        logging.info("임베딩 및 벡터 저장 중...")
        embedding_model = UpstageEmbeddings(model="solar-embedding-1-large")
        vectorstore = Chroma.from_documents(split_docs, embedding_model, persist_directory=persist_dir)
        logging.info(f"벡터 DB 저장 완료: {persist_dir}")
    except Exception as e:
        logging.error(f"임베딩/벡터 저장 실패: {e}")

def run_rules_embedding(pdf_path: str = "축구규칙.pdf", vector_dir: str = DEFAULT_VECTOR_DB_DIR):
    all_docs = load_documents(URLS, pdf_path)
    build_vector_db(all_docs, vector_dir)