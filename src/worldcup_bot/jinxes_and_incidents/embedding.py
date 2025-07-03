# embedding.py
import os
from typing import List

from langchain.schema import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import Chroma


def load_wikipedia_docs() -> List[Document]:
    years = [2006, 2010, 2014, 2018, 2022]
    wiki_urls = [
        f"https://en.wikipedia.org/wiki/List_of_{year}_FIFA_World_Cup_controversies"
        for year in years
    ]
    wiki_loader = WebBaseLoader(wiki_urls)
    wiki_docs = wiki_loader.load()
    print(f"📘 Wikipedia 문서 {len(wiki_docs)}개 로드 완료.")
    return wiki_docs


def load_namuwiki_docs(folder_path: str = "worldcup_incidents") -> List[Document]:
    namu_docs = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                content = f.read()
            namu_docs.append(Document(
                page_content=content,
                metadata={"source": filename, "title": filename.replace(".txt", "")}
            ))

    print(f"📗 나무위키 문서 {len(namu_docs)}개 수동 로드 완료.")
    return namu_docs

def load_jinxes_embedding(
    wiki_docs: List[Document],
    namu_docs: List[Document],
    persist_dir: str = "./chroma_jinxes"
) -> None:
    """
    문서 로딩 → 분할 → 임베딩 → 벡터스토어 저장 (Chroma)
    """
    all_docs = wiki_docs + namu_docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(all_docs)
    print(f"📄 총 {len(split_docs)}개의 청크 생성 완료.")

    embedding_model = UpstageEmbeddings(model="solar-embedding-1-large")
    Chroma.from_documents(split_docs, embedding_model, persist_directory=persist_dir)
    print(f"🧠 Chroma 벡터스토어 저장 완료: {persist_dir}")
