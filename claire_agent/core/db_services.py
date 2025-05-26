# claire_agent/core/db_services.py
import streamlit as st
import sqlite3
import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# config 및 llm_services 모듈에서 필요한 요소 가져오기
from .config import (
    RAG_DOCS_PATH,
    VECTOR_DB_RAG_PATH,
    VECTOR_DB_MEMORY_PATH,
    SQLITE_DB_NAME
)
from .llm_services import get_embedding_model

@st.cache_resource
def get_rag_vector_store() -> Chroma:
    print("Initializing RAG Vector Store...")
    embedding_func = get_embedding_model()
    if os.path.exists(VECTOR_DB_RAG_PATH) and os.listdir(VECTOR_DB_RAG_PATH):
        print(f"Loading RAG vector store from {VECTOR_DB_RAG_PATH}")
        return Chroma(persist_directory=VECTOR_DB_RAG_PATH, embedding_function=embedding_func)
    else:
        print(f"No existing RAG vector store at {VECTOR_DB_RAG_PATH}. Attempting to create new one.")
        try:
            # RAG_DOCS_PATH에 문서가 있는지 확인
            if not os.path.exists(RAG_DOCS_PATH) or not os.listdir(RAG_DOCS_PATH):
                print(f"No documents found in RAG directory: {RAG_DOCS_PATH}. Creating an empty RAG DB.")
                # 빈 DB 생성
                db = Chroma(persist_directory=VECTOR_DB_RAG_PATH, embedding_function=embedding_func)
                db.persist()
                return db

            loader = DirectoryLoader(
                RAG_DOCS_PATH,
                glob="**/*.*", # 모든 파일 대상
                show_progress=True,
                use_multithreading=True,
                loader_cls=TextLoader, # 텍스트 파일 로더 사용
                silent_errors=True # 오류 발생 시 경고만 출력하고 계속 진행
            )
            documents = loader.load()

            if not documents:
                print(f"No documents successfully loaded from {RAG_DOCS_PATH}. Creating an empty RAG DB.")
                db = Chroma(persist_directory=VECTOR_DB_RAG_PATH, embedding_function=embedding_func)
                db.persist()
                return db

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            texts = text_splitter.split_documents(documents)
            print(f"Creating RAG vector store with {len(texts)} document chunks.")
            db = Chroma.from_documents(texts, embedding_func, persist_directory=VECTOR_DB_RAG_PATH)
            db.persist()
            print("RAG vector store created and persisted.")
            return db
        except Exception as e:
            st.error(f"Error creating RAG vector store: {e}")
            # 오류 발생 시에도 빈 DB 생성 시도
            db = Chroma(persist_directory=VECTOR_DB_RAG_PATH, embedding_function=embedding_func)
            db.persist()
            return db

@st.cache_resource
def get_memory_vector_store() -> Chroma:
    print(f"Initializing/Loading Memory Vector Store from {VECTOR_DB_MEMORY_PATH}...")
    embedding_func = get_embedding_model()
    db = Chroma(persist_directory=VECTOR_DB_MEMORY_PATH, embedding_function=embedding_func)
    # db.persist() # Chroma는 persist_directory 지정 시 자동으로 로드/저장 관리하므로 명시적 호출 불필요할 수 있음
    return db

def init_sqlite_db():
    """SQLite 데이터베이스와 long_term_memories 테이블을 초기화합니다."""
    print(f"Initializing SQLite DB at: {SQLITE_DB_NAME}")
    conn = sqlite3.connect(SQLITE_DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS long_term_memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        vector_id TEXT UNIQUE,
        session_id TEXT NOT NULL,
        summary TEXT NOT NULL,
        keywords TEXT,
        full_conversation_snippet TEXT,
        creation_time TEXT NOT NULL,
        last_accessed_time TEXT NOT NULL,
        access_count INTEGER DEFAULT 0,
        user_importance_score REAL DEFAULT 0.5
    )""")
    conn.commit()
    conn.close()
    print("SQLite DB initialized successfully.")