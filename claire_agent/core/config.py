# claire_agent/core/config.py
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# LLM 및 임베딩 모델 설정
DEFAULT_OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "gemma3:4b")
HF_EMBEDDING_MODEL_NAME = os.getenv("HF_EMBEDDING_MODEL_NAME", "jhgan/ko-sbert-nli")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 프로젝트 루트 디렉토리
RAG_DOCS_PATH = os.path.join(BASE_DIR, "rag_documents/")
VECTOR_DB_ROOT_PATH = os.path.join(BASE_DIR, "vector_dbs") # Chroma DB 저장용 루트 폴더
VECTOR_DB_RAG_PATH = os.path.join(VECTOR_DB_ROOT_PATH, "chroma_db_rag/")
VECTOR_DB_MEMORY_PATH = os.path.join(VECTOR_DB_ROOT_PATH, "chroma_db_memory/")
SQLITE_DB_NAME = os.path.join(BASE_DIR, "claire_memory.db") # SQLite DB 파일 경로

# 기타 설정
CURRENT_LOCATION_STR = os.getenv("CURRENT_USER_LOCATION", "대한민국 경기도 용인시") # 사용자의 현재 위치

# 폴더 생성 (존재하지 않을 경우)
if not os.path.exists(RAG_DOCS_PATH):
    os.makedirs(RAG_DOCS_PATH)
if not os.path.exists(VECTOR_DB_ROOT_PATH):
    os.makedirs(VECTOR_DB_ROOT_PATH)
if not os.path.exists(VECTOR_DB_RAG_PATH):
    os.makedirs(VECTOR_DB_RAG_PATH)
if not os.path.exists(VECTOR_DB_MEMORY_PATH):
    os.makedirs(VECTOR_DB_MEMORY_PATH)