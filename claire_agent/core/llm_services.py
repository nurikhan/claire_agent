# claire_agent/core/llm_services.py
import streamlit as st
import subprocess
from typing import List
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings

# config 모듈에서 설정값 가져오기
from .config import (
    DEFAULT_OLLAMA_MODEL_NAME,
    HF_EMBEDDING_MODEL_NAME,
    OLLAMA_BASE_URL
)

@st.cache_resource
def get_embedding_model() -> HuggingFaceEmbeddings:
    print(f"Initializing HuggingFace Embedding Model: {HF_EMBEDDING_MODEL_NAME}...")
    return HuggingFaceEmbeddings(
        model_name=HF_EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}, # CPU 사용 명시
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def get_chat_llm_instance(model_name: str = DEFAULT_OLLAMA_MODEL_NAME) -> ChatOllama:
    print(f"Initializing ChatOllama with model: {model_name} at {OLLAMA_BASE_URL}...")
    return ChatOllama(model=model_name, base_url=OLLAMA_BASE_URL)

def get_ollama_models_available() -> List[str]:
    """시스템에 설치된 Ollama 모델 목록을 가져옵니다."""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1: # 첫 줄은 헤더
            return [line.split()[0] for line in lines[1:]] # 모델명은 첫 번째 컬럼
    except FileNotFoundError:
        print("Ollama command not found. Please ensure Ollama is installed and in your PATH.")
        return [DEFAULT_OLLAMA_MODEL_NAME]
    except subprocess.CalledProcessError as e:
        print(f"Error fetching Ollama models via subprocess: {e}")
        return [DEFAULT_OLLAMA_MODEL_NAME]
    except Exception as e:
        print(f"An unexpected error occurred while fetching Ollama models: {e}")
        return [DEFAULT_OLLAMA_MODEL_NAME]
    return [DEFAULT_OLLAMA_MODEL_NAME] # 예외 발생 시 기본 모델 반환