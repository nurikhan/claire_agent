# Claire - 개인 AI 에이전트

Claire는 사용자와 대화하고, 이전 대화 내용을 기억하며, 관련 정보를 참조하여 개인화된 답변을 제공하는 것을 목표로 하는 AI 에이전트입니다. Streamlit을 사용하여 웹 인터페이스를 제공하며, LangChain 프레임워크와 로컬 Ollama 모델을 활용합니다.

## 주요 기능

-   **대화형 AI**: 사용자와 자연스러운 대화를 나눕니다.
-   **장기 기억**: SQLite와 ChromaDB를 사용하여 이전 대화의 요약을 저장하고 회상합니다.
    -   사용자가 직접 또는 LLM을 통해 대화 요약을 저장할 수 있습니다.
    -   매 응답 후 자동으로 대화 내용을 요약하여 저장하는 옵션을 제공합니다.
-   **단기 기억**: 최근 대화 내용을 기억하여 문맥을 유지합니다 (LangChain `ConversationBufferWindowMemory`).
-   **RAG (Retrieval Augmented Generation)**: `rag_documents` 폴더 내의 문서를 참조하여 답변의 근거를 보강합니다.
-   **사용자 프로필**: 사용자의 이름과 선호도를 설정하여 개인화된 상호작용을 지원합니다.
-   **모델 선택**: 로컬에 설치된 Ollama 모델 중 원하는 모델을 선택하여 사용할 수 있습니다.
-   **기억 관리**:
    -   장기 기억에 대한 중요도 피드백을 적용할 수 있습니다.
    -   주기적인 기억 유지보수 (오래된 기억의 중요도 감소, 낮은 중요도 기억 삭제) 기능을 제공합니다.

## 프로젝트 구조

claire_agent/
├── app.py                   # Streamlit 메인 애플리케이션
├── core/
│   ├── init.py
│   ├── config.py            # 설정 변수
│   ├── data_models.py       # Pydantic 데이터 모델
│   ├── llm_services.py      # LLM 관련 서비스
│   ├── db_services.py       # 데이터베이스 관련 서비스
│   └── memory_system.py     # MemorySystem 클래스
├── prompts/
│   ├── init.py
│   └── system_prompts.py    # 시스템 프롬프트 템플릿
├── rag_documents/           # RAG 문서 저장 폴더 (사용자 추가)
├── vector_dbs/              # ChromaDB 데이터 저장 폴더 (자동 생성)
│   ├── chroma_db_memory/
│   └── chroma_db_rag/
├── .env                     # 환경 변수 설정 파일 (사용자 생성 필요)
├── requirements.txt         # 필요한 Python 패키지 목록
├── claire_memory.db         # SQLite 데이터베이스 파일 (자동 생성)
└── README.md                # 본 파일

## 설치 및 실행 방법

### 사전 준비 사항

1.  **Python**: Python 3.9 이상 버전이 설치되어 있어야 합니다.
2.  **Ollama**: 로컬에서 Ollama가 설치되고 실행 중이어야 합니다. 원하는 LLM 모델(예: `gemma3:4b`, `llama3` 등)이 Ollama를 통해 다운로드되어 있어야 합니다.
    -   Ollama 설치: [https://ollama.com/](https://ollama.com/)
    -   모델 다운로드 예시: `ollama pull gemma3:4b`
3.  **(선택) Git**: 버전 관리를 위해 Git을 사용하는 것이 좋습니다.

### 설치 과정

1.  **프로젝트 클론 (선택 사항):**
    ```bash
    git clone <저장소_URL>
    cd claire_agent
    ```
    또는 제공된 파일들을 `claire_agent` 폴더 아래에 직접 복사합니다.

2.  **가상 환경 생성 및 활성화 (권장):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **필수 패키지 설치:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **`.env` 파일 생성:**
    프로젝트 루트 디렉토리 (`claire_agent/`)에 `.env` 파일을 생성하고 아래 내용을 참고하여 개인 설정을 입력합니다.
    ```env
    # .env (예시)
    OLLAMA_MODEL_NAME="gemma3:4b"  # Ollama에서 사용할 기본 모델 이름
    HF_EMBEDDING_MODEL_NAME="jhgan/ko-sbert-nli" # 한국어 임베딩 모델
    OLLAMA_BASE_URL="http://localhost:11434"
    # CURRENT_USER_LOCATION="대한민국 서울" # 선택 사항
    ```

5.  **RAG 문서 추가 (선택 사항):**
    Claire가 답변 시 참조할 문서가 있다면 `claire_agent/rag_documents/` 폴더 안에 텍스트 파일 (`.txt`, `.md` 등) 형태로 넣어주세요. 애플리케이션 첫 실행 시 이 문서들이 RAG 벡터 스토어에 자동으로 인덱싱됩니다.

### 애플리케이션 실행

1.  Ollama 서버가 실행 중인지 확인합니다.
2.  `claire_agent` 폴더 (또는 프로젝트 루트)에서 다음 명령어를 실행합니다:
    ```bash
    streamlit run app.py
    ```
3.  웹 브라우저가 자동으로 열리거나, 터미널에 표시된 URL (보통 `http://localhost:8501`)로 접속하여 Claire와 대화를 시작할 수 있습니다.

## 사용 방법

-   **채팅**: 화면 하단의 입력창에 메시지를 입력하여 Claire와 대화합니다.
-   **사이드바 설정**:
    -   **Ollama 모델 선택**: 사용 가능한 Ollama 모델 목록에서 원하는 모델을 선택할 수 있습니다. 모델 변경 시 애플리케이션이 새로고침됩니다.
    -   **사용자 프로필**: '사용자 이름'과 '선호도/정보 요약'을 입력하여 Claire가 사용자를 더 잘 이해하도록 도울 수 있습니다.
    -   **장기 기억 관리**:
        -   **수동 저장**: 현재 대화 내용을 LLM 요약 또는 직접 입력한 요약으로 장기 기억에 저장할 수 있습니다.
        -   **자동 저장**: '매 응답 후 자동 장기 기억 저장' 옵션을 선택하면, Claire의 모든 응답 후 대화 내용이 자동으로 요약되어 저장됩니다. (LLM 호출이 추가로 발생)
        -   **피드백**: 저장된 장기 기억의 ID와 새로운 중요도 점수를 입력하여 기억의 가중치를 조절할 수 있습니다.
        -   **유지보수**: 오래된 기억의 중요도를 낮추거나 매우 낮은 중요도의 기억을 삭제하는 유지보수 작업을 실행할 수 있습니다.
    -   **대화창 초기화**: 현재 채팅창의 내용과 단기 기억을 모두 지우고 새 대화를 시작합니다.

## 데이터 저장 위치

-   **SQLite 데이터베이스**: `claire_agent/claire_memory.db` 파일에 장기 기억의 메타데이터(요약, 키워드, 생성 시간 등)가 저장됩니다.
-   **ChromaDB (벡터 스토어)**:
    -   `claire_agent/vector_dbs/chroma_db_memory/`: 장기 기억 요약문에 대한 벡터 임베딩이 저장됩니다 (유사도 검색용).
    -   `claire_agent/vector_dbs/chroma_db_rag/`: `rag_documents` 폴더 내 문서들에 대한 벡터 임베딩이 저장됩니다 (RAG용).
-   **RAG 문서**: `claire_agent/rag_documents/` 사용자가 직접 추가하는 참조 문서들이 위치합니다.

**주의**: `claire_memory.db` 파일과 `vector_dbs` 폴더는 애플리케이션 실행 중 자동으로 생성되거나 업데이트됩니다. 중요한 데이터를 백업하거나 초기화하려면 이 파일/폴더들을 직접 관리할 수 있습니다. 초기화 시에는 해당 파일과 폴더를 삭제 후 앱을 재시작하면 됩니다.

## 향후 개선 방향 (TODO)

-   장기 기억 요약 시 좀 더 지능적인 키워드 추출 방법 적용
-   RAG 검색 결과와 장기 기억 내용을 보다 정교하게 조합하여 프롬프트에 반영
-   사용자 피드백을 통해 기억의 내용을 직접 수정하는 기능 추가
-   세션 관리 기능 강화 (예: 이전 세션 불러오기)
-   더 다양한 데이터 소스 지원 (PDF, 웹페이지 등) for RAG
-   UI/UX 개선

## 기여

버그 리포트, 기능 제안, 코드 기여 등 모든 형태의 기여를 환영합니다. (실제 오픈소스 프로젝트라면 기여 가이드라인 추가)

---
