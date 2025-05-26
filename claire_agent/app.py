# claire_agent/app.py
import streamlit as st
import datetime
from typing import List

from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage

# 내부 모듈 import
from core.config import (
    DEFAULT_OLLAMA_MODEL_NAME, 
    CURRENT_LOCATION_STR
)
from core.llm_services import (
    get_chat_llm_instance, 
    get_ollama_models_available,
    get_embedding_model # MemorySystem 초기화에 필요할 수 있음 (직접 사용은 X)
)
from core.db_services import (
    init_sqlite_db, 
    get_rag_vector_store, 
    get_memory_vector_store
)
from core.memory_system import MemorySystem
from prompts.system_prompts import SYSTEM_PROMPT_CONTENT_TEMPLATE

# --- 0. 애플리케이션 초기 설정 ---
st.set_page_config(page_title="Claire - 개인 AI 에이전트", layout="wide")
st.title("Claire 🧠 - Personal AI Agent")

# --- 1. 세션 상태 및 서비스 초기화 ---
if "app_initialized" not in st.session_state:
    init_sqlite_db() # SQLite DB 테이블 생성/확인
    st.session_state.app_initialized = True

if "selected_ollama_model" not in st.session_state:
    st.session_state.selected_ollama_model = DEFAULT_OLLAMA_MODEL_NAME
if "messages" not in st.session_state: # UI 표시용 메시지 리스트
    st.session_state.messages = [{"role": "assistant", "content": f"안녕하세요! Claire입니다. (모델: {st.session_state.selected_ollama_model})"}]
if "lc_memory" not in st.session_state: # LangChain 대화 기록용 메모리
    st.session_state.lc_memory = ConversationBufferWindowMemory(
        k=7, # 최근 7개 턴 기억 (조정 가능)
        return_messages=True, 
        memory_key="history", # 프롬프트의 MessagesPlaceholder와 일치
        input_key="human_input" # 프롬프트의 HumanMessage content와 일치
    )
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = datetime.datetime.now().strftime("session_%Y%m%d%H%M%S_%f")
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {"name": "주인님", "preferences_summary": "파악된 사용자 특정 선호 정보 없음."}

# --- 2. 핵심 서비스 인스턴스 로드 (Streamlit 캐싱 활용) ---
# LLM, 임베딩, 벡터DB는 Streamlit의 cache_resource를 통해 효율적으로 관리
embedding_model_global = get_embedding_model() # MemorySystem 생성자에 필요
rag_vector_store_global = get_rag_vector_store()
memory_vector_store_global = get_memory_vector_store()

# 현재 선택된 모델에 따라 ChatOllama 인스턴스 가져오기
# 이 인스턴스는 채팅 응답 생성 및 메모리 요약에 사용됩니다.
current_chat_llm = get_chat_llm_instance(st.session_state.selected_ollama_model)

# MemorySystem 인스턴스 (LLM, 임베딩, 메모리 VDB 전달)
memory_system_instance = MemorySystem(
    llm_for_summarization=current_chat_llm, # 채팅 LLM을 요약에도 사용
    embedding_instance=embedding_model_global,
    memory_vdb_instance=memory_vector_store_global
)

# --- 3. 사이드바 UI 구성 ---
with st.sidebar:
    st.header("🤖 모델 및 시스템 설정")
    
    available_ollama_models = get_ollama_models_available()
    # 현재 선택된 모델이 목록에 없으면 첫 번째 모델을 기본값으로 사용
    try:
        current_model_index = available_ollama_models.index(st.session_state.selected_ollama_model)
    except ValueError:
        current_model_index = 0
        if available_ollama_models: # 목록이 비어있지 않다면
            st.session_state.selected_ollama_model = available_ollama_models[0]
        else: # 사용 가능한 모델이 전혀 없다면 (Ollama 서버 문제 등)
            st.error("사용 가능한 Ollama 모델을 찾을 수 없습니다. Ollama 서버를 확인해주세요.")
            # 이 경우 앱 실행이 어려울 수 있으므로, 적절한 처리 필요 (예: 기본 모델 하드코딩)
            st.session_state.selected_ollama_model = "gemma3:4b" # 임시 fallback

    newly_selected_model = st.selectbox(
        "Ollama 모델 선택:",
        options=available_ollama_models,
        index=current_model_index,
        key="model_selector_sidebar_app" # 키 변경으로 충돌 방지
    )

    if newly_selected_model != st.session_state.selected_ollama_model:
        st.session_state.selected_ollama_model = newly_selected_model
        # 모델 변경 시 관련 캐시된 리소스 초기화
        get_chat_llm_instance.clear() 
        # MemorySystem은 내부적으로 LLM을 사용하므로, LLM이 바뀌면 MemorySystem도 다시 만들어야 함
        # 하지만 @st.cache_resource는 인자가 바뀌면 자동으로 재생성하므로 명시적 clear 불필요할 수 있음
        # 여기서는 명확성을 위해 호출 (get_memory_system_instance는 인자로 llm을 받음)
        st.success(f"모델이 {newly_selected_model}로 변경되었습니다. 페이지를 새로고침합니다.")
        st.rerun() # 변경사항 적용 및 LLM 인스턴스 재생성을 위해 페이지 새로고침

    st.markdown(f"<sub>현재 LLM: **{st.session_state.selected_ollama_model}**</sub>", unsafe_allow_html=True)

    with st.expander("⚙️ 사용자 프로필 및 기억 관리", expanded=False):
        st.subheader("사용자 프로필")
        st.session_state.user_profile["name"] = st.text_input(
            "사용자 이름 (호칭용)", 
            value=st.session_state.user_profile.get("name", "주인님"), 
            key="user_name_input_sidebar_app"
        )
        st.session_state.user_profile["preferences_summary"] = st.text_area(
            "사용자 선호도/정보 요약", 
            value=st.session_state.user_profile.get("preferences_summary", ""), 
            height=100, 
            help="AI가 사용자를 더 잘 이해하고 개인화된 답변을 제공하는 데 사용됩니다.",
            key="user_prefs_sidebar_app"
        )

        st.subheader("장기 기억 관리")
        use_llm_for_manual_summary = st.checkbox(
            "LLM으로 현재 대화 자동 요약 (수동 저장 시)", 
            value=True, 
            key="llm_summary_checkbox_sidebar_app"
        )
        user_provided_manual_summary = st.text_area(
            "또는, 직접 요약 입력 (수동 저장용):", 
            height=100, 
            key="user_summary_input_sidebar_app", 
            help="AI 요약 대신 직접 요약 입력 시 사용합니다."
        )

        if st.button("현재 대화 요약 저장 (수동)", key="save_summary_button_sidebar_app"):
            conversation_history_for_summary = "\n".join(
                [f"{m['role']}: {m['content']}" for m in st.session_state.messages if m['role'] != 'system']
            )
            if not conversation_history_for_summary.strip() and not user_provided_manual_summary.strip():
                st.warning("요약할 대화 내용이 없거나 직접 입력된 요약이 없습니다.")
            else:
                summary_input_for_consolidation = user_provided_manual_summary if user_provided_manual_summary.strip() else None
                with st.spinner("장기 기억 저장 중..."):
                    stored_ltm_entry = memory_system_instance.consolidate_session_memory(
                        session_id=st.session_state.current_session_id,
                        conversation_history_str=conversation_history_for_summary,
                        user_provided_summary=summary_input_for_consolidation,
                        use_llm_summary=use_llm_for_manual_summary if not summary_input_for_consolidation else False
                    )
                if stored_ltm_entry:
                    st.success(f"대화 요약 (ID: {stored_ltm_entry.id})이 장기 기억에 저장되었습니다!")
                else:
                    st.error("장기 기억 저장에 실패했습니다.")
        
        # 자동 장기 기억 저장 기능 활성화 체크박스
        st.session_state.auto_save_ltm_checkbox = st.checkbox( # session_state에 직접 할당
            "매 응답 후 자동 장기 기억 저장 (LLM 요약 사용)", 
            value=st.session_state.get("auto_save_ltm_checkbox", False), # 이전 값 유지
            key="auto_save_ltm_checkbox_app",
            help="LLM 응답 후 현재 대화 내용을 자동으로 요약하여 장기 기억에 저장합니다. (매번 LLM 호출이 추가로 발생)"
        )

        st.subheader("장기 기억 피드백")
        memory_id_for_feedback = st.number_input("피드백할 기억 ID (SQLite ID):", min_value=1, step=1, format="%d", key="fb_id_sidebar_app")
        new_importance_for_feedback = st.slider("새로운 중요도 점수:", 0.0, 1.0, 0.5, 0.05, key="fb_imp_sidebar_app")
        # new_summary_for_feedback = st.text_area("수정된 요약 (선택):", height=80, key="fb_summary_sidebar_app") # 향후 확장 가능

        if st.button("선택한 기억에 피드백 적용", key="fb_apply_sidebar_app"):
            if memory_id_for_feedback:
                feedback_success = memory_system_instance.apply_user_feedback_to_memory(
                    memory_sqlite_id=memory_id_for_feedback,
                    new_importance=new_importance_for_feedback
                    # new_summary=new_summary_for_feedback # 요약 수정 기능 추가 시
                )
                if feedback_success:
                    st.success(f"기억 ID {memory_id_for_feedback}에 피드백이 적용되었습니다.")
                else:
                    st.error(f"기억 ID {memory_id_for_feedback}에 대한 피드백 적용에 실패했습니다.")
            else:
                st.warning("피드백할 기억 ID를 입력하세요.")

        if st.button("주기적 기억 유지보수 실행", key="maint_button_sidebar_app"):
            with st.spinner("기억 유지보수 작업 진행 중..."):
                memory_system_instance.periodic_memory_maintenance()
            st.success("기억 유지보수 작업이 완료되었습니다.")
    
    st.caption(f"현재 세션 ID: {st.session_state.current_session_id}")
    st.markdown("---")
    if st.button("현재 대화창 내용 지우기 (단기 기억 초기화)", key="clear_chat_button_sidebar_app"):
        st.session_state.messages = [{"role": "assistant", "content": f"Claire입니다. (모델: {st.session_state.selected_ollama_model})"}]
        st.session_state.lc_memory.clear() # LangChain 메모리도 초기화
        # 새 세션 ID 부여 (선택 사항)
        # st.session_state.current_session_id = datetime.datetime.now().strftime("session_%Y%m%d%H%M%S_%f")
        st.rerun()

# --- 4. 메인 채팅 인터페이스 ---
# 이전 대화 내용 UI에 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if user_query := st.chat_input("Claire에게 메시지를 보내세요..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # AI 응답 생성
    with st.chat_message("assistant"):
        progress_bar = st.progress(0, text="Claire가 생각 중입니다...")
        response_placeholder = st.empty() # 스트리밍 응답을 위한 빈 공간
        assistant_response_final = ""
        
        try:
            # 1. 장기 기억 회상 (MemorySystem 사용)
            progress_bar.progress(10, text="과거 기억을 회상하는 중...")
            recalled_long_term_memories = memory_system_instance.retrieve_relevant_memories(user_query, top_k=1)
            recalled_ltm_str_for_prompt = "\n".join(
                [f"- (ID:{mem.id}, 중요도:{mem.user_importance_score:.2f}) {mem.summary}" for mem in recalled_long_term_memories]
            ) if recalled_long_term_memories else "회상된 주요 과거 대화 없음."

            # 2. RAG 문서 검색 (VectorStoreRetriever 사용)
            progress_bar.progress(30, text="관련 정보를 검색하는 중 (RAG)...")
            rag_context_str_for_prompt = ""
            if rag_vector_store_global: # RAG 벡터 스토어가 성공적으로 로드되었는지 확인
                try:
                    rag_retriever = rag_vector_store_global.as_retriever(search_kwargs={"k": 2}) # 상위 2개 문서 검색
                    retrieved_rag_documents = rag_retriever.invoke(user_query)
                    rag_context_str_for_prompt = "\n\n".join(
                        [f"문서 출처: {doc.metadata.get('source', '알 수 없음')}\n내용: {doc.page_content[:250]}..." for doc in retrieved_rag_documents]
                    ) if retrieved_rag_documents else "현재 질의와 관련된 외부 참조 정보 없음."
                except Exception as e_rag:
                    st.warning(f"RAG 검색 중 오류 발생: {e_rag}")
                    rag_context_str_for_prompt = "RAG 정보 검색에 실패했습니다."
            else:
                rag_context_str_for_prompt = "RAG 기능이 활성화되지 않았습니다 (벡터 스토어 로드 실패)."

            # 3. LLM에 전달할 최종 프롬프트 구성
            progress_bar.progress(50, text="AI에게 전달할 정보를 종합하는 중...")
            current_time_for_prompt = datetime.datetime.now().strftime("%Y년 %m월 %d일 %A %p %I:%M")
            
            # 시스템 프롬프트 내용 포매팅
            # str()로 감싸서 None일 경우 "None" 문자열이 들어가지 않도록 방지 (특히 recalled_ltm_str, rag_context_str)
            formatted_system_content = SYSTEM_PROMPT_CONTENT_TEMPLATE.format(
                current_datetime=str(current_time_for_prompt),
                current_location=str(CURRENT_LOCATION_STR),
                user_name=str(st.session_state.user_profile.get("name", "사용자")),
                user_profile_summary=str(st.session_state.user_profile.get("preferences_summary", "정보 없음")),
                recalled_ltm_str=str(recalled_ltm_str_for_prompt),
                rag_context_str=str(rag_context_str_for_prompt)
            )
            system_message_object = SystemMessage(content=formatted_system_content)

            # LangChain 메모리에서 이전 대화 기록 가져오기
            short_term_memory_messages: List[BaseMessage] = st.session_state.lc_memory.chat_memory.messages
            
            # 현재 사용자 입력 메시지 객체
            human_message_object = HumanMessage(content=user_query)

            # LLM에 전달할 전체 메시지 리스트
            # 순서: 시스템 메시지 -> 단기 기억(과거 대화) -> 현재 사용자 입력
            final_messages_for_llm: List[BaseMessage] = [system_message_object] + short_term_memory_messages + [human_message_object]
            
            # --- 디버깅용 프린트 (필요시 주석 해제) ---
            # print("\n--- DEBUG: 최종 메시지 리스트 (LLM 전달 직전) ---")
            # for i, msg_obj in enumerate(final_messages_for_llm):
            #     print(f"Msg {i}: type={type(msg_obj).__name__}, content='{msg_obj.content[:200]}...'")
            # print("--- END DEBUG ---")
            # --- 디버깅용 프린트 끝 ---

            # 4. ChatOllama 스트리밍 호출
            progress_bar.progress(70, text="Claire가 답변을 생성하는 중...")
            
            stream_from_llm = current_chat_llm.stream(final_messages_for_llm) # ChatOllama의 stream 메서드 사용
            for chunk in stream_from_llm: # chunk는 AIMessageChunk 객체
                assistant_response_final += chunk.content
                response_placeholder.markdown(assistant_response_final + "▌") # 타이핑 효과
            
            response_placeholder.markdown(assistant_response_final) # 최종 응답 표시

        except Exception as e_main_chat:
            st.error(f"AI 응답 생성 중 오류 발생: {e_main_chat}")
            # traceback.print_exc() # 개발 시 상세 오류 확인용
            assistant_response_final = "죄송합니다, 답변을 생성하는 중 예상치 못한 오류가 발생했습니다."
            response_placeholder.markdown(assistant_response_final)
        
        finally:
            progress_bar.empty() # 진행 표시줄 제거

    # UI 메시지 리스트 및 LangChain 메모리에 AI 응답 추가
    st.session_state.messages.append({"role": "assistant", "content": assistant_response_final})
    st.session_state.lc_memory.save_context(
        {"human_input": user_query}, # LangChain 메모리의 input_key
        {"output": assistant_response_final} # LangChain 메모리의 output_key (기본값)
    )

    # 자동 장기 기억 저장 기능 (활성화된 경우)
    if st.session_state.get("auto_save_ltm_checkbox", False):
        with st.spinner("자동 장기 기억 저장 중 (LLM 요약 사용)..."):
            # 현재까지의 전체 대화 내용을 기반으로 요약 (st.session_state.messages 사용)
            current_conversation_history_for_auto_save = "\n".join(
                [f"{m['role']}: {m['content']}" for m in st.session_state.messages if m['role'] != 'system']
            )
            if current_conversation_history_for_auto_save.strip():
                print(f"Auto-saving LTM for session: {st.session_state.current_session_id}")
                auto_saved_entry = memory_system_instance.consolidate_session_memory(
                    session_id=st.session_state.current_session_id, # 현재 세션 ID 사용
                    conversation_history_str=current_conversation_history_for_auto_save,
                    use_llm_summary=True # 자동 저장은 항상 LLM 요약 사용 (또는 다른 규칙 적용 가능)
                )
                if auto_saved_entry:
                    st.toast(f"대화 내용이 자동으로 장기 기억에 저장되었습니다 (ID: {auto_saved_entry.id}).", icon="💾")
                else:
                    st.toast("자동 장기 기억 저장에 실패했습니다.", icon="⚠️")