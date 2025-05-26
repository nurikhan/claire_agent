# claire_agent/core/memory_system.py
import sqlite3
import datetime
import json
from typing import List, Optional

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.schema import SystemMessage, HumanMessage

# 내부 모듈 import
from .config import SQLITE_DB_NAME
from .data_models import StoredMemoryEntry

class MemorySystem:
    def __init__(self, llm_for_summarization: ChatOllama, embedding_instance: HuggingFaceEmbeddings, memory_vdb_instance: Chroma):
        self.llm_summarizer = llm_for_summarization
        self.embeddings = embedding_instance # 현재 직접 사용하지 않으나, 향후 확장성 위해 유지
        self.memory_vdb = memory_vdb_instance

    def _execute_sqlite_query(self, query: str, params: tuple = (), fetch_one: bool = False, commit: bool = False):
        conn = sqlite3.connect(SQLITE_DB_NAME)
        cursor = conn.cursor()
        try:
            cursor.execute(query, params)
            result = None
            if fetch_one:
                result = cursor.fetchone()
            elif not commit: # SELECT 쿼리 (fetchall)
                result = cursor.fetchall()
            
            last_row_id = None
            if commit: # INSERT, UPDATE, DELETE 쿼리
                conn.commit()
                last_row_id = cursor.lastrowid
        except sqlite3.Error as e:
            print(f"SQLite error: {e} in query: {query} with params: {params}")
            # 프로덕션 환경에서는 더 정교한 로깅 및 예외 처리 필요
            raise  # 호출한 쪽에서 예외를 알 수 있도록 다시 발생
        finally:
            conn.close()
        
        # commit=True (INSERT/UPDATE/DELETE) 시에는 (결과, 마지막 행 ID) 반환
        # commit=False (SELECT) 시에는 결과만 반환
        return (result, last_row_id) if commit else result

    def _summarize_text_with_llm(self, text_to_summarize: str, max_length_chars: int = 200) -> str:
        if not self.llm_summarizer:
            return text_to_summarize[:max_length_chars] + ("..." if len(text_to_summarize) > max_length_chars else "")
        
        print(f"Summarizing text with LLM (approx. {max_length_chars} chars)... Input length: {len(text_to_summarize)} chars.")
        try:
            messages_for_summary = [
                SystemMessage(content=f"다음 텍스트를 한국어로 {max_length_chars}자 내외의 핵심만 간결하게 요약해줘. 다른 부연 설명 없이 요약 내용만 정확히 반환해줘."),
                HumanMessage(content=text_to_summarize)
            ]
            summary_text = ""
            # ChatOllama의 stream 메서드는 AIMessageChunk의 제너레이터를 반환
            for chunk in self.llm_summarizer.stream(messages_for_summary):
                summary_text += chunk.content # AIMessageChunk의 content 속성 사용
            
            summary_text = summary_text.strip()
            print(f"LLM Summary generated (first 100 chars): {summary_text[:100]}...")
            return summary_text if summary_text else "요약 내용을 생성하지 못했습니다."
        except Exception as e:
            print(f"Error during LLM summarization: {e}")
            # 요약 실패 시 원본 텍스트의 일부를 반환 (fallback)
            return text_to_summarize[:max_length_chars] + ("..." if len(text_to_summarize) > max_length_chars else "")

    def consolidate_session_memory(self, session_id: str, conversation_history_str: str, user_provided_summary: Optional[str] = None, use_llm_summary: bool = True):
        print(f"Consolidating memory for session: {session_id}")
        summary_to_store = ""
        if user_provided_summary and user_provided_summary.strip():
            summary_to_store = user_provided_summary.strip()
        elif use_llm_summary and conversation_history_str.strip():
            summary_to_store = self._summarize_text_with_llm(conversation_history_str)
        elif conversation_history_str.strip():
            summary_to_store = conversation_history_str[:300] + "..." if len(conversation_history_str) > 300 else conversation_history_str
        else:
            print("No content to consolidate.")
            return None
        
        if not summary_to_store:
            print("Summary could not be generated or provided.")
            return None
        
        keywords_list = [word for word in summary_to_store.replace(",", " ").replace(".", " ").split() if len(word) > 2][:5]
        keywords_json_str = json.dumps(keywords_list, ensure_ascii=False)
        now_iso = datetime.datetime.now().isoformat()

        last_sqlite_id = None 
        final_vector_id = None

        try:
            insert_query_step1 = """
                INSERT INTO long_term_memories 
                (session_id, summary, keywords, full_conversation_snippet, creation_time, last_accessed_time, user_importance_score) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            insert_params_step1 = (session_id, summary_to_store, keywords_json_str, conversation_history_str[:1000], now_iso, now_iso, 0.6)
            
            _, temp_last_sqlite_id = self._execute_sqlite_query(insert_query_step1, insert_params_step1, commit=True)

            if temp_last_sqlite_id:
                last_sqlite_id = temp_last_sqlite_id
                final_vector_id = str(last_sqlite_id) 

                update_query_step2 = "UPDATE long_term_memories SET vector_id = ? WHERE id = ?"
                self._execute_sqlite_query(update_query_step2, (final_vector_id, last_sqlite_id), commit=True)

                metadata = {
                    "sqlite_id": last_sqlite_id, 
                    "session_id": session_id, 
                    "type": "conversation_summary", 
                    "creation_time": now_iso, 
                    "keywords_json": keywords_json_str, 
                    "user_importance": 0.6
                }
                doc_to_add = Document(page_content=summary_to_store, metadata=metadata)
                
                self.memory_vdb.add_documents([doc_to_add], ids=[final_vector_id])
                # self.memory_vdb.persist() # ChromaDB는 persist_directory 사용 시 자동 관리 경향
                
                print(f"Memory (SQLite ID: {last_sqlite_id}, Vector ID: {final_vector_id}) stored.")
                return StoredMemoryEntry(
                    id=last_sqlite_id, vector_id=final_vector_id, session_id=session_id, 
                    summary=summary_to_store, keywords=keywords_list, creation_time=now_iso, 
                    last_accessed_time=now_iso, user_importance_score=0.6, 
                    full_conversation_snippet=conversation_history_str[:1000]
                )
            else:
                print("Failed to get last_sqlite_id after initial insert.")
                return None
        except sqlite3.IntegrityError as ie:
            print(f"SQLite IntegrityError during memory consolidation: {ie}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during memory consolidation: {e}")
            return None

    def retrieve_relevant_memories(self, query_text: str, top_k: int = 2) -> List[StoredMemoryEntry]:
        print(f"Retrieving LTM for query (first 50 chars): '{query_text[:50]}...' (top_k={top_k})")
        if not query_text.strip() or not self.memory_vdb:
            return []
        try:
            # score_threshold를 사용하여 너무 낮은 유사도 결과는 필터링 가능
            retrieved_docs = self.memory_vdb.similarity_search_with_score(query_text, k=top_k) 
        except Exception as e:
            print(f"Memory VDB search error: {e}")
            return []
        
        memories = []
        for doc, score in retrieved_docs:
            # print(f"Retrieved doc from VDB with score {score:.4f}: {doc.metadata}") # 디버깅
            sqlite_id_from_vdb_meta = doc.metadata.get("sqlite_id")
            if sqlite_id_from_vdb_meta:
                # SQLite 테이블 컬럼 순서: 
                # 0:id, 1:vector_id, 2:session_id, 3:summary, 4:keywords, 
                # 5:full_conversation_snippet, 6:creation_time, 7:last_accessed_time, 
                # 8:access_count, 9:user_importance_score
                row = self._execute_sqlite_query("SELECT * FROM long_term_memories WHERE id = ?", (sqlite_id_from_vdb_meta,), fetch_one=True)
                if row:
                    try:
                        keywords_json_data = row[4] # keywords는 5번째 컬럼 (인덱스 4)
                        loaded_keywords = []
                        if keywords_json_data and isinstance(keywords_json_data, str) and keywords_json_data.strip():
                            loaded_keywords = json.loads(keywords_json_data)
                        
                        entry_data = {
                            "id": row[0],
                            "vector_id": row[1], 
                            "session_id": row[2],
                            "summary": row[3],
                            "keywords": loaded_keywords,
                            "full_conversation_snippet": row[5],
                            "creation_time": row[6],
                            "last_accessed_time": row[7],
                            "access_count": row[8],
                            "user_importance_score": row[9]
                        }
                        entry = StoredMemoryEntry(**entry_data)
                        memories.append(entry)
                        
                        self._execute_sqlite_query(
                            "UPDATE long_term_memories SET last_accessed_time = ?, access_count = access_count + 1 WHERE id = ?", 
                            (datetime.datetime.now().isoformat(), sqlite_id_from_vdb_meta), 
                            commit=True
                        )
                    except json.JSONDecodeError as je:
                        print(f"Error decoding JSON for keywords (SQLite ID: {sqlite_id_from_vdb_meta}, data: '{row[4]}'): {je}")
                    except Exception as ex: # Pydantic ValidationError 등 포함
                        print(f"Error processing or validating memory entry (SQLite ID: {sqlite_id_from_vdb_meta}): {ex}")
        return memories

    def apply_user_feedback_to_memory(self, memory_sqlite_id: int, new_importance: float, new_summary: Optional[str] = None):
        print(f"Applying feedback to memory ID: {memory_sqlite_id}, New Importance: {new_importance}")
        current_row = self._execute_sqlite_query("SELECT * FROM long_term_memories WHERE id = ?", (memory_sqlite_id,), fetch_one=True)
        if not current_row:
            print(f"Memory ID {memory_sqlite_id} not found in SQLite."); return False

        summary_to_update = new_summary.strip() if new_summary and new_summary.strip() else current_row[3] 
        
        # SQLite 업데이트
        self._execute_sqlite_query(
            "UPDATE long_term_memories SET user_importance_score = ?, summary = ? WHERE id = ?", 
            (new_importance, summary_to_update, memory_sqlite_id), 
            commit=True
        )
        
        # 요약 내용이 실제로 변경된 경우 (또는 중요도가 변경된 경우에도 VDB 메타데이터 업데이트)
        if (new_summary and new_summary.strip() and new_summary.strip() != current_row[3]) or new_importance != current_row[9]:
            vector_id_str = current_row[1] 
            if not vector_id_str: vector_id_str = str(memory_sqlite_id)

            updated_row_for_vdb = self._execute_sqlite_query("SELECT * FROM long_term_memories WHERE id = ?", (memory_sqlite_id,), fetch_one=True)
            if not updated_row_for_vdb:
                print(f"Could not retrieve updated row for VDB sync (ID: {memory_sqlite_id})."); return True 

            keywords_json_str_updated = updated_row_for_vdb[4] 
            metadata_for_vdb = {
                "sqlite_id": updated_row_for_vdb[0], 
                "session_id": updated_row_for_vdb[2], 
                "type": "conversation_summary",
                "creation_time": updated_row_for_vdb[6], 
                "keywords_json": keywords_json_str_updated, 
                "user_importance": updated_row_for_vdb[9] # 업데이트된 중요도 사용
            }
            doc_for_vdb = Document(page_content=summary_to_update, metadata=metadata_for_vdb)
            
            try:
                # ChromaDB는 ID를 지정하여 add_documents를 호출하면 기존 문서를 덮어씁니다 (upsert 동작).
                self.memory_vdb.add_documents([doc_for_vdb], ids=[vector_id_str])
                # self.memory_vdb.persist() # 변경사항 즉시 반영
                print(f"VectorDB document for ID {vector_id_str} updated/added.")
            except Exception as e:
                print(f"Error updating/adding document in VectorDB for ID {vector_id_str}: {e}")
                # 필요시 삭제 후 재시도 로직 (이전 답변 참조)
        print(f"User feedback applied to memory ID: {memory_sqlite_id}")
        return True

    def periodic_memory_maintenance(self):
        print("Running periodic memory maintenance...")
        thirty_days_ago = (datetime.datetime.now() - datetime.timedelta(days=30)).isoformat()
        # 1. 오래된 기억의 중요도 감소
        self._execute_sqlite_query(
            "UPDATE long_term_memories SET user_importance_score = user_importance_score * 0.9 WHERE last_accessed_time < ? AND user_importance_score > 0.05", 
            (thirty_days_ago,), 
            commit=True
        )
        # print(f"Decayed importance for affected entries.") # 실제 영향받은 row 수는 알 수 없음

        # 2. 중요도가 매우 낮은 기억 삭제
        rows_to_delete_info = self._execute_sqlite_query("SELECT id, vector_id FROM long_term_memories WHERE user_importance_score < 0.01")
        
        if rows_to_delete_info: # None이 아니고 비어있지 않은 경우
            sqlite_ids_to_delete = [row[0] for row in rows_to_delete_info]
            # vector_id가 None일 수도 있으므로 필터링
            vector_ids_to_delete = [row[1] for row in rows_to_delete_info if row[1]] 
            
            if sqlite_ids_to_delete:
                 self._execute_sqlite_query(
                     f"DELETE FROM long_term_memories WHERE id IN ({','.join(['?']*len(sqlite_ids_to_delete))})", 
                     tuple(sqlite_ids_to_delete), 
                     commit=True
                 )
                 print(f"Pruned {len(sqlite_ids_to_delete)} entries from SQLite based on low importance.")

            if vector_ids_to_delete:
                try:
                    self.memory_vdb.delete(ids=vector_ids_to_delete)
                    # self.memory_vdb.persist() # 변경사항 즉시 반영
                    print(f"Pruned {len(vector_ids_to_delete)} entries from VectorDB.")
                except Exception as e:
                    print(f"Error pruning from VectorDB: {e}")
        else:
            print("No memories to prune based on low importance score.")