"""
RAG Chatbot tư vấn bán xe hơi
- Tìm kiếm xe từ database Qdrant
- Trả lời câu hỏi có ngữ cảnh với LangChain
- Nhớ lịch sử hội thoại
"""

import os
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# =============================================================================
# CONFIGURATION - Cấu hình
# =============================================================================

QDRANT_URL = "http://47.129.184.129:6333/"
COLLECTION_NAME = "car_sales_data"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-5.1-2025-11-13"
LLM_TEMPERATURE = 0.3
TOP_K_RESULTS = 7

# =============================================================================
# SETUP - Khởi tạo components
# =============================================================================

def setup_vector_db():
    """Khởi tạo Qdrant vector database và retriever"""
    qdrant_client = QdrantClient(url=QDRANT_URL)
    qdrant_client.get_collection(collection_name=COLLECTION_NAME)
    
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_db = Qdrant(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings
    )
    
    return vector_db.as_retriever(search_kwargs={"k": TOP_K_RESULTS})

def setup_llm():
    """Khởi tạo ChatGPT model"""
    return ChatOpenAI(model_name=LLM_MODEL, temperature=LLM_TEMPERATURE)

# Khởi tạo
retriever = setup_vector_db()
llm = setup_llm()

# =============================================================================
# PROMPTS - Template câu hỏi
# =============================================================================

# Prompt 1: Viết lại câu hỏi thành câu độc lập có đầy đủ ngữ cảnh
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Nhiệm vụ: Viết lại câu hỏi cuối của user thành một câu hỏi ĐỘC LẬP có đầy đủ ngữ cảnh.\n\n"
     "QUY TẮC QUAN TRỌNG:\n"
     "1. CHỈ viết lại câu hỏi, KHÔNG được trả lời câu hỏi\n"
     "2. Nếu câu hỏi đã độc lập, giữ nguyên\n"
     "3. Nếu câu hỏi thiếu ngữ cảnh, thêm thông tin từ lịch sử chat\n\n"
     "Ví dụ:\n"
     "- Input: 'Xe nào rẻ nhất?' (sau khi hỏi về Toyota)\n"
     "- Output: 'Trong các xe Toyota vừa được nhắc đến, xe nào có giá rẻ nhất?'\n"
     "- KHÔNG ĐƯỢC: 'Toyota Vios là xe rẻ nhất...' (đây là trả lời, không phải viết lại!)"),
    MessagesPlaceholder("chat_history"),
    ("human", "Viết lại câu hỏi sau thành câu độc lập: {question}")
])

# Prompt 2: Trả lời câu hỏi dựa trên context từ database
answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Bạn là chuyên gia tư vấn bán xe hơi thông thái.\n"
     "Chỉ trả lời dựa trên CONTEXT (kho xe). Nếu không có xe phù hợp, nói rõ là kho chưa có.\n"
     "Trả lời ngắn gọn, liệt kê dạng bullet nếu là danh sách."),
    MessagesPlaceholder("chat_history"),
    ("human",
     "CONTEXT:\n{context}\n\n"
     "Câu hỏi: {question}")
])

# =============================================================================
# CHAINS - Chuỗi xử lý
# =============================================================================

# Chain 1: Viết lại câu hỏi có ngữ cảnh
standalone_question_chain = (
    contextualize_prompt
    | llm
    | StrOutputParser()
)

def retrieve_context(inputs: dict) -> str:
    """
    Lấy context từ vector database
    1. Viết lại câu hỏi thành câu độc lập
    2. Tìm kiếm trong Qdrant
    3. Trả về context dạng text
    """
    standalone_q = standalone_question_chain.invoke({
        "question": inputs["question"],
        "chat_history": inputs.get("chat_history", [])
    })
    docs = retriever.invoke(standalone_q)
    return "\n\n".join([d.page_content for d in docs])

# Chain 2: RAG pipeline hoàn chỉnh (chưa có memory)
base_rag_chain = (
    {
        "context": retrieve_context,  # Lấy context từ database
        "question": lambda x: x["question"],  # Lấy câu hỏi gốc
        "chat_history": lambda x: x.get("chat_history", []),  # Lấy lịch sử
    }
    | answer_prompt  # Ghép vào prompt
    | llm  # Gửi cho ChatGPT
    | StrOutputParser()  # Parse output
)

# =============================================================================
# MEMORY - Quản lý lịch sử hội thoại
# =============================================================================

_store = {}  # Dictionary lưu lịch sử: session_id -> ChatMessageHistory

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Lấy hoặc tạo mới lịch sử chat cho session"""
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]

# Chain 3: RAG với memory - TỰ ĐỘNG lưu/load lịch sử
rag_chain_with_memory = RunnableWithMessageHistory(
    base_rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# =============================================================================
# DEMO - Chạy thử
# =============================================================================

if __name__ == "__main__":
    session_id = "truc_car_chat_001"
    
    # Câu hỏi 1
    print("-" * 50)
    print("🔍 Câu hỏi 1: Cho tôi danh sách tầm 5 xe ô tô honda ?")
    print("-" * 50)
    response1 = rag_chain_with_memory.invoke(
        {"question": "Cho tôi danh sách tầm 5 xe ô tô honda ?"},
        config={"configurable": {"session_id": session_id}},
    )
    print(response1)
    
    # Câu hỏi 2 - Bot sẽ nhớ câu 1
    print("\n" + "-" * 50)
    print("🔍 Câu hỏi 2: Trong danh sách bạn mới gửi, xe nào mắc nhất")
    print("-" * 50)
    response2 = rag_chain_with_memory.invoke(
        {"question": "Trong danh sách bạn mới gửi, xe nào mắc nhất"},
        config={"configurable": {"session_id": session_id}},
    )
    print(response2)


    # Câu hỏi 3 - Bot sẽ nhớ câu 1 và câu 2
    print("\n" + "-" * 50)
    print("🔍 Câu hỏi 3: Cho tôi mô tả xe này với")
    print("-" * 50)
    response3 = rag_chain_with_memory.invoke(
        {"question": "Cho tôi mô tả xe này với"},
        config={"configurable": {"session_id": session_id}},
    )
    print(response3)