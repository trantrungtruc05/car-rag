from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os


QDRANT_URL = "http://qdrant.tructran-api.xyz/"  # Thay bằng địa chỉ server Qdrant của bạn
COLLECTION_NAME = "car_sales_data"

qdrant_client = QdrantClient(url=QDRANT_URL)

qdrant_client.get_collection(collection_name=COLLECTION_NAME)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


vector_db = Qdrant(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings
    )

retriever = vector_db.as_retriever(search_kwargs={"k": 7})


# --- BƯỚC 3: XÂY DỰNG RAG CHAIN ---
template = """Bạn là chuyên gia tư vấn bán xe hơi thông thái.
Hãy trả lời câu hỏi của khách hàng CHỈ dựa trên thông tin xe được cung cấp dưới đây.
Nếu không tìm thấy xe phù hợp, hãy nói rõ là kho xe hiện tại chưa có xe đó.

Thông tin xe tìm được từ kho dữ liệu:
{context}

Câu hỏi của khách hàng: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(model_name="gpt-5.1-2025-11-13", temperature=0.3)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- BƯỚC 4: CHẠY THỬ NGHIỆM ---
print("-" * 50)


response = rag_chain.invoke("Giá xe veloz  như thế nào ?")
print(f"AI: {response}")