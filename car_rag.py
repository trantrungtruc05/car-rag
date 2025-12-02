import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from qdrant_client import QdrantClient, models
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os


# CẤU HÌNH QDRANT
QDRANT_URL = "http://qdrant.tructran-api.xyz/"  # Thay bằng địa chỉ server Qdrant của bạn
COLLECTION_NAME = "car_sales_data"

data = pd.read_csv("cars.csv")
df = pd.DataFrame(data)

# Kỹ thuật quan trọng: Gộp các cột quan trọng thành một đoạn văn bản (Content)
# Để khi tìm kiếm, AI hiểu được toàn bộ ngữ cảnh của chiếc xe đó
df['page_content'] = df.apply(lambda x: f"Xe: {x['brand']} | Năm: {x['year']} | Giá: {x['price']} |  Mô tả: {x['description']}", axis=1)

# --- BƯỚC 2: VECTOR HÓA & LƯU TRỮ (INDEXING) ---
print("Đang nạp dữ liệu vào Vector DB...")

# Load dữ liệu từ DataFrame vào LangChain Documents
loader = DataFrameLoader(df, page_content_column="page_content")
docs = loader.load()

# --- BƯỚC 2: KẾT NỐI VÀ INDEXING LÊN QDRANT SERVER ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
qdrant_client = QdrantClient(url=QDRANT_URL)
print(f"DEBUG: QdrantClient version: {qdrant_client.__version__ if hasattr(qdrant_client, '__version__') else 'unknown'}")
print(f"DEBUG: Has search: {hasattr(qdrant_client, 'search')}")
# print(f"DEBUG: Dir: {dir(qdrant_client)}") # Uncomment if needed, but might be too verbose

try:
    # 1. KIỂM TRA: Xem Collection đã tồn tại trên Server chưa
    qdrant_client.get_collection(collection_name=COLLECTION_NAME)
    
    # Nếu tồn tại, chỉ cần kết nối và tái sử dụng (TẢI)
    print(f"Đã tìm thấy Collection '{COLLECTION_NAME}'. Đang kết nối...")
    vector_db = Qdrant(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings
    )

except Exception as e:
    # Nếu chưa tồn tại, tiến hành tạo mới (INDEXING & UPLOAD)
    print(f"Collection '{COLLECTION_NAME}' chưa tồn tại. Đang tiến hành Vector hóa và tải lên Qdrant Server...")
    
    # ⚠️ CHỈ CHẠY DÒNG NÀY MỘT LẦN KHI TẠO DỮ LIỆU BAN ĐẦU
    # Tạo collection thủ công để tránh lỗi tương thích thư viện và lỗi kết nối gRPC
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
    )
    
    # Khởi tạo vector store với client đã kết nối sẵn (HTTP)
    vector_db = Qdrant(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings
    )
    
    # Thêm dữ liệu
    vector_db.add_documents(docs)
    print("Indexing hoàn tất trên Server Qdrant.")


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
questions = [
    "Tôi cần một chiếc xe êm, yên tĩnh và mạnh mẽ, tôi có khoảng 600 triệu.",
    "Tìm xe bán tải màu cam cho tôi.",
]

response = rag_chain.invoke("Giá xe veloz ở TPHCM như thế nào ?")
print(f"AI: {response}")

# for q in questions:
#     print(f"\nUser: {q}")
    # response = rag_chain.invoke(q)
    # print(f"AI: {response}")

