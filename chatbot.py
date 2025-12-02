import streamlit as st

st.set_page_config(page_title="Chatbot Demo", page_icon="🤖")

st.title("🤖 Chatbot cars")

# Khởi tạo history trong session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def fake_llm_reply(user_message: str) -> str:
    # TODO: chỗ này bạn thay bằng call OpenAI/Ollama/FastAPI
    return f"Mình là bot demo, bạn vừa nói: **{user_message}**"

# Ô input chat của user
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    # Lưu và hiển thị message của user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gọi "LLM"
    reply = fake_llm_reply(prompt)

    # Lưu & hiển thị message của bot
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)