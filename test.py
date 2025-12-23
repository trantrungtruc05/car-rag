from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
import os



llm = ChatOpenAI()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Bạn là trợ lý hữu ích."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}")
])

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain = prompt | llm

conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

response = conversation.invoke(
    {"input": "Tôi tên là Trực"},
    config={"configurable": {"session_id": "user1"}}
)

response = conversation.invoke(
    {"input": "Tôi tên là gì?"},
    config={"configurable": {"session_id": "user1"}}
)

print(response.content)