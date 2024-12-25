import streamlit as st
from RAG.generate import generate_answer

from RAG.generator import RAGChain
import datetime
import time

st.set_page_config(page_title="Chat Bot", page_icon="🌍")


def say_hello():
    current_hour = datetime.datetime.now().hour
    if 5 <= current_hour < 12:
        greeting = "Chào buổi sáng"
    elif 12 <= current_hour < 18:
        greeting = "Chào buổi chiều"
    elif 18 <= current_hour < 22:
        greeting = "Chào buổi tối"
    else:
        greeting = "Khuya rồi Chào bạn "
    return f"{greeting} , rất vui được hỗ trợ bạn,cần giúp đỡ gì ạ 😊?"


def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)


st.markdown(
    f"""
    <div style="display: flex; justify-content: center;">
        <img src="https://i.pinimg.com/originals/0c/67/5a/0c675a8e1061478d2b7b21b330093444.gif" width="200">
    </div>
    <div style="display: flex; justify-content: center;">
                    <p text-align:center;>  {say_hello()} </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = RAGChain()
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    answer = st.session_state.chain.invoke(prompt)
    # # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(stream_data(text=answer))
    # # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
