from logging import PlaceHolder
from typing import Set
import streamlit as st
from backend.core import run_llm
from dotenv import load_dotenv

load_dotenv()

st.header("LangChain Chat")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here...")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []


# 引用元をフォーマット
def create_sources_string(source_urls: Set[str]):
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(query=prompt)
        sources = [doc.metadata["source"] for doc in generated_response["source_documents"] ]

        # デバッグ用のprint文を追加
        print("Raw sources:", sources)
        for source in sources:
            print("Source URL:", repr(source))  # reprを使用して特殊文字を可視化

        formatted_response = (
            f"{generated_response['result']} \n\n {create_sources_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)

if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(generated_response)
