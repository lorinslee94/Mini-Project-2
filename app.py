
import os

import streamlit as st
from openai import OpenAI

from part3.head_agent import Head_Agent

def get_secret(name):
    return st.secrets.get(name) or os.getenv(name)

openai_api_key = get_secret("OPENAI_API_KEY")
pinecone_api_key = get_secret("PINECONE_API_KEY")
PINECONE_INDEX = "ml-textbook-rag-1536"

DEBUG = False

if not openai_api_key:
    st.error("OPENAI_API_KEY environment var not set.")
    st.stop()

if not pinecone_api_key:
    st.error("PINECONE_API_KEY environment var not set")
    st.stop()

# create the head agent once
if "head_agent" not in st.session_state:
    st.session_state.head_agent = Head_Agent(
        openai_key=openai_api_key,
        pinecone_key=pinecone_api_key,
        pinecone_index_name=PINECONE_INDEX,
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    result = st.session_state.head_agent.handle_turn(
        prompt, st.session_state.messages)

    with st.chat_message("assistant"):
        st.markdown(result["response"])

    st.session_state.messages.append(
        {"role": "assistant", "content": result["response"]}
    )

    if DEBUG:
        st.caption(
            f"path={result['agent_path']} refusal={result['refusal_flag']}"
        )

        if "metadata" in result:
            st.markdown("### Debug:")
            st.json(result["metadata"])
