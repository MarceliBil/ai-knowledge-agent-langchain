import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

st.set_page_config(page_title="RAG Knowledge Agent")


@st.cache_resource(show_spinner=False)
def _build_chain():
    from rag.rag_chain import get_rag_chain
    return get_rag_chain()


if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

st.title("RAG Knowledge Agent")

hide_ui = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_ui, unsafe_allow_html=True)

custom_chat_css = """
<style>
.st-emotion-cache-khw9fs {
    background-color: #f2f2f2;
}

.st-emotion-cache-z68l0b {
    background-color: rgb(0 207 255);
}

[data-testid="stChatInput"] > div {
    border: 1px solid rgba(255, 255, 255, 0.6) !important;
    box-shadow: none !important;
    border-radius: 9999px !important;
}

[data-testid="stChatInput"] > div:focus-within {
    border: 1px solid #ffffff !important;
    box-shadow: 0 0 6px rgba(255, 255, 255, 0.4) !important;
}

.st-emotion-cache-184dg47 {
    background-color: rgb(0, 173, 64) !important;
}
</style>
"""
st.markdown(custom_chat_css, unsafe_allow_html=True)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Zadaj pytanie na temat zasad w Twojej organizacji...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.pending_prompt = prompt
    st.rerun()

if st.session_state.pending_prompt:
    pending = st.session_state.pending_prompt

    history = []
    for m in st.session_state.messages[:-1]:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")

        if "chain" not in st.session_state:
            st.session_state.chain = _build_chain()

        raw = st.session_state.chain.invoke(
            {"input": pending, "chat_history": history}
        )

        if isinstance(raw, dict):
            response = raw.get("answer") or raw.get("output") or str(raw)
        elif hasattr(raw, "content"):
            response = raw.content
        else:
            response = str(raw)

        placeholder.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response})
    st.session_state.pending_prompt = None
    st.rerun()
