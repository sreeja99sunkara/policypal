import os

import requests
import streamlit as st

API_URL = os.getenv("POLICYPAL_API_URL", "http://localhost:8000")

st.set_page_config(page_title="PolicyPal", page_icon="🏥", layout="centered")


def api_error(resp: requests.Response) -> str:
    # Backend returns {"detail": "..."} for 400/404/500
    try:
        return resp.json().get("detail", resp.text)
    except ValueError:
        return resp.text


def backend_is_up() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=3).status_code == 200
    except requests.RequestException:
        return False


# ---------- session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_doc" not in st.session_state:
    st.session_state.active_doc = None

# ---------- sidebar: upload ----------
with st.sidebar:
    st.header("Policy document")

    if backend_is_up():
        st.success("Backend connected")
    else:
        st.error(f"Backend not reachable at {API_URL}")

    uploaded = st.file_uploader("Upload a policy PDF", type=["pdf"])

    if st.button("Index document", disabled=uploaded is None, use_container_width=True):
        with st.spinner("Reading, chunking and embedding. This can take a minute..."):
            try:
                resp = requests.post(
                    f"{API_URL}/api/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                    timeout=300,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.active_doc = data["filename"]
                    st.session_state.messages = []  # new document, fresh chat
                    st.success(f"Indexed {data['chunks_created']} chunks")
                else:
                    st.error(api_error(resp))
            except requests.RequestException:
                st.error("Could not reach the backend. Is uvicorn running?")

    if st.session_state.active_doc:
        st.caption(f"Active document: {st.session_state.active_doc}")

    st.divider()
    st.caption(
        "Uploading replaces the previously indexed document. "
        "Page numbers in citations are PDF page indices and start at 0."
    )
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- main: chat ----------
st.title("PolicyPal")
st.caption("Ask questions about your insurance policy. Answers come only from the uploaded document.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources used"):
                st.text(msg["sources"])

question = st.chat_input("e.g. What is the waiting period for pre-existing diseases?")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the policy..."):
            try:
                resp = requests.post(
                    f"{API_URL}/api/query",
                    json={"question": question},
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    answer, sources = data["answer"], data["sources"]
                    st.markdown(answer)
                    with st.expander("Sources used"):
                        st.text(sources)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                else:
                    err = api_error(resp)
                    st.error(err)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Error: {err}"}
                    )
            except requests.RequestException:
                st.error("Could not reach the backend. Is uvicorn running?")