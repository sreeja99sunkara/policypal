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


def backend_health() -> dict | None:
    """The API's /health payload, or None if it is not reachable."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        return resp.json() if resp.status_code == 200 else None
    except (requests.RequestException, ValueError):
        return None


def env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def forwarded_for() -> dict:
    """X-Forwarded-For for the API's per-IP rate limit. The API only ever sees
    this app as its client, so pass on the browser's address: the header the
    host's proxy set if there is one, else the address Streamlit saw."""
    try:
        xff = st.context.headers.get("X-Forwarded-For")
        ip = xff or getattr(st.context, "ip_address", None)
    except Exception:
        ip = None
    # only a real string: anything else (e.g. a test mock) would make requests fail
    return {"X-Forwarded-For": ip} if isinstance(ip, str) and ip.strip() else {}


@st.cache_data(show_spinner=False)
def read_pdf(path: str) -> bytes | None:
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return None


@st.cache_data(show_spinner=False, max_entries=32)
def render_page_png(pdf_bytes: bytes, page_number: int, needle: str) -> bytes | None:
    """PNG of one PDF page (viewer numbering), with `needle` highlighted if found."""
    try:
        import fitz  # PyMuPDF
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            if not 1 <= page_number <= doc.page_count:
                return None
            page = doc[page_number - 1]
            if needle:
                for rect in page.search_for(needle)[:5]:
                    page.add_highlight_annot(rect)
            return page.get_pixmap(dpi=110).tobytes("png")
    except Exception:
        return None


def highlight_needle(src: dict) -> str:
    """Text to highlight on the page: a table's title (its paragraph above the
    table appears verbatim on the page), else the chunk's first line."""
    if src.get("chunk_type") == "table" and src.get("title"):
        text = src["title"]
    else:
        text = next((ln for ln in src.get("text", "").splitlines() if ln.strip()), "")
    return " ".join(text.split())[:60]


def pages_label(src: dict) -> str:
    pages = src.get("pages") or [src["page"]]
    return f"Page {pages[0]}" if len(pages) == 1 else "Pages " + ", ".join(map(str, pages))


def render_sources(sources, cited_pages, key: str) -> None:
    if not sources:
        return
    if isinstance(sources, str):  # message stored before structured sources existed
        with st.expander("Sources used"):
            st.text(sources)
        return
    cited = set(cited_pages or [])
    is_cited = [bool(cited & set(s.get("pages") or [s["page"]])) for s in sources]
    order = sorted(range(len(sources)), key=lambda i: not is_cited[i])  # cited first, stable
    pdf_bytes = st.session_state.get("pdf_bytes")

    st.markdown("**Sources**")
    for i in order:
        src = sources[i]
        name = src.get("title") or ("Table" if src.get("chunk_type") == "table" else "Text")
        if len(name) > 70:
            name = name[:70].rsplit(" ", 1)[0] + " …"
        label = f"{pages_label(src)}, {name}" + ("  ·  ✅ cited in answer" if is_cited[i] else "")
        with st.expander(label):
            st.text(src.get("text", ""))
            if pdf_bytes and st.button("Show page", key=f"show_{key}_{i}"):
                png = render_page_png(pdf_bytes, src["page"], highlight_needle(src))
                if png:
                    st.image(png, caption=f"Page {src['page']}")
                else:
                    st.info("Could not render this page.")
        st.caption(src.get("snippet", ""))


# ---------- session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_doc" not in st.session_state:
    st.session_state.active_doc = None

# ---------- sidebar: upload ----------
with st.sidebar:
    st.header("Policy document")

    health = backend_health()
    # The API decides; DEMO_MODE here only matters while the API is unreachable.
    demo = bool(health.get("demo_mode")) if health else env_flag("DEMO_MODE")
    if health:
        st.success("Backend connected")
    elif demo:
        st.warning("The backend is starting (loading models and indexing the policy). "
                   "Refresh in a minute.")
    else:
        st.error(f"Backend not reachable at {API_URL}")

    uploaded = None
    if demo:
        policy_name = os.getenv("DEMO_POLICY_NAME", "Star Health Assure Insurance Policy (policy wording)")
        st.info(f"This demo answers questions about one fixed policy: {policy_name}. "
                "Uploads are disabled.")
        demo_pdf = os.getenv("DEFAULT_POLICY_PDF", "")
        if demo_pdf and "pdf_bytes" not in st.session_state:
            pdf = read_pdf(demo_pdf)  # same container: lets "Show page" render pages
            if pdf:
                st.session_state.pdf_bytes = pdf
    else:
        uploaded = st.file_uploader("Upload a policy PDF", type=["pdf"])

    if not demo and st.button("Index document", disabled=uploaded is None, use_container_width=True):
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
                    # kept so "Show page" can render pages of the indexed PDF
                    st.session_state.pdf_bytes = uploaded.getvalue()
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
        ("" if demo else "Uploading replaces the previously indexed document. ")
        + "Page numbers in answers and sources are the PDF viewer's page numbers "
        "(the first page is Page 1)."
    )
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- main: chat ----------
st.title("PolicyPal")
st.caption("Ask questions about your insurance policy. Answers come only from the uploaded document.")

for n, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_sources(msg.get("sources"), msg.get("cited_pages"), key=f"h{n}")

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
                    headers=forwarded_for(),
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]
                    sources = data.get("sources") or []
                    cited_pages = data.get("cited_pages") or []
                    st.markdown(answer)
                    # same key the history loop will use for this message after a rerun
                    render_sources(sources, cited_pages, key=f"h{len(st.session_state.messages)}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer,
                         "sources": sources, "cited_pages": cited_pages}
                    )
                else:
                    err = api_error(resp)
                    st.error(err)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Error: {err}"}
                    )
            except requests.RequestException:
                st.error("Could not reach the backend. Is uvicorn running?")