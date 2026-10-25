import io
import json
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from docx import Document as DocxDocument
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# Enabled initial_sidebar_state so the hide/unhide toggle arrow is active
st.set_page_config(page_title="hasnain RAG", page_icon="📚", layout="wide", initial_sidebar_state="expanded")

DATA_DIR = Path("chat_data")
DATA_DIR.mkdir(exist_ok=True)

# Make sure the light theme config exists with our maroon/cream tones
_cfg = Path(".streamlit/config.toml")
if not _cfg.exists():
    _cfg.parent.mkdir(exist_ok=True)
    _cfg.write_text(
        '[theme]\nbase = "light"\nprimaryColor = "#800020"\nbackgroundColor = "#FDFBF7"\n'
        'secondaryBackgroundColor = "#F4ECE1"\ntextColor = "#2C1A1D"\n',
        encoding="utf-8",
    )

TYPE_STYLE = {
    "pdf": ("PDF", "#800020"),
    "docx": ("DOCX", "#6A2E35"),
    "txt": ("TXT", "#A45A2A"),
    "json": ("JSON", "#4A6B41"),
}

CHAT_MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.7-flash"]
EMBED_MODEL = "models/gemini-embedding-001"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=DM+Sans:wght@400;500;600&display=swap');

:root {
    --cream: #FDFBF7;
    --sand: #F4ECE1;
    --ink: #2C1A1D;
    --muted: #7A696C;
    --maroon: #800020;
    --maroon-deep: #500013;
    --rose-accent: #B03052;
    --gold: #D4A373;
    --line: #E8DCCF;
}
:root, .stApp { color-scheme: light; }

html, body, [class*="css"], .stApp { font-family: 'DM Sans', sans-serif; color: var(--ink); }
.stApp { background: var(--cream); }

/* Keep sidebar toggle icon visible */
header[data-testid="stHeader"] { visibility: visible; background: transparent; }
#MainMenu, footer { visibility: hidden; height: 0; }
.block-container { padding-top: 1.2rem; max-width: 1100px; }

/* ---------- Animations ---------- */
@keyframes fadeUp   { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes slideIn  { from { opacity: 0; transform: translateX(-12px); } to { opacity: 1; transform: none; } }
@keyframes floatA   { 0%,100% { transform: translate(0,0) scale(1); } 50% { transform: translate(-30px,18px) scale(1.12); } }
@keyframes floatB   { 0%,100% { transform: translate(0,0) scale(1); } 50% { transform: translate(26px,-16px) scale(1.08); } }
@keyframes dots     { 0%,80%,100% { transform: scale(.5); opacity: .35; } 40% { transform: scale(1); opacity: 1; } }

@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] { background: var(--sand); border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] * { color: var(--ink); }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    font-family: 'Fraunces', serif; color: var(--maroon-deep);
}
section[data-testid="stSidebar"] a { color: var(--maroon) !important; }
.side-title { font-family: 'Fraunces', serif; font-weight: 700; color: var(--maroon-deep);
              font-size: 1.15rem; margin: 1.1rem 0 .4rem 0; }

[data-testid="stWidgetLabel"] p, label p, label { color: var(--ink) !important; font-weight: 500; }

/* ---------- Inputs / selects ---------- */
div[data-baseweb="input"], div[data-baseweb="base-input"] {
    background: #fff !important; border-radius: 12px !important;
}
div[data-baseweb="input"] { border: 1px solid #D9C8B8 !important; }
div[data-baseweb="input"]:focus-within { border-color: var(--maroon) !important; box-shadow: 0 0 0 3px rgba(128,0,32,.12); }
input, textarea {
    color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important;
    caret-color: var(--maroon) !important; background: transparent !important;
}
input::placeholder, textarea::placeholder { color: #9A878A !important; -webkit-text-fill-color: #9A878A !important; }

div[data-baseweb="select"] > div {
    background: #fff !important; border: 1px solid #D9C8B8 !important; border-radius: 12px !important;
}
div[data-baseweb="select"] * { color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important; }
div[data-baseweb="select"] svg { fill: var(--maroon-deep) !important; }

/* Buttons */
.stButton > button {
    background: #fff !important; color: var(--maroon-deep) !important;
    border: 1.5px solid var(--maroon) !important; border-radius: 12px; font-weight: 600;
    transition: transform .15s ease, background .2s ease, box-shadow .2s ease;
}
.stButton > button:hover {
    background: var(--maroon) !important; color: #fff !important;
    transform: translateY(-1px); box-shadow: 0 6px 14px rgba(128,0,32,.22);
}
.stButton > button p { color: inherit !important; }

/* ---------- Hero (Maroon & Cream Theme) ---------- */
.hero {
    position: relative; overflow: hidden;
    background: linear-gradient(145deg, #420A14 0%, var(--maroon-deep) 50%, var(--maroon) 100%);
    border-radius: 24px; padding: 2.5rem 2.8rem; margin-bottom: 1.5rem;
    box-shadow: 0 12px 35px rgba(80,0,19,.22);
    border: 1px solid rgba(255, 255, 255, 0.1);
    animation: fadeUp .6s ease both;
}
.hero::before, .hero::after { content: ""; position: absolute; border-radius: 50%; filter: blur(12px); pointer-events: none; }
.hero::before { width: 300px; height: 300px; right: -70px; top: -110px;
                background: radial-gradient(circle, rgba(212,163,115,.35), transparent 65%);
                animation: floatA 9s ease-in-out infinite; }
.hero::after  { width: 340px; height: 340px; right: 120px; bottom: -190px;
                background: radial-gradient(circle, rgba(176,48,82,.4), transparent 65%);
                animation: floatB 11s ease-in-out infinite; }
.hero > * { position: relative; z-index: 1; }
.hero h1 { font-family: 'Fraunces', serif; font-weight: 700; font-size: 2.6rem;
           margin: 0 0 .4rem 0; color: #FFF8F0 !important; letter-spacing: -0.5px; }
.hero p { margin: 0; font-size: 1.05rem; color: #FFF8F0; opacity: .92; max-width: 620px; line-height: 1.5; }
.hero .pills { margin-top: 1.2rem; display: flex; gap: .55rem; flex-wrap: wrap; }
.hero .pill { font-size: .75rem; font-weight: 600; letter-spacing: .6px; color: #FFF8F0;
              padding: .3rem .8rem; border-radius: 999px;
              background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.25);
              transition: transform .2s ease, background .2s ease; }
.hero .pill:hover { transform: translateY(-2px); background: rgba(255,255,255,.22); }

/* ---------- Stat cards ---------- */
.stat { background: #fff; border: 1px solid var(--line); border-radius: 16px;
        padding: 1.1rem 1.3rem; border-top: 4px solid var(--maroon);
        box-shadow: 0 4px 12px rgba(44,26,29,.03);
        animation: fadeUp .55s ease both; transition: transform .2s ease, box-shadow .2s ease; }
.stat:hover { transform: translateY(-3px); box-shadow: 0 10px 22px rgba(44,26,29,.07); }
.stat.rose { border-top-color: var(--rose-accent); }
.stat.gold { border-top-color: var(--gold); }
.stat .num { font-family: 'Fraunces', serif; font-size: 1.9rem; font-weight: 700; line-height: 1.1; color: var(--ink); }
.stat .lbl { color: var(--muted); font-size: .82rem; margin-top: .2rem; }

/* ---------- File rows & Source cards ---------- */
.filerow { display: flex; align-items: center; gap: .75rem; background: #fff;
           border: 1px solid var(--line); border-radius: 12px; padding: .6rem .9rem; margin-bottom: .45rem; }
.badge { color: #fff !important; font-size: .68rem; font-weight: 700; letter-spacing: .8px;
         padding: .22rem .5rem; border-radius: 6px; min-width: 46px; text-align: center; }
.filerow .name { font-weight: 500; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--ink); }
.filerow .meta { color: var(--muted); font-size: .8rem; }

.src { background: #fff; border: 1px solid var(--line); border-left: 4px solid var(--gold);
       border-radius: 12px; padding: .8rem 1rem; margin-bottom: .6rem; font-size: .88rem; color: var(--ink); }
.src .tag { font-weight: 600; color: var(--maroon-deep); font-size: .8rem; margin-bottom: .3rem; }

/* ---------- Uploader & Chat UI ---------- */
[data-testid="stFileUploader"] section { background: #fff !important; border: 2px dashed #D2BFB0; border-radius: 18px; }
[data-testid="stChatMessage"] { background: #fff !important; border: 1px solid var(--line); border-radius: 18px; padding: 1rem 1.1rem; }
[data-testid="stChatInput"], [data-testid="stChatInput"] > div { background: #fff !important; border-radius: 18px !important; border: 1.5px solid #D9C8B8; }
[data-testid="stChatInput"]:focus-within { border-color: var(--maroon); box-shadow: 0 6px 22px rgba(128,0,32,.15); }
[data-testid="stExpander"] { background: #fff !important; border: 1px solid var(--line) !important; border-radius: 14px; }

.typing { display: inline-flex; gap: 6px; padding: .35rem 0; }
.typing span { width: 9px; height: 9px; border-radius: 50%; background: var(--maroon); animation: dots 1.2s infinite ease-in-out; }
.typing span:nth-child(2) { animation-delay: .15s; background: var(--rose-accent); }
.typing span:nth-child(3) { animation-delay: .3s; background: var(--gold); }

.empty { text-align: center; color: var(--muted); padding: 2.2rem 1.2rem;
         border: 1px dashed #D9C8B8; border-radius: 18px; background: rgba(255,255,255,.6); }
.empty b { color: var(--maroon-deep); }

/* Chat history sidebar items styling */
[class*="st-key-chat_"] {
    background: #fff; border: 1px solid var(--line); border-radius: 12px;
    padding: .3rem .4rem .1rem .4rem; margin-bottom: .5rem;
}
[class*="st-key-chat_on_"] { border: 1.5px solid var(--maroon); background: #FAF2F4; }
[class*="st-key-open_"] button { background: transparent !important; border: none !important; box-shadow: none !important; justify-content: flex-start !important; text-align: left; }
[class*="st-key-del_"] button { background: transparent !important; border: none !important; box-shadow: none !important; color: var(--rose-accent) !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

TYPING_HTML = '<div class="typing"><span></span><span></span><span></span></div>'

def chat_path(cid: str) -> Path:
    return DATA_DIR / cid

def save_chat(meta: dict):
    meta["updated"] = datetime.now().isoformat(timespec="seconds")
    p = chat_path(meta["id"])
    p.mkdir(parents=True, exist_ok=True)
    (p / "chat.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def read_chat(cid: str) -> dict | None:
    f = chat_path(cid) / "chat.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None

def list_chats() -> list[dict]:
    chats = []
    for d in DATA_DIR.iterdir():
        if d.is_dir():
            m = read_chat(d.name)
            if m:
                chats.append(m)
    return sorted(chats, key=lambda m: m.get("updated", ""), reverse=True)

for k, v in {
    "chat_id": None, "messages": [], "vectorstore": None, "files_meta": [],
    "stats": (0, 0, 0), "sig": None, "files_sig": None, "uploader_key": 0,
    "pending_load": None, "skipped": [],
}.items():
    st.session_state.setdefault(k, v)

def reset_chat():
    st.session_state.update(
        chat_id=None, messages=[], vectorstore=None, files_meta=[], stats=(0, 0, 0),
        sig=None, files_sig=None, skipped=[], pending_load=None,
        uploader_key=st.session_state.uploader_key + 1,
    )

def request_load(cid: str):
    st.session_state.pending_load = cid

def delete_chat(cid: str):
    shutil.rmtree(chat_path(cid), ignore_errors=True)
    if st.session_state.chat_id == cid:
        reset_chat()

# Sidebar Layout
with st.sidebar:
    st.subheader("Settings")
    api_key = st.text_input("Gemini API key", type="password", placeholder="Paste your Gemini API key")
    st.caption("Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)")

    choice = st.selectbox("Chat model", CHAT_MODELS + ["Custom..."])
    if choice == "Custom...":
        model_name = st.text_input("Model name", placeholder="e.g. gemini-3.5-flash")
    else:
        model_name = choice

    st.markdown('<div class="side-title">Your chats</div>', unsafe_allow_html=True)
    st.button("＋  New chat", key="new_chat", use_container_width=True, on_click=reset_chat)

    history = list_chats()
    if not history:
        st.caption("Upload a file and your chat will be saved here automatically.")
    for m in history:
        cid = m["id"]
        active = cid == st.session_state.chat_id
        n_qa = len(m.get("messages", [])) // 2
        updated = m.get("updated", "")[:16].replace("T", " ")
        with st.container(key=f"chat_{'on' if active else 'off'}_{cid}"):
            c1, c2 = st.columns([6, 1], vertical_alignment="center")
            c1.button(
                m.get("title", "Untitled"), key=f"open_{cid}", use_container_width=True,
                on_click=request_load, args=(cid,), help=m.get("title", ""),
            )
            c2.button("✕", key=f"del_{cid}", on_click=delete_chat, args=(cid,), help="Delete this chat")
            st.caption(f"{len(m.get('files', []))} file(s) · {n_qa} question(s) · {updated}")

    with st.expander("Retrieval settings"):
        chunk_size = st.slider("Chunk size", 500, 2000, 1000, step=100)
        chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, step=50)
        top_k = st.slider("Chunks per answer (k)", 2, 10, 4)

# Hero Header
st.markdown(
    """
    <div class="hero">
        <h1>hasnain RAG</h1>
        <p>Drop in your documents and ask questions in plain language.
        Every answer is pulled only from your files, with the source shown.</p>
        <div class="pills">
            <span class="pill">PDF</span><span class="pill">DOCX</span>
            <span class="pill">TXT</span><span class="pill">JSON</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not api_key:
    st.info("Add your Gemini API key in the sidebar to get started.")
    st.stop()
if not model_name:
    st.info("Type a model name in the sidebar.")
    st.stop()

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=api_key)

if st.session_state.pending_load:
    cid = st.session_state.pending_load
    st.session_state.pending_load = None
    meta = read_chat(cid)
    if meta:
        try:
            store = FAISS.load_local(
                str(chat_path(cid) / "index"), get_embeddings(),
                allow_dangerous_deserialization=True,
            )
            st.session_state.update(
                chat_id=cid, messages=meta.get("messages", []), vectorstore=store,
                files_meta=meta.get("files", []), stats=tuple(meta.get("stats", (0, 0, 0))),
                sig=None, files_sig=None, skipped=[],
                uploader_key=st.session_state.uploader_key + 1,
            )
        except Exception as e:
            st.error(f"Couldn't open that chat: {e}")

def decode_text(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")

def flatten_json(obj, prefix=""):
    lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            lines += flatten_json(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            lines += flatten_json(v, f"{prefix}[{i}]")
    else:
        lines.append(f"{prefix}: {obj}")
    return lines

def load_file(name: str, raw: bytes) -> list[Document]:
    ext = name.rsplit(".", 1)[-1].lower()
    docs: list[Document] = []

    if ext == "pdf":
        reader = PdfReader(io.BytesIO(raw))
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(Document(page_content=text, metadata={"source": name, "label": f"{name}, p. {i + 1}"}))
    elif ext == "docx":
        d = DocxDocument(io.BytesIO(raw))
        parts = [p.text for p in d.paragraphs if p.text.strip()]
        for table in d.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        if parts:
            docs.append(Document(page_content="\n".join(parts), metadata={"source": name, "label": name}))
    elif ext == "txt":
        text = decode_text(raw).strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": name, "label": name}))
    elif ext == "json":
        text = "\n".join(flatten_json(json.loads(decode_text(raw))))
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": name, "label": name}))
    return docs

def embed_in_batches(chunks: list[Document], embeddings, batch_size: int = 40) -> FAISS:
    store = None
    bar = st.progress(0.0, text="Embedding chunks...")
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        for attempt in range(4):
            try:
                if store is None:
                    store = FAISS.from_documents(batch, embeddings)
                else:
                    store.add_documents(batch)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(6 * (attempt + 1))
        bar.progress(min((start + batch_size) / len(chunks), 1.0), text="Embedding chunks...")
    bar.empty()
    return store

def build_vectorstore(files, size, overlap):
    all_docs, skipped = [], []
    for name, raw in files:
        try:
            loaded = load_file(name, raw)
        except Exception as e:
            skipped.append(f"{name} ({e.__class__.__name__})")
            continue
        if not loaded:
            skipped.append(f"{name} (no readable text)")
        all_docs += loaded

    if not all_docs:
        return None, 0, 0, skipped

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(all_docs)
    store = embed_in_batches(chunks, get_embeddings())
    return store, len(chunks), sum(len(d.page_content) for d in all_docs), skipped

uploaded = st.file_uploader(
    "Upload your documents", type=["pdf", "docx", "txt", "json"],
    accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}",
)

if uploaded:
    files_sig = tuple((f.name, f.size) for f in uploaded)
    sig = (files_sig, chunk_size, chunk_overlap)
    if st.session_state.sig != sig:
        with st.spinner("Reading your files..."):
            try:
                store, n_chunks, n_chars, skipped = build_vectorstore(
                    [(f.name, f.getvalue()) for f in uploaded], chunk_size, chunk_overlap
                )
            except Exception as e:
                st.error(f"Indexing failed: {e}")
                st.stop()
        st.session_state.sig = sig
        st.session_state.skipped = skipped

        if store is None:
            st.session_state.vectorstore = None
            st.error("None of those files had readable text.")
        else:
            same_files = st.session_state.files_sig == files_sig and st.session_state.chat_id
            cid = st.session_state.chat_id if same_files else uuid.uuid4().hex[:12]
            files_meta = [{"name": f.name, "size": f.size} for f in uploaded]
            title = files_meta[0]["name"] + (f" +{len(files_meta) - 1}" if len(files_meta) > 1 else "")

            chat_path(cid).mkdir(parents=True, exist_ok=True)
            store.save_local(str(chat_path(cid) / "index"))

            st.session_state.update(
                chat_id=cid, messages=[], vectorstore=store, files_meta=files_meta,
                stats=(len(uploaded), n_chunks, n_chars), files_sig=files_sig,
            )
            save_chat({
                "id": cid, "title": title, "created": datetime.now().isoformat(timespec="seconds"),
                "files": files_meta, "stats": [len(uploaded), n_chunks, n_chars], "messages": [],
            })
            st.rerun()

for s in st.session_state.skipped:
    st.warning(f"Skipped {s}")

if st.session_state.vectorstore is not None:
    n_files, n_chunks, n_chars = st.session_state.stats
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="stat"><div class="num">{n_files}</div><div class="lbl">Files indexed</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="stat rose"><div class="num">{n_chunks}</div><div class="lbl">Searchable chunks</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="stat gold"><div class="num">{n_chars:,}</div><div class="lbl">Characters read</div></div>', unsafe_allow_html=True)

    with st.expander("Your library", expanded=False):
        for f in st.session_state.files_meta:
            ext = f["name"].rsplit(".", 1)[-1].lower()
            label, color = TYPE_STYLE.get(ext, ("FILE", "#666"))
            st.markdown(
                f'<div class="filerow"><span class="badge" style="background:{color}">{label}</span>'
                f'<span class="name">{f["name"]}</span><span class="meta">{f["size"] / 1024:.1f} KB</span></div>',
                unsafe_allow_html=True,
            )

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions strictly using the context taken from the user's documents.\n"
     "Rules:\n"
     "1. Use ONLY the context below. Do not use outside knowledge.\n"
     "2. If the answer isn't in the context, say: \"I couldn't find that in the documents.\"\n"
     "3. Cite where you found it using the bracketed source labels, e.g. (report.pdf, p. 3).\n"
     "4. If several files are relevant, combine them and cite each one.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata['label']}]\n{d.page_content}" for d in docs)

def chunk_text(chunk) -> str:
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in chunk)
    return str(chunk)

def render_sources(sources: list[dict]):
    with st.expander(f"Sources ({len(sources)} chunks)"):
        for s in sources:
            st.markdown(f'<div class="src"><div class="tag">{s["label"]}</div>{s["text"]}</div>',
                        unsafe_allow_html=True)

if st.session_state.vectorstore is not None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])

    question = st.chat_input("Ask something about your documents...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": top_k})
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=api_key)
        chain = RAG_PROMPT | llm

        with st.chat_message("assistant"):
            slot = st.empty()
            slot.markdown(TYPING_HTML, unsafe_allow_html=True)
            answer = ""
            try:
                docs = retriever.invoke(question)
                for chunk in chain.stream({"context": format_docs(docs), "question": question}):
                    answer += chunk_text(chunk.content)
                    slot.markdown(answer + " ▌")
                slot.markdown(answer)
            except Exception as e:
                slot.empty()
                st.session_state.messages.pop()
                st.error(f"Something went wrong: {e}")
                st.stop()

            sources = [{"label": d.metadata["label"],
                        "text": d.page_content[:400].replace("\n", " ")} for d in docs]
            render_sources(sources)

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

        meta = read_chat(st.session_state.chat_id)
        if meta:
            if not any(m["role"] == "user" for m in meta.get("messages", [])):
                meta["title"] = question[:48] + ("…" if len(question) > 48 else "")
            meta["messages"] = st.session_state.messages
            save_chat(meta)
            if len(st.session_state.messages) == 2:
                st.rerun()
elif not uploaded:
    st.markdown(
        '<div class="empty"><b>Nothing here yet.</b><br>Upload a few files above, or open one of your saved chats from the sidebar.</div>',
        unsafe_allow_html=True,
    )