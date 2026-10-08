import os
import tempfile

import streamlit as st

from chat_store import (
    delete_all_chats,
    delete_chat,
    list_chats,
    load_chat,
    make_title,
    new_chat_id,
    save_chat,
)
from extras import chat_to_pdf, chat_to_text, highlight_html, save_feedback
from rag_core import (
    LANGUAGES,
    answer_question,
    build_book_index,
    get_embeddings,
    get_llm,
    list_books,
    load_books,
    text_to_speech,
    transcribe,
)

st.set_page_config(page_title="MediBot", page_icon="🩺")

ALL_BOOKS = "📚 All books"
DISCLAIMER = "⚠️ For information only, not a substitute for professional medical advice."

st.session_state.setdefault("messages", [])
st.session_state.setdefault("audio_key", 0)
st.session_state.setdefault("chat_id", None)      # None = a new chat that is not saved yet
st.session_state.setdefault("chat_title", "")


# ---------- cached resources ----------
@st.cache_resource
def get_models():
    return get_embeddings(), get_llm()


@st.cache_resource
def get_db(selection, books_signature):
    embeddings, _ = get_models()
    names = list(books_signature) if selection == ALL_BOOKS else [selection]
    return load_books(names, embeddings)


# ---------- chat history ----------
def persist():
    """Save the current chat to chats/<id>.json (creates it on the first message)."""
    msgs = st.session_state.messages
    if not msgs:
        return
    if st.session_state.chat_id is None:
        st.session_state.chat_id = new_chat_id()
        st.session_state.chat_title = make_title(msgs[0]["content"])
    save_chat(st.session_state.chat_id, st.session_state.chat_title, msgs)


def new_chat():
    st.session_state.chat_id = None
    st.session_state.chat_title = ""
    st.session_state.messages = []


def open_chat(chat_id):
    chat = load_chat(chat_id)
    if chat:
        st.session_state.chat_id = chat_id
        st.session_state.chat_title = chat.get("title", "Untitled chat")
        st.session_state.messages = chat.get("messages", [])


def remove_chat(chat_id):
    delete_chat(chat_id)
    if st.session_state.chat_id == chat_id:
        new_chat()


def remove_all_chats():
    delete_all_chats()
    new_chat()


# ---------- button callbacks ----------
def on_feedback(idx, rating):
    msg = st.session_state.messages[idx]
    question = st.session_state.messages[idx - 1]["content"]
    pages = ",".join(str(p) for p in sorted({s["page"] for s in msg.get("sources", [])}))
    msg["feedback"] = rating
    save_feedback(question, msg["content"], rating, msg.get("book", ""), pages)
    persist()


def on_listen(idx):
    msg = st.session_state.messages[idx]
    try:
        msg["audio"] = text_to_speech(msg["content"], msg.get("lang_code", "en"))
    except Exception as e:
        st.toast(f"Voice output failed: {e}")


# ---------- chat rendering ----------
def render_message(idx, msg):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] != "assistant" or msg.get("error"):
            return

        if msg.get("sources"):
            with st.expander(f"📖 Passages used ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['book']} - PDF page {s['page']}**")
                    st.markdown(
                        highlight_html(s["text"], msg.get("search_query", "")),
                        unsafe_allow_html=True,
                    )
                    st.divider()

        if msg.get("audio"):
            st.audio(msg["audio"], format="audio/mp3")

        st.caption(DISCLAIMER)

        fb_col, listen_col, _ = st.columns([2, 2, 5])
        with fb_col:
            if msg.get("feedback"):
                st.caption("✅ Thanks for the feedback")
            else:
                up, down = st.columns(2)
                up.button("👍", key=f"up_{idx}", on_click=on_feedback, args=(idx, "up"))
                down.button("👎", key=f"down_{idx}", on_click=on_feedback, args=(idx, "down"))
        listen_col.button("🔊 Listen", key=f"tts_{idx}", on_click=on_listen, args=(idx,))


# ---------- sidebar ----------
books = list_books()

with st.sidebar:
    # --- chat history ---
    st.button("➕ New chat", on_click=new_chat, use_container_width=True, type="primary")

    chats = list_chats()
    if chats:
        search = st.text_input(
            "Search chats", placeholder="Search chats", label_visibility="collapsed"
        )
        shown = [c for c in chats if search.lower() in c["title"].lower()]
        with st.container(height=230):
            if not shown:
                st.caption("No chats match your search.")
            for c in shown:
                row_open, row_del = st.columns([5, 1])
                is_active = c["id"] == st.session_state.chat_id
                row_open.button(
                    c["title"],
                    key=f"open_{c['id']}",
                    on_click=open_chat,
                    args=(c["id"],),
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                )
                row_del.button(
                    "🗑️", key=f"del_{c['id']}", on_click=remove_chat, args=(c["id"],),
                    help="Delete this chat",
                )
    else:
        st.caption("Your saved chats will appear here.")

    if st.session_state.chat_id:
        with st.expander("✏️ Rename this chat"):
            new_title = st.text_input(
                "Title", value=st.session_state.chat_title,
                key=f"rename_{st.session_state.chat_id}",
            )
            if st.button("Save title"):
                st.session_state.chat_title = make_title(new_title)
                persist()
                st.rerun()

    if chats:
        with st.expander("⚠️ Delete all chats"):
            sure = st.checkbox("Yes, delete every saved chat")
            if sure:
                st.button("Delete all chats", on_click=remove_all_chats)

    st.divider()

    # --- books ---
    st.header("📚 Books")
    options = ([ALL_BOOKS] if len(books) > 1 else []) + books
    selection = st.selectbox("Search in", options) if options else None

    uploaded = st.file_uploader("Add your own PDF", type="pdf")
    if uploaded is not None and st.button("Build index for this PDF"):
        embeddings, _ = get_models()
        bar = st.progress(0.0, text="Reading PDF and creating embeddings...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name
        try:
            name, n_pages, n_chunks = build_book_index(
                tmp_path,
                uploaded.name,
                embeddings,
                progress=lambda f: bar.progress(f, text=f"Embedding... {int(f * 100)}%"),
            )
            get_db.clear()
            st.session_state["flash"] = (
                f"✅ '{name}' is ready ({n_pages} pages, {n_chunks} chunks). "
                "Select it in the dropdown."
            )
            st.rerun()
        except Exception as e:
            bar.empty()
            st.error(f"Could not build the index: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # --- language and voice ---
    st.header("🌐 Language")
    lang_label = st.selectbox("Answer language", list(LANGUAGES))
    language, lang_code = LANGUAGES[lang_label]

    st.header("🎤 Voice")
    audio = None
    if hasattr(st, "audio_input"):
        audio = st.audio_input(
            "Record your question", key=f"audio_{st.session_state.audio_key}"
        )
    else:
        st.caption("Voice input needs a newer Streamlit: pip install --upgrade streamlit")

    # --- export ---
    st.header("💾 Export this chat")
    if st.session_state.messages:
        st.download_button(
            "Download as .txt",
            chat_to_text(st.session_state.messages),
            file_name="medibot_chat.txt",
            mime="text/plain",
        )
        try:
            st.download_button(
                "Download as .pdf",
                chat_to_pdf(st.session_state.messages),
                file_name="medibot_chat.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.caption(f"PDF export unavailable: {e}")
    else:
        st.caption("Ask a question first, then export it here.")


# ---------- main page ----------
st.title("🩺 MediBot - Medical RAG Chatbot")
if st.session_state.chat_id:
    st.caption(f"Chat: {st.session_state.chat_title}  •  saved automatically")
else:
    st.caption("New chat. Answers come only from the selected book(s).")

if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))

if not options:
    st.info("No book found yet. Upload a PDF in the sidebar, or run create_memory_for_llm.py.")
    st.stop()

for idx, msg in enumerate(st.session_state.messages):
    render_message(idx, msg)

# voice question (recorded in the sidebar)
voice_question = None
if audio is not None:
    with st.spinner("Transcribing your voice..."):
        try:
            voice_question = transcribe(audio.getvalue(), lang_code)
        except Exception as e:
            st.error(f"Voice transcription failed: {e}")
    st.session_state.audio_key += 1   # resets the recorder so it is not processed twice
    if voice_question is not None and not voice_question.strip():
        st.warning("I could not hear anything. Please try recording again.")
        voice_question = None

typed_question = st.chat_input("Ask your medical question here...")
question = typed_question or voice_question

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Searching the book(s)..."):
        try:
            db = get_db(selection, tuple(books))
            _, llm = get_models()
            answer, sources, search_query = answer_question(question, db, llm, language)
            reply = {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "search_query": search_query,
                "book": selection,
                "lang_code": lang_code,
            }
        except Exception as e:
            reply = {"role": "assistant", "content": f"Error: {e}", "error": True}

    st.session_state.messages.append(reply)
    persist()
    st.rerun()