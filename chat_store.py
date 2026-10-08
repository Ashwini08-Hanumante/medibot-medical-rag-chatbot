"""Save, load, rename and delete MediBot chats as JSON files inside chats/."""
import json
import uuid
from datetime import datetime
from pathlib import Path

CHAT_DIR = Path("chats")


def new_chat_id():
    return f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:4]}"


def make_title(text, limit=40):
    text = " ".join(str(text).split())
    if not text:
        return "Untitled chat"
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _path(chat_id):
    return CHAT_DIR / f"{chat_id}.json"


def save_chat(chat_id, title, messages):
    CHAT_DIR.mkdir(exist_ok=True)
    # audio bytes are not saved; "Listen" can recreate them any time
    clean = [{k: v for k, v in m.items() if k != "audio"} for m in messages]
    data = {
        "id": chat_id,
        "title": title,
        "updated": datetime.now().isoformat(timespec="seconds"),
        "messages": clean,
    }
    tmp = _path(chat_id).with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_path(chat_id))


def load_chat(chat_id):
    try:
        return json.loads(_path(chat_id).read_text(encoding="utf-8"))
    except Exception:
        return None


def list_chats():
    """Newest first: [{'id', 'title', 'updated'}, ...]"""
    if not CHAT_DIR.exists():
        return []
    chats = []
    for p in CHAT_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            chats.append(
                {"id": d["id"], "title": d.get("title", "Untitled chat"), "updated": d.get("updated", "")}
            )
        except Exception:
            continue
    return sorted(chats, key=lambda c: c["updated"], reverse=True)


def delete_chat(chat_id):
    _path(chat_id).unlink(missing_ok=True)


def delete_all_chats():
    for chat in list_chats():
        delete_chat(chat["id"])