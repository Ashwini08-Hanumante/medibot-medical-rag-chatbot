"""Feedback saving, passage highlighting and chat export for MediBot."""
import csv
import html
import re
from datetime import datetime
from pathlib import Path

FEEDBACK_FILE = Path("feedback.csv")

STOP_WORDS = {
    "what", "which", "that", "this", "with", "from", "have", "does", "about",
    "used", "uses", "using", "there", "their", "when", "where", "your", "into",
    "should", "would", "could", "tell", "give", "please", "good",
}


# ---------- feedback ----------
def save_feedback(question, answer, rating, book, pages):
    new_file = not FEEDBACK_FILE.exists()
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["time", "rating", "question", "answer", "book", "pages"])
        writer.writerow(
            [datetime.now().isoformat(timespec="seconds"), rating, question, answer, book, pages]
        )


# ---------- source highlighting ----------
def highlight_html(text, query):
    """Return the passage as HTML with the question's keywords highlighted."""
    text = re.sub(r"\s*\n\s*", " ", text)
    safe = html.escape(text, quote=False)
    words = {
        w for w in re.findall(r"[A-Za-z]{4,}", query.lower()) if w not in STOP_WORDS
    }
    if words:
        pattern = re.compile(
            r"\b(" + "|".join(sorted(map(re.escape, words), key=len, reverse=True)) + r")\w*",
            re.IGNORECASE,
        )
        safe = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", safe)
    return f"<div style='font-size:0.9rem;line-height:1.5'>{safe}</div>"


# ---------- chat export ----------
def chat_to_text(messages):
    lines = [f"MediBot chat export - {datetime.now():%Y-%m-%d %H:%M}", "=" * 50, ""]
    for m in messages:
        who = "You" if m["role"] == "user" else "MediBot"
        lines.append(f"{who}: {m['content']}")
        if m.get("sources"):
            pages = sorted({s["page"] for s in m["sources"]})
            lines.append("   (Sources: PDF pages " + ", ".join(map(str, pages)) + ")")
        lines.append("")
    return "\n".join(lines)


def _latin(text):
    """The built-in PDF font only supports Latin characters."""
    replacements = {
        "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2022": "-", "\u2026": "...", "**": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def chat_to_pdf(messages):
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 15)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 10, "MediBot chat export")
    pdf.ln(2)

    for m in messages:
        who = "You" if m["role"] == "user" else "MediBot"
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 6, who)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 6, _latin(m["content"]))
        pdf.ln(3)

    return bytes(pdf.output())