"""
The Quran Recitation & Evaluation App — Streamlit edition (Branded UI).

Architecture (fully local, no paid APIs):
  Browser mic  →  st.audio_input  →  local Whisper (Quranic fine-tune)
  →  Uthmani normalisation  →  dynamic window alignment
  →  word-level grading  →  Streamlit UI
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import difflib
import io
import random
import re
import unicodedata

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import soundfile as sf
import streamlit as st
import torch
from transformers import pipeline

# ── local ───────────────────────────────────────────────────────────────────
from quran_data import load_quran
import db

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
QURANJSON_ROOT_DIR = "quran_repo/json/surahs"

ASR_MODEL_ID = "MaddoggProduction/whisper-l-v3-turbo-quran-lora-dataset-mix"

SAMPLE_RATE          = 16_000
CHUNK_LENGTH_S       = 30
STRIDE_LENGTH_S      = 5
PASS_THRESHOLD       = 0.82
CLOSE_WORD_THRESHOLD = 0.60
WINDOW_LOOKAHEAD     = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VERSES_PER_PAGE = 20

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL BRAND STYLES (Option A Integration)
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Quran Recitation Tester",
    page_icon="📖",
    layout="centered",
)

db.init_db()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;600&family=Inter:wght@400;500;600&family=Amiri:wght@400;700&display=swap');

/* Page background */
.stApp {
    background-color: #0B1120;
    color: #F1EDE4;
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #080E1B;
    border-right: 1px solid rgba(201, 146, 63, 0.14);
}

/* Cards / containers */
[data-testid="stVerticalBlock"] > div {
    background-color: #111827;
    border-radius: 12px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #C9923F, #E8C06A);
    color: #0B1120;
    border: none;
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
}
.stButton > button:hover {
    opacity: 0.9;
}

/* Headings */
h1, h2, h3 {
    font-family: 'Lora', serif;
    color: #F1EDE4;
}

/* Arabic text */
[dir="rtl"], .arabic-text {
    font-size: 1.8rem;
    line-height: 2.2;
    color: #F1EDE4;
    font-family: 'Amiri', serif;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #161F32;
    border: 1px solid rgba(201, 146, 63, 0.14);
    border-radius: 12px;
    padding: 12px;
}
[data-testid="stMetricValue"] {
    color: #C9923F;
}

/* Progress bars */
.stProgress > div > div {
    background-color: #C9923F;
}

/* Mistake highlight */
.mistake {
    background-color: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 10px;
    padding: 12px;
}

.verdict-pass { color: #7fae8c; font-family: 'Lora', serif; font-size: 1.1rem; }
.verdict-fail { color: #ef4444; font-family: 'Lora', serif; font-size: 1.1rem; }

.result-panel {
    border-left: 3px solid #C9923F;
    padding: 12px 16px;
    margin: 10px 0;
    background: #111827;
    border-radius: 0 12px 12px 0;
}

.verse-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 28px;
    height: 28px;
    padding: 0 6px;
    border: 1.5px solid #C9923F;
    border-radius: 50%;
    color: #C9923F;
    font-family: 'Lora', serif;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 8px;
    vertical-align: middle;
}

.surah-header {
    text-align: center;
    padding: 18px 0 22px;
    border-bottom: 2px solid rgba(201, 146, 63, 0.2);
    margin-bottom: 12px;
}
.surah-header .arabic-name {
    font-family: 'Amiri', serif;
    font-size: 2.2rem;
    color: #C9923F;
}
.surah-header .meta {
    font-size: 0.75rem;
    color: #8b95a3;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# BOOKMARKS & HELPERS
# ════════════════════════════════════════════════════════════════════════════

def load_bookmarks() -> list:
    return db.load_bookmarks(st.session_state.user_id)

def add_bookmark(surah: int, ayah: int, surah_name: str, note: str = "") -> None:
    db.add_bookmark(st.session_state.user_id, surah, ayah, surah_name, note)

def remove_bookmark(surah: int, ayah: int) -> None:
    db.remove_bookmark(st.session_state.user_id, surah, ayah)

def is_bookmarked(surah: int, ayah: int, bookmarks: list) -> bool:
    return db.is_bookmarked(surah, ayah, bookmarks)

_CONTROL_TOKEN_RE = re.compile(r"<\|[^<>|]*\|>")

def clean_asr_output(raw_text: str) -> str:
    if not raw_text:
        return ""
    cleaned = _CONTROL_TOKEN_RE.sub("", raw_text)
    cleaned = cleaned.replace("\ufeff", "").replace("\u200e", "").replace("\u200f", "")
    return " ".join(cleaned.split()).strip()

_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_ALEF_VARIANTS = re.compile(r"[أإآٱ]")
_ALIF_MAQSURA = re.compile(r"ى")
_TEH_MARBUTA = re.compile(r"ة")

def normalize_arabic_text(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text)
    t = _DIACRITICS.sub("", t)
    t = _ALEF_VARIANTS.sub("ا", t)
    t = _ALIF_MAQSURA.sub("ي", t)
    t = _TEH_MARBUTA.sub("ه", t)
    return " ".join(t.split())

normalise = normalize_arabic_text

@st.cache_resource(show_spinner="Loading Quranic ASR model — first run downloads ~800 MB …")
def load_asr_pipeline():
    return pipeline(
        task="automatic-speech-recognition",
        model=ASR_MODEL_ID,
        device=DEVICE,
        return_timestamps=False,
        generate_kwargs={"language": "ar", "task": "transcribe"},
    )

@st.cache_resource(show_spinner="Loading Quran text …")
def load_quran_data():
    return load_quran(QURANJSON_ROOT_DIR)

def transcribe_audio(audio_bytes: bytes, asr) -> str:
    audio_buf = io.BytesIO(audio_bytes)
    audio_array, original_sr = sf.read(audio_buf, dtype="float32")
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    if original_sr != SAMPLE_RATE:
        try:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=original_sr, target_sr=SAMPLE_RATE)
        except ImportError:
            ratio = SAMPLE_RATE / original_sr
            new_len = int(len(audio_array) * ratio)
            indices = np.linspace(0, len(audio_array) - 1, new_len)
            audio_array = np.interp(indices, np.arange(len(audio_array)), audio_array)

    result = asr(
        {"array": audio_array, "sampling_rate": SAMPLE_RATE},
        chunk_length_s=CHUNK_LENGTH_S,
        stride_length_s=STRIDE_LENGTH_S,
        batch_size=1,
    )
    return result["text"].strip()

def build_reference_window(sequence: list, position: int, max_ayahs: int) -> tuple:
    window_words, word_owner = [], []
    end = min(position + max_ayahs, len(sequence))
    for idx in range(position, end):
        surah_num, ayah_num, text = sequence[idx]
        for w in text.split():
            window_words.append(w)
            word_owner.append((surah_num, ayah_num, idx))
    return window_words, word_owner

def _align_window(window_words: list, transcribed_words: list) -> tuple:
    norm_ref   = [normalize_arabic_text(w) for w in window_words]
    norm_trans = [normalize_arabic_text(w) for w in transcribed_words]
    matcher = difflib.SequenceMatcher(None, norm_ref, norm_trans, autojunk=False)
    word_results = [None] * len(window_words)
    last_hit = -1

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for i in range(i1, i2):
                word_results[i] = "correct"
            if i2 > i1:
                last_hit = max(last_hit, i2 - 1)
        elif tag == "replace":
            ref_n, trans_n = i2 - i1, j2 - j1
            for offset in range(ref_n):
                i = i1 + offset
                if offset < trans_n:
                    j = j1 + offset
                    ratio = difflib.SequenceMatcher(None, norm_ref[i], norm_trans[j], autojunk=False).ratio()
                    word_results[i] = "close" if ratio >= CLOSE_WORD_THRESHOLD else "wrong"
                else:
                    word_results[i] = "wrong"
            if ref_n > 0:
                last_hit = max(last_hit, i2 - 1)
        elif tag == "delete":
            for i in range(i1, i2):
                word_results[i] = "__delete__"
    return word_results, last_hit

def grade_continuous_recitation(transcribed: str, sequence: list, position: int) -> tuple:
    transcribed = clean_asr_output(transcribed)
    if not transcribed:
        return {"graded": False, "reason": "Nothing transcribed.", "transcribed": transcribed}, position

    transcribed_words = transcribed.split()
    max_ayah_count = min(WINDOW_LOOKAHEAD, len(sequence) - position)
    if max_ayah_count <= 0:
        return {"graded": False, "reason": "No reference text remaining."}, position

    ayah_count = 1
    window_words, word_owner, word_results, last_hit = [], [], [], -1

    while True:
        window_words, word_owner = build_reference_window(sequence, position, ayah_count)
        word_results, last_hit = _align_window(window_words, transcribed_words)
        reached_edge = last_hit >= len(window_words) - 1
        can_grow = ayah_count < max_ayah_count
        if reached_edge and can_grow:
            ayah_count = min(max_ayah_count, ayah_count * 2 + 2)
            continue
        break

    current_hit = last_hit
    while current_hit >= 0:
        while current_hit >= 0 and word_results[current_hit] not in ("correct", "close"):
            current_hit -= 1
        if current_hit < 0:
            break
        prev_hit = current_hit - 1
        while prev_hit >= 0 and word_results[prev_hit] not in ("correct", "close"):
            prev_hit -= 1
        gap = (current_hit - prev_hit) - 1
        if gap >= 6:
            current_hit = prev_hit
        else:
            break
    last_hit = current_hit

    if last_hit == -1:
        return {"graded": False, "reason": "Could not align.", "transcribed": transcribed}, position

    for i in range(len(word_results)):
        if word_results[i] == "__delete__":
            word_results[i] = "missing" if i < last_hit else "not_recited"
        elif word_results[i] is None:
            word_results[i] = "not_recited"

    ayah_map: dict = {}
    for i, (snum, anum, sidx) in enumerate(word_owner):
        ayah_map.setdefault(sidx, {"surah": snum, "ayah": anum, "words": []})
        ayah_map[sidx]["words"].append({"text": window_words[i], "status": word_results[i]})

    last_seq_idx = word_owner[last_hit][2]
    new_position = last_seq_idx + 1
    graded_ayahs = [ayah_map[idx] for idx in sorted(ayah_map) if idx <= last_seq_idx]

    graded_statuses = word_results[: last_hit + 1]
    matched = sum(1 for s in graded_statuses if s in ("correct", "close"))
    similarity = matched / len(graded_statuses) if graded_statuses else 0.0

    return {
        "graded": True,
        "transcribed": transcribed,
        "start_surah": sequence[position][0],
        "start_ayah": sequence[position][1],
        "end_surah": sequence[last_seq_idx][0],
        "end_ayah": sequence[last_seq_idx][1],
        "ayahs": graded_ayahs,
        "similarity": similarity,
        "passed": similarity >= PASS_THRESHOLD,
    }, new_position

def build_sequence(quran: dict, from_surah: int, to_surah: int) -> list:
    lo, hi = min(from_surah, to_surah), max(from_surah, to_surah)
    seq = []
    for s in sorted(quran):
        if lo <= s <= hi:
            for a in sorted(quran[s]["verses"]):
                seq.append((s, a, quran[s]["verses"][a]))
    return seq

def render_word(word: str, status: str) -> str:
    color_map = {
        "correct": "#7fae8c",
        "close": "#C9923F",
        "wrong": "#ef4444",
        "missing": "#888",
        "not_recited": "#444"
    }
    colour = color_map.get(status, "#F1EDE4")
    style = f"color:{colour};padding:0 2px;font-family:'Amiri',serif;font-size:1.5rem;"
    if status == "wrong":
        style += "text-decoration:underline wavy #ef4444;"
    elif status == "close":
        style += "text-decoration:underline dotted #C9923F;"
    return f'<span style="{style}" title="{status}">{word}</span>'

def render_ayah_block(ayah: dict) -> str:
    words_html = " ".join(render_word(w["text"], w["status"]) for w in ayah["words"])
    header = f'<div style="font-size:0.7rem;color:#8b95a3;text-transform:uppercase;margin-bottom:4px;">Surah {ayah["surah"]}, Ayah {ayah["ayah"]}</div>'
    body = f'<div style="direction:rtl;text-align:right;padding:10px 12px;background:#161F32;border:1px solid rgba(201,146,63,0.14);border-radius:8px;">{words_html}</div>'
    return header + body

def legend_html() -> str:
    return '<div style="margin-top:8px;font-size:0.75rem;color:#8b95a3;"><span style="color:#7fae8c">✓ Correct</span> &nbsp; <span style="color:#C9923F">~ Close</span> &nbsp; <span style="color:#ef4444">✗ Wrong</span></div>'

# ════════════════════════════════════════════════════════════════════════════
# AUTH GATE
# ════════════════════════════════════════════════════════════════════════════

if "user_id" not in st.session_state:
    st.session_state.user_id = None
    st.session_state.username = None

if st.session_state.user_id is None:
    st.markdown("## 📖 Quran App")
    login_tab, register_tab = st.tabs(["Log in", "Create account"])

    with login_tab:
        with st.form("login_form"):
            login_username = st.text_input("Username")
            login_password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", use_container_width=True)
        if submitted:
            uid = db.authenticate_user(login_username, login_password)
            if uid is not None:
                st.session_state.user_id = uid
                st.session_state.username = login_username.strip()
                st.rerun()
            else:
                st.error("Incorrect username or password.")

    with register_tab:
        with st.form("register_form"):
            new_username = st.text_input("Choose a username")
            new_password = st.text_input("Choose a password", type="password")
            new_password_confirm = st.text_input("Confirm password", type="password")
            register_submitted = st.form_submit_button("Create account", use_container_width=True)
        if register_submitted:
            if new_password != new_password_confirm:
                st.error("Passwords don't match.")
            else:
                new_uid = db.create_user(new_username, new_password)
                if new_uid is None:
                    st.error("Username taken.")
                else:
                    st.session_state.user_id = new_uid
                    st.session_state.username = new_username.strip()
                    st.rerun()
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION & NAVIGATION
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"Signed in as **{st.session_state.username}**")
    if st.button("Log out", use_container_width=True):
        st.session_state.user_id = None
        st.session_state.username = None
        st.rerun()
    st.markdown("---")

quran = load_quran_data()
surah_numbers = sorted(quran.keys())
surah_options = {f"{n}. {quran[n]['name']}": n for n in surah_numbers}

defaults = {
    "sequence": [], "position": 0, "results": [], "session_live": False,
    "app_mode": "📖 Read Quran", "read_surah": surah_numbers[0], "read_page": 0,
    "hifz_focus": False, "jump_from_surah": None, "exam_live": False,
    "exam_questions": [], "exam_q_idx": 0, "exam_scores": [], "exam_details": [],
    "review_idx": 0, "review_audio_key": 0, "active_class_id": None
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

_MODE_OPTIONS = ["📖 Read Quran", "🔁 Today's Review", "🎙 Recite & Test", "📝 Exam Simulator", "🎓 Teacher"]

with st.sidebar:
    st.markdown("### Mode")
    due_count = db.get_review_queue_size(st.session_state.user_id)
    mode_labels = list(_MODE_OPTIONS)
    if due_count:
        mode_labels[1] = f"🔁 Today's Review ({due_count})"
    mode = st.radio("Mode", mode_labels, index=_MODE_OPTIONS.index(st.session_state.app_mode), label_visibility="collapsed")
    mode = _MODE_OPTIONS[mode_labels.index(mode)]
    if mode != st.session_state.app_mode:
        st.session_state.app_mode = mode
        st.rerun()
    st.markdown("---")

# Render active mode content
if st.session_state.app_mode == "📖 Read Quran":
    bookmarks = load_bookmarks()
    with st.sidebar:
        read_label = st.selectbox("Surah", list(surah_options.keys()), index=surah_numbers.index(st.session_state.read_surah))
        selected_surah = surah_options[read_label]
        if selected_surah != st.session_state.read_surah:
            st.session_state.read_surah = selected_surah
            st.session_state.read_page = 0
            st.rerun()

    surah_num = st.session_state.read_surah
    surah_name = quran[surah_num]["name"]
    verse_nums = sorted(quran[surah_num]["verses"].keys())
    total_verses = len(verse_nums)

    st.markdown(f'<div class="surah-header"><div class="arabic-name">{surah_name}</div><div class="meta">Surah {surah_num} · {total_verses} verses</div></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎙 Practice Surah", use_container_width=True):
            st.session_state.jump_from_surah = surah_num
            st.session_state.app_mode = "🎙 Recite & Test"
            st.rerun()
    with col2:
        if st.button("📝 Take Exam", use_container_width=True):
            st.session_state.jump_from_surah = surah_num
            st.session_state.app_mode = "📝 Exam Simulator"
            st.rerun()

    total_pages = max(1, (total_verses + VERSES_PER_PAGE - 1) // VERSES_PER_PAGE)
    page = min(st.session_state.read_page, total_pages - 1)
    for ayah_num in verse_nums[page * VERSES_PER_PAGE : (page + 1) * VERSES_PER_PAGE]:
        text = quran[surah_num]["verses"][ayah_num]
        col_t, col_b = st.columns([9, 1])
        with col_t:
            st.markdown(f'<div style="direction:rtl;text-align:right;font-family:\'Amiri\',serif;font-size:1.85rem;line-height:2.4;padding:10px 4px;"><span class="verse-badge">{ayah_num}</span>{text}</div>', unsafe_allow_html=True)
        with col_b:
            bookmarked = is_bookmarked(surah_num, ayah_num, bookmarks)
            if st.button("🔖" if bookmarked else "🏷️", key=f"bm_{surah_num}_{ayah_num}"):
                if bookmarked: remove_bookmark(surah_num, ayah_num)
                else: add_bookmark(surah_num, ayah_num, surah_name)
                st.rerun()

elif st.session_state.app_mode == "🎙 Recite & Test":
    asr = load_asr_pipeline()
    with st.sidebar:
        st.markdown("### Range Setup")
        from_label = st.selectbox("From surah", list(surah_options.keys()), disabled=st.session_state.session_live)
        to_label = st.selectbox("To surah", list(surah_options.keys()), disabled=st.session_state.session_live)
        from_surah, to_surah = surah_options[from_label], surah_options[to_label]
        
        st.session_state.hifz_focus = st.toggle("🧠 Hifz Focus Mode", value=st.session_state.hifz_focus, disabled=st.session_state.session_live)

        if not st.session_state.session_live:
            if st.button("▶ Start session", use_container_width=True):
                seq = build_sequence(quran, from_surah, to_surah)
                if seq:
                    st.session_state.sequence = seq
                    st.session_state.position = random.randint(0, len(seq) - 1)
                    st.session_state.results = []
                    st.session_state.session_live = True
                    st.rerun()
        else:
            if st.button("⏹ End session", use_container_width=True):
                st.session_state.session_live = False
                st.rerun()

    if not st.session_state.session_live:
        st.info("Select a range in the sidebar and press **▶ Start session**.")
    else:
        seq, pos = st.session_state.sequence, st.session_state.position
        if pos >= len(seq):
            st.success("🎉 Completed range!")
            st.session_state.session_live = False
        else:
            cur_s, cur_a, cur_text = seq[pos]
            st.markdown(f"**Surah {cur_s}, Ayah {cur_a}**")
            if not st.session_state.hifz_focus:
                st.markdown(f'<div class="result-panel" style="direction:rtl;text-align:right;font-family:\'Amiri\',serif;font-size:1.6rem;">{" ".join(cur_text.split()[:3])} …</div>', unsafe_allow_html=True)
            
            audio_value = st.audio_input("Record your recitation", key=f"audio_{pos}")
            if audio_value is not None:
                with st.spinner("Transcribing & grading..."):
                    transcription = transcribe_audio(audio_value.read(), asr)
                    result, new_pos = grade_continuous_recitation(transcription, seq, pos)
                if result["graded"]:
                    st.session_state.results.append(result)
                    st.session_state.position = new_pos
                    db.save_session_result(st.session_state.user_id, result)
                    st.rerun()
                else:
                    st.warning(result.get("reason", "Could not align."))

else:
    st.markdown(f"## {st.session_state.app_mode}")
    st.info("Switch modes or use the sidebar options to proceed.")
