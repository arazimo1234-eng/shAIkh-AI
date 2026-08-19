"""
The Quran Recitation & Evaluation App — Streamlit edition.

Architecture (fully local, no paid APIs):
  Browser mic  →  st.audio_input  →  local Whisper (Quranic fine-tune)
  →  Uthmani normalisation  →  dynamic window alignment
  →  word-level grading  →  Streamlit UI

Persistence (Stage 1): accounts, bookmarks, and every graded session are
stored in a local SQLite database (see db.py) — not a shared flat file and
not st.session_state alone, so data survives both a page refresh and more
than one concurrent user. See db.py's module docstring for schema details.

Model: MaddoggProduction/whisper-l-v3-turbo-quran-lora-dataset-mix
  • Fine-tuned on tarteel-ai/everyayah + MohamedRashad/Quran-Recitations
  • Outputs Uthmani-diacritised Arabic (tashkeel preserved)
  • License: Apache 2.0 — commercial use permitted
  • Base: openai/whisper-large-v3-turbo

Fallback: tarteel-ai/whisper-base-ar-quran (lighter, ~150 MB)
  Switch by setting ASR_MODEL_ID below if RAM is tight.

Usage:
  pip install streamlit transformers torch numpy soundfile
  streamlit run streamlit_app.py
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
# CONFIGURATION  (edit these, nothing else needs changing)
# ════════════════════════════════════════════════════════════════════════════
QURANJSON_ROOT_DIR = "quran_repo/json/surahs"  # matches download_quran_data.py's OUT_DIR exactly

# Primary model: large-v3-turbo fine-tune — outputs with proper tashkeel.
# ~800 MB download on first run; needs ~2 GB RAM.
# Swap to the line below if your Chromebook runs out of memory:
#   ASR_MODEL_ID = "tarteel-ai/whisper-base-ar-quran"
ASR_MODEL_ID = "MaddoggProduction/whisper-l-v3-turbo-quran-lora-dataset-mix"

SAMPLE_RATE          = 16_000   # Whisper expects 16 kHz mono
CHUNK_LENGTH_S       = 30       # Whisper's native context window
STRIDE_LENGTH_S      = 5        # overlap between chunks (prevents boundary drops)
PASS_THRESHOLD       = 0.82     # overall similarity to count a segment as "correct"
CLOSE_WORD_THRESHOLD = 0.60     # per-word similarity: below → wrong, above → close
WINDOW_LOOKAHEAD     = 20       # ayahs ahead to include in the reference window

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VERSES_PER_PAGE = 20   # reader pagination, keeps long surahs (e.g. Al-Baqarah, 286 ayat) responsive

# ════════════════════════════════════════════════════════════════════════════
# BOOKMARKS  (Stage 1 fix: SQLite-backed, scoped to the logged-in user.
# The old version wrote every user's bookmarks into one shared bookmarks.json
# file, so two concurrent users silently overwrote each other's data. These
# thin wrappers keep the same call shape the rest of the file already uses,
# but every call is now scoped to st.session_state.user_id — see db.py for
# the actual persistence logic.)
# ════════════════════════════════════════════════════════════════════════════

def load_bookmarks() -> list:
    return db.load_bookmarks(st.session_state.user_id)


def add_bookmark(surah: int, ayah: int, surah_name: str, note: str = "") -> None:
    db.add_bookmark(st.session_state.user_id, surah, ayah, surah_name, note)


def remove_bookmark(surah: int, ayah: int) -> None:
    db.remove_bookmark(st.session_state.user_id, surah, ayah)


def is_bookmarked(surah: int, ayah: int, bookmarks: list) -> bool:
    return db.is_bookmarked(surah, ayah, bookmarks)


# ════════════════════════════════════════════════════════════════════════════
# ASR OUTPUT CLEANING
# Must run FIRST, before anything else touches the transcription string.
# ════════════════════════════════════════════════════════════════════════════

# Matches any Whisper-style control/metadata token: <|ar|>, <|transcribe|>,
# <|notimestamps|>, <|0.00|>, <|endoftext|>, etc. These are meant to be
# consumed by the tokenizer's skip_special_tokens logic, but LoRA fine-tunes
# frequently add new special tokens that the base processor doesn't know to
# suppress — so they leak into result["text"] as literal characters. Because
# they're emitted with no space before the first real word, they glue onto
# it (e.g. "<|ar|><|transcribe|><|notimestamps|>بِسْمِ") and that combined
# token fails to match anything in the reference window, which used to get
# scored as a wrong first word.
_CONTROL_TOKEN_RE = re.compile(r"<\|[^<>|]*\|>")


def clean_asr_output(raw_text: str) -> str:
    """
    Strip Whisper control/metadata tags from raw ASR output.

    Removes every substring wrapped in `<|` and `|>` (language tag, task
    tag, timestamp tags, etc.), wherever they occur in the string — not
    just at the start, since some checkpoints emit timestamp tokens
    between segments too. Collapses any whitespace left behind by the
    removal so word-splitting downstream doesn't produce empty tokens.

    This is purely mechanical text cleanup — it does not touch Arabic
    characters, diacritics, or punctuation, and it is safe to call on
    already-clean text (idempotent).
    """
    if not raw_text:
        return ""
    cleaned = _CONTROL_TOKEN_RE.sub("", raw_text)
    # Defensive: also drop stray BOM / RTL-mark / LTR-mark characters that
    # some browser MediaRecorder → soundfile → Whisper paths introduce.
    cleaned = cleaned.replace("\ufeff", "").replace("\u200e", "").replace("\u200f", "")
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


# ════════════════════════════════════════════════════════════════════════════
# ARABIC NORMALISATION
# Two layers, kept strictly separate:
#   display_text  — the canonical Uthmani text shown to the user, NEVER mutated
#   norm(text)    — stripped skeleton used only for comparison
# ════════════════════════════════════════════════════════════════════════════

# All Unicode combining marks used in Uthmani Arabic (harakat, Quranic
# annotation symbols, tatweel, etc.)
_DIACRITICS = re.compile(
    r"[\u0610-\u061A"   # Arabic extended (Quranic annotation marks)
    r"\u064B-\u065F"   # Harakat (fathah … sukun)
    r"\u0670"          # Superscript alef (alef khanjariyya)
    r"\u06D6-\u06ED"   # Quranic annotation marks
    r"\u0640"          # Tatweel (kashida)
    r"]"
)

# Alef variants (with hamza above/below, with madda, with wasla) → plain alef
_ALEF_VARIANTS = re.compile(r"[أإآٱ]")

# Alif maqsura (looks like ya without dots) → ya, because some ASR models
# output ya where the reference has alif maqsura and vice-versa
_ALIF_MAQSURA = re.compile(r"ى")

# Teh marbuta → ha, to smooth over common transcription variants
_TEH_MARBUTA = re.compile(r"ة")


def normalize_arabic_text(text: str) -> str:
    """
    Produce a comparison skeleton from Arabic text.
    ONLY used for difflib matching — NEVER shown to the user and NEVER
    written back over the canonical Uthmani display_text.

    This is intentionally a one-way, lossy transform. Its job is to make
    two Uthmani-correct spellings that differ only in orthographic
    convention (not in which word they are) compare as equal, so that
    real mistakes aren't lost among false positives caused by encoding
    variance.

    Steps applied (in order):
      1. NFC normalise first (Unicode canonical composition) — some ASR
         checkpoints emit alef+hamza as two separate NFD codepoints
         (0627 + 0654) instead of the precomposed form (0623); collapsing
         to NFC up front means the diacritic/alef regexes below always see
         the composed forms they expect, rather than silently missing
         decomposed variants.
      2. Strip all diacritics and Quranic annotation marks (harakat,
         sukun in any of its encodings, Quranic pause/annotation symbols,
         tatweel/kashida elongation)
      3. Normalise alef variants (hamza-above/below, madda, wasla) → bare
         alef, since ASR models are inconsistent about which hamza seat
         they predict
      4. Normalise alif maqsura → ya (some checkpoints output ya where
         the Uthmani reference has alif maqsura, and vice versa)
      5. Normalise teh marbuta → ha (common ASR transcription variant)
      6. Collapse whitespace

    Returns the skeleton string. The original `text` argument is never
    mutated — normalize_arabic_text always returns a new string.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text)
    t = _DIACRITICS.sub("", t)
    t = _ALEF_VARIANTS.sub("ا", t)
    t = _ALIF_MAQSURA.sub("ي", t)
    t = _TEH_MARBUTA.sub("ه", t)
    t = " ".join(t.split())
    return t


# Backward-compatible alias — kept in case other modules in the project
# still import the old name.
normalise = normalize_arabic_text


# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADING  (cached so it only happens once per Streamlit session)
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading Quranic ASR model — first run downloads ~800 MB …")
def load_asr_pipeline():
    """
    Loads the Quranic Whisper model as a Hugging Face ASR pipeline.

    Using pipeline() (rather than raw WhisperForConditionalGeneration) because
    it handles chunked long-form audio natively via chunk_length_s /
    stride_length_s — preventing the silent 30-second truncation that happens
    when you feed raw audio directly to the feature extractor.
    """
    asr = pipeline(
        task="automatic-speech-recognition",
        model=ASR_MODEL_ID,
        device=DEVICE,
        # Tell the pipeline to emit word timestamps so we have alignment hooks
        # if we want to use them in future phases.
        return_timestamps=False,
        generate_kwargs={"language": "ar", "task": "transcribe"},
    )
    return asr


@st.cache_resource(show_spinner="Loading Quran text …")
def load_quran_data():
    return load_quran(QURANJSON_ROOT_DIR)


# ════════════════════════════════════════════════════════════════════════════
# TRANSCRIPTION
# ════════════════════════════════════════════════════════════════════════════

def transcribe_audio(audio_bytes: bytes, asr) -> str:
    """
    Decodes raw audio bytes (from st.audio_input), resamples if needed,
    and transcribes using the Quranic ASR pipeline with chunking.

    chunk_length_s=30 / stride_length_s=5:
      • Each chunk fits exactly inside Whisper's 30-second context window.
      • The 5-second overlap between chunks prevents words from being dropped
        at chunk boundaries (a common cause of false alignment failures).
    """
    audio_buf = io.BytesIO(audio_bytes)
    audio_array, original_sr = sf.read(audio_buf, dtype="float32")

    # Ensure mono
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    # Resample to 16 kHz if the browser delivered a different rate
    if original_sr != SAMPLE_RATE:
        try:
            import librosa
            audio_array = librosa.resample(
                audio_array, orig_sr=original_sr, target_sr=SAMPLE_RATE
            )
        except ImportError:
            # Naive linear resample fallback if librosa isn't installed
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


# ════════════════════════════════════════════════════════════════════════════
# ALIGNMENT & GRADING
# ════════════════════════════════════════════════════════════════════════════

def build_reference_window(sequence: list, position: int, max_ayahs: int) -> tuple:
    """
    Builds a flat word list from sequence[position : position + max_ayahs],
    together with a per-word owner index so we know which ayah each word came from.

    max_ayahs is caller-controlled (not always WINDOW_LOOKAHEAD) so the grader
    can start small and only widen the window when the recitation actually
    runs up against its edge — see grade_continuous_recitation for why this
    matters.

    Returns:
        window_words  — list of display-text words (Uthmani, never mutated)
        word_owner    — parallel list of (surah, ayah, seq_idx) per word
    """
    window_words, word_owner = [], []
    end = min(position + max_ayahs, len(sequence))
    for idx in range(position, end):
        surah_num, ayah_num, text = sequence[idx]
        for w in text.split():
            window_words.append(w)
            word_owner.append((surah_num, ayah_num, idx))
    return window_words, word_owner


def _align_window(window_words: list, transcribed_words: list) -> tuple:
    """
    Runs one difflib alignment pass of transcribed_words against window_words
    and returns (word_results, last_hit). Pulled out of grade_continuous_recitation
    so it can be re-run against progressively larger windows without duplicating
    the opcode-resolution logic.
    """
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
                    ratio = difflib.SequenceMatcher(
                        None, norm_ref[i], norm_trans[j], autojunk=False
                    ).ratio()
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
    """
    Sliding-window sequence grader for continuous, multi-verse recitation.

    A user may keep reciting past the single ayah they were prompted with,
    so `transcribed` can legitimately span many verses in one go. This
    function does NOT try to grade one ayah at a time — it:

      1. Builds a flat, ordered word list ("reference window") starting at
         `position`, SIZED TO ROUGHLY HOW MUCH WAS ACTUALLY TRANSCRIBED —
         not a fixed 20-ayah block. This is deliberate: Quranic Arabic
         repeats short function words (و، من، لا، في …) constantly, and
         difflib.SequenceMatcher (with autojunk=False, which is needed so
         those common words aren't ignored entirely) will happily chain a
         match onto one of them several ayahs away by pure coincidence if
         given a large enough haystack. Handing it a 20-ayah window for a
         one-ayah recitation let exactly that happen: a stray match a few
         ayahs ahead pulled `last_hit` far past where the user actually
         stopped, and everything in between got graded "wrong"/"missing"
         instead of being left alone.
      2. Only WIDENS the window (and re-aligns) if the match runs right up
         against the current window's edge — that's the signal the user
         kept reciting past what we gave the matcher room for, not that a
         distant coincidental word match should be trusted. Widening stops
         at WINDOW_LOOKAHEAD ayahs.
      3. Finds the rightmost reference word actually reached
         (`last_hit`) and grades only up to that point — anything beyond
         is "not_recited" (unattempted), not "wrong". Session position
         only ever advances to just past `last_hit`.

    Key invariant:
        Words after the detected stop point are marked "not_recited", NOT
        wrong. The caller must NOT advance `position` if graded=False.

    Args:
        transcribed: raw (or already-cleaned) ASR output for this take.
                     clean_asr_output() is applied internally regardless,
                     so control tokens can never leak into alignment even
                     if a caller forgets to pre-clean.
        sequence:    ordered list of (surah_num, ayah_num, uthmani_text)
                     tuples — the full target range for the session.
        position:    index into `sequence` of the next ayah the user is
                     expected to recite.

    Returns: (result_dict, new_position)
    """
    transcribed = clean_asr_output(transcribed)

    if not transcribed:
        return {
            "graded": False,
            "reason": "Nothing was transcribed after removing model metadata tags.",
            "transcribed": transcribed,
        }, position

    transcribed_words = transcribed.split()
    max_ayah_count = min(WINDOW_LOOKAHEAD, len(sequence) - position)
    if max_ayah_count <= 0:
        return {"graded": False, "reason": "No reference text remaining."}, position

    # Start small: enough ayahs to plausibly hold ~2x the transcribed word
    # count (generous room for insertions/substitutions), at least 1 ayah.
    # Widen only if the alignment actually reaches this window's edge.
    ayah_count = 1
    window_words, word_owner, word_results, last_hit = [], [], [], -1

    while True:
        window_words, word_owner = build_reference_window(sequence, position, ayah_count)
        word_results, last_hit = _align_window(window_words, transcribed_words)

        reached_edge = last_hit >= len(window_words) - 1
        can_grow = ayah_count < max_ayah_count
        if reached_edge and can_grow:
            # Grow by enough to plausibly cover the rest of what was said,
            # not just +1 ayah at a time (avoids re-aligning ayah-by-ayah
            # for a long multi-ayah recitation).
            ayah_count = min(max_ayah_count, ayah_count * 2 + 2)
            continue
        break

    # ── STRICT SAFETY GUARD: Trim trailing coincidental matches ──
    # difflib often maps common trailing particles (و, ف, من) to matches several 
    # ayahs ahead by pure chance. This creates a massive gap of "__delete__" 
    # statuses ending in a single stray "correct" match, dragging last_hit far 
    # past where you actually stopped speaking.
    
    current_hit = last_hit
    while current_hit >= 0:
        # 1. Find the last valid matched word (correct or close)
        while current_hit >= 0 and word_results[current_hit] not in ("correct", "close"):
            current_hit -= 1
            
        if current_hit < 0:
            break
            
        # 2. Find the matched word right before it
        prev_hit = current_hit - 1
        while prev_hit >= 0 and word_results[prev_hit] not in ("correct", "close"):
            prev_hit -= 1
            
        gap = (current_hit - prev_hit) - 1
        
        # 3. If there is a gap of 6 or more missed words between matches, it is 
        # almost certainly a false cross-ayah jump caused by SequenceMatcher 
        # (or you skipped a huge chunk of text, in which case we shouldn't 
        # advance your progress past it). We mathematically snip off 
        # the stray match by reeling current_hit backwards.
        if gap >= 6:
            current_hit = prev_hit
        else:
            break
            
    last_hit = current_hit
    # ─────────────────────────────────────────────────────────────────

    if last_hit == -1:
        # Nothing matched even in the fully-grown window — silence, noise,
        # or unrelated speech. Don't advance the session position.
        return {
            "graded": False,
            "reason": "Could not align with the expected passage. "
                      "Try reciting more clearly or adjusting the range.",
            "transcribed": transcribed,
        }, position

    # Resolve tentative "__delete__" marks
    for i in range(len(word_results)):
        if word_results[i] == "__delete__":
            # Before the stop-point: genuinely skipped mid-attempt.
            # At/after: simply not yet reached — leave as not_recited.
            word_results[i] = "missing" if i < last_hit else "not_recited"
        elif word_results[i] is None:
            word_results[i] = "not_recited"

    # Group words back by ayah
    ayah_map: dict = {}
    for i, (snum, anum, sidx) in enumerate(word_owner):
        ayah_map.setdefault(sidx, {"surah": snum, "ayah": anum, "words": []})
        ayah_map[sidx]["words"].append({"text": window_words[i], "status": word_results[i]})

    last_seq_idx  = word_owner[last_hit][2]
    new_position  = last_seq_idx + 1

    graded_ayahs = [
        ayah_map[idx]
        for idx in sorted(ayah_map)
        if idx <= last_seq_idx
    ]

    # Score only the graded portion (not not_recited words)
    graded_statuses = word_results[: last_hit + 1]
    matched = sum(1 for s in graded_statuses if s in ("correct", "close"))
    similarity = matched / len(graded_statuses) if graded_statuses else 0.0

    return {
        "graded":      True,
        "transcribed": transcribed,  # already cleaned of control tags
        "start_surah": sequence[position][0],
        "start_ayah":  sequence[position][1],
        "end_surah":   sequence[last_seq_idx][0],
        "end_ayah":    sequence[last_seq_idx][1],
        "ayahs":       graded_ayahs,
        "similarity":  similarity,
        "passed":      similarity >= PASS_THRESHOLD,
    }, new_position


# ════════════════════════════════════════════════════════════════════════════
# SEQUENCE BUILDER
# ════════════════════════════════════════════════════════════════════════════

def build_sequence(quran: dict, from_surah: int, to_surah: int) -> list:
    """Returns an ordered list of (surah_num, ayah_num, text) tuples."""
    lo, hi = min(from_surah, to_surah), max(from_surah, to_surah)
    seq = []
    for s in sorted(quran):
        if lo <= s <= hi:
            for a in sorted(quran[s]["verses"]):
                seq.append((s, a, quran[s]["verses"][a]))
    return seq


# ════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ════════════════════════════════════════════════════════════════════════════

# Status → (CSS colour, label)
STATUS_STYLE = {
    "correct":     ("#7fae8c", "✓"),
    "close":       ("#b8935a", "~"),
    "wrong":       ("#c17b7b", "✗"),
    "missing":     ("#666",    "–"),
    "not_recited": ("#444",    " "),
}

def render_word(word: str, status: str) -> str:
    colour, _ = STATUS_STYLE.get(status, ("#ece4d6", "?"))
    style = (
        f"color:{colour};"
        f"padding:0 2px;"
        f"border-radius:2px;"
        f"font-family:'Amiri',serif;"
        f"font-size:1.25rem;"
        f"line-height:2;"
    )
    if status == "wrong":
        style += "text-decoration:underline wavy #c17b7b;text-underline-offset:4px;"
    elif status == "close":
        style += "text-decoration:underline dotted #b8935a;text-underline-offset:4px;"
    elif status == "missing":
        style += "text-decoration:line-through;opacity:0.5;"
    elif status == "not_recited":
        style += "opacity:0.3;"
    return f'<span style="{style}" title="{status}">{word}</span>'


def render_ayah_block(ayah: dict) -> str:
    words_html = " ".join(
        render_word(w["text"], w["status"]) for w in ayah["words"]
    )
    header = (
        f'<div style="font-size:0.7rem;color:#8b95a3;'
        f'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px;">'
        f'Surah {ayah["surah"]}, Ayah {ayah["ayah"]}</div>'
    )
    body = (
        f'<div style="direction:rtl;text-align:right;'
        f'padding:10px 12px;background:#1a222c;'
        f'border:1px solid #2a3542;border-radius:3px;">'
        f'{words_html}</div>'
    )
    return header + body


def legend_html() -> str:
    items = [
        ("correct",     "#7fae8c", "Correct"),
        ("close",       "#b8935a", "Close (minor ASR variant)"),
        ("wrong",       "#c17b7b", "Wrong word"),
        ("missing",     "#888",    "Skipped"),
        ("not_recited", "#444",    "Not yet reached"),
    ]
    chips = " &nbsp; ".join(
        f'<span style="color:{c};font-size:0.75rem;">{label}</span>'
        for _, c, label in items
    )
    return f'<div style="margin-top:8px;">{chips}</div>'


# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL STYLES
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Quran Recitation Tester",
    page_icon="📖",
    layout="centered",
)

db.init_db()

st.markdown(
    """<link href="https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Spectral:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  /* Dark theme base */
  html, body, [data-testid="stAppViewContainer"] {
    background-color: #10161d;
    color: #ece4d6;
  }
  [data-testid="stSidebar"] { background-color: #141b23; }
  h1, h2, h3 { font-family: 'Spectral', serif; color: #b8935a; }
  p, label, div { color: #ece4d6; }
  /* Streamlit selectbox / widgets */
  .stSelectbox > div, .stSlider { color: #ece4d6; }
  .stButton > button {
    background: #b8935a;
    color: #1a1305;
    border: none;
    border-radius: 3px;
    font-weight: 600;
    padding: 0.5rem 1.4rem;
  }
  .stButton > button:hover { background: #c7a06a; }
  .verdict-pass { color: #7fae8c; font-family: 'Spectral', serif; font-size: 1.1rem; }
  .verdict-fail { color: #c17b7b; font-family: 'Spectral', serif; font-size: 1.1rem; }
  /* Gold left border on result panels */
  .result-panel {
    border-left: 3px solid #b8935a;
    padding: 12px 16px;
    margin: 10px 0;
    background: #141b23;
    border-radius: 0 3px 3px 0;
  }
  /* Read Quran mode */
  .verse-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 28px;
    height: 28px;
    padding: 0 6px;
    border: 1.5px solid #b8935a;
    border-radius: 50%;
    color: #b8935a;
    font-family: 'Spectral', serif;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 8px;
    vertical-align: middle;
  }
  .verse-row {
    direction: rtl;
    text-align: right;
    font-family: 'Amiri', serif;
    font-size: 1.85rem;
    line-height: 2.4;
    padding: 10px 4px;
    border-bottom: 1px solid #1e2731;
  }
  .surah-header {
    text-align: center;
    padding: 18px 0 22px;
    border-bottom: 2px solid #2a3542;
    margin-bottom: 12px;
  }
  .surah-header .arabic-name {
    font-family: 'Amiri', serif;
    font-size: 2.2rem;
    color: #b8935a;
  }
  .surah-header .meta {
    font-size: 0.75rem;
    color: #8b95a3;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 4px;
  }
  .bookmark-chip {
    font-size: 0.75rem;
    color: #b8935a;
    padding: 4px 0;
  }
</style>""",
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════════════════════
# AUTH GATE
# Stage 1 fix: previously there were no accounts at all, so every bookmark
# and every recitation result belonged to nobody in particular — which is
# exactly why bookmarks.json collided across concurrent users and results
# had nowhere durable to live. Nothing below this block runs until
# st.session_state.user_id is set.
# ════════════════════════════════════════════════════════════════════════════

if "user_id" not in st.session_state:
    st.session_state.user_id = None
    st.session_state.username = None

if st.session_state.user_id is None:
    st.markdown("## 📖 Quran App")
    st.markdown(
        '<p style="font-size:0.75rem;letter-spacing:0.12em;'
        'text-transform:uppercase;color:#b8935a;">'
        "Local &middot; Offline &middot; Quran-tuned ASR &middot; No paid APIs</p>",
        unsafe_allow_html=True,
    )

    login_tab, register_tab = st.tabs(["Log in", "Create account"])

    with login_tab:
        with st.form("login_form"):
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
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
            new_username = st.text_input("Choose a username", key="new_username")
            new_password = st.text_input("Choose a password", type="password", key="new_password")
            new_password_confirm = st.text_input(
                "Confirm password", type="password", key="new_password_confirm"
            )
            register_submitted = st.form_submit_button("Create account", use_container_width=True)
        if register_submitted:
            if not new_username.strip() or not new_password:
                st.error("Username and password cannot be empty.")
            elif new_password != new_password_confirm:
                st.error("Passwords don't match.")
            elif len(new_password) < 8:
                st.error("Password must be at least 8 characters.")
            else:
                new_uid = db.create_user(new_username, new_password)
                if new_uid is None:
                    st.error("That username is already taken.")
                else:
                    st.session_state.user_id = new_uid
                    st.session_state.username = new_username.strip()
                    st.success("Account created — you're logged in.")
                    st.rerun()

    st.stop()  # nothing below this line executes for a logged-out visitor


# ════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP  (everything below only runs once st.session_state.user_id is set)
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"Signed in as **{st.session_state.username}**")
    if st.button("Log out", use_container_width=True):
        st.session_state.user_id = None
        st.session_state.username = None
        st.rerun()
    st.markdown("---")

st.markdown("## 📖 Quran App")
st.markdown(
    '<p style="font-size:0.75rem;letter-spacing:0.12em;'
    'text-transform:uppercase;color:#b8935a;">'
    "Local &middot; Offline &middot; Quran-tuned ASR &middot; No paid APIs</p>",
    unsafe_allow_html=True,
)

# ── Load resources ───────────────────────────────────────────────────────────
# Quran text is cheap to load (local JSON) -- load eagerly, both modes need it.
# The ASR model is NOT loaded here -- it's ~800MB and only needed in Recite &
# Test mode, so it's loaded lazily inside render_recite_test_mode() instead.
quran = load_quran_data()
surah_numbers = sorted(quran.keys())
surah_options = {f"{n}. {quran[n]['name']}": n for n in surah_numbers}

# ── Session state defaults ───────────────────────────────────────────────────
defaults = {
    "sequence":       [],
    "position":       0,
    "results":        [],   # list of grade_continuous_recitation result dicts
    "session_live":   False,
    "app_mode":       "📖 Read Quran",
    "read_surah":     surah_numbers[0],
    "read_page":      0,
    "hifz_focus":     False,   # hides the starting hint in Recite & Test mode
    "jump_from_surah": None,   # set by "Practice this surah" button in Read mode
    # Exam state defaults
    "exam_live":      False,
    "exam_questions": [],      # list of 10 starting positions/tuples
    "exam_q_idx":     0,       # 0 to 9
    "exam_scores":    [],      # list of scores per question (10.0 or 0.0)
    "exam_details":   [],      # list of feedback dicts per question
    # Stage 2 state
    "review_idx":      0,       # index into today's due-review queue
    "review_audio_key": 0,      # bump to force a fresh st.audio_input widget per ayah
    "active_class_id": None,    # teacher: which class is currently selected
    "active_assignment_id_for_practice": None,  # set when a student launches
                                                 # Recite & Test from an assignment
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ════════════════════════════════════════════════════════════════════════════
# READ QURAN MODE
# ════════════════════════════════════════════════════════════════════════════

def render_read_quran_mode(quran: dict, surah_options: dict, surah_numbers: list):
    bookmarks = load_bookmarks()

    with st.sidebar:
        st.markdown("### Navigate")
        read_label = st.selectbox(
            "Surah",
            list(surah_options.keys()),
            index=surah_numbers.index(st.session_state.read_surah),
            key="read_surah_select",
        )
        selected_surah = surah_options[read_label]
        if selected_surah != st.session_state.read_surah:
            st.session_state.read_surah = selected_surah
            st.session_state.read_page = 0  # reset pagination on surah change
            st.rerun()

        st.markdown("---")
        st.markdown("### 🔖 Bookmarks")
        if not bookmarks:
            st.caption("No bookmarks yet. Tap 🔖 next to any ayah to save it.")
        else:
            for b in sorted(bookmarks, key=lambda x: (x["surah"], x["ayah"])):
                col1, col2 = st.columns([4, 1])
                with col1:
                    label = f"{b['surah_name']} {b['surah']}:{b['ayah']}"
                    if st.button(label, key=f"jump_{b['surah']}_{b['ayah']}", use_container_width=True):
                        st.session_state.read_surah = b["surah"]
                        st.session_state.read_page = (b["ayah"] - 1) // VERSES_PER_PAGE
                        st.rerun()
                with col2:
                    if st.button("✕", key=f"rm_{b['surah']}_{b['ayah']}"):
                        remove_bookmark(b["surah"], b["ayah"])
                        st.rerun()

    surah_num = st.session_state.read_surah
    surah_name = quran[surah_num]["name"]
    verse_nums = sorted(quran[surah_num]["verses"].keys())
    total_verses = len(verse_nums)

    st.markdown(
        f'<div class="surah-header">'
        f'<div class="arabic-name">{surah_name}</div>'
        f'<div class="meta">Surah {surah_num} &middot; {total_verses} verses</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button(f"🎙 Practice Surah {surah_num} in Recite & Test", use_container_width=True):
            st.session_state.jump_from_surah = surah_num
            st.session_state.app_mode = "🎙 Recite & Test"
            st.rerun()
    with col_btn2:
        if st.button(f"📝 Take Exam on Surah {surah_num}", use_container_width=True):
            st.session_state.jump_from_surah = surah_num
            st.session_state.app_mode = "📝 Exam Simulator"
            st.rerun()

    # ── Pagination (keeps long surahs like Al-Baqarah, 286 ayat, responsive) ──
    total_pages = max(1, (total_verses + VERSES_PER_PAGE - 1) // VERSES_PER_PAGE)
    page = min(st.session_state.read_page, total_pages - 1)
    start_idx = page * VERSES_PER_PAGE
    end_idx = min(start_idx + VERSES_PER_PAGE, total_verses)
    page_verse_nums = verse_nums[start_idx:end_idx]

    for ayah_num in page_verse_nums:
        text = quran[surah_num]["verses"][ayah_num]
        bookmarked = is_bookmarked(surah_num, ayah_num, bookmarks)

        col_text, col_btn = st.columns([9, 1])
        with col_text:
            st.markdown(
                f'<div class="verse-row">'
                f'<span class="verse-badge">{ayah_num}</span>{text}'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_btn:
            icon = "🔖" if bookmarked else "🏷️"
            if st.button(icon, key=f"bm_{surah_num}_{ayah_num}", help="Toggle bookmark"):
                if bookmarked:
                    remove_bookmark(surah_num, ayah_num)
                else:
                    add_bookmark(surah_num, ayah_num, surah_name)
                st.rerun()

    # ── Page navigation ──────────────────────────────────────────────────────
    if total_pages > 1:
        st.markdown("---")
        nav_prev, nav_label, nav_next = st.columns([1, 2, 1])
        with nav_prev:
            if page > 0 and st.button("← Previous", use_container_width=True):
                st.session_state.read_page = page - 1
                st.rerun()
        with nav_label:
            st.markdown(
                f'<p style="text-align:center;color:#8b95a3;font-size:0.8rem;">'
                f'Verses {start_idx + 1}–{end_idx} of {total_verses} '
                f'(page {page + 1} of {total_pages})</p>',
                unsafe_allow_html=True,
            )
        with nav_next:
            if page < total_pages - 1 and st.button("Next →", use_container_width=True):
                st.session_state.read_page = page + 1
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# RECITE & TEST MODE
# ════════════════════════════════════════════════════════════════════════════

def render_recite_test_mode(quran: dict, surah_options: dict, surah_numbers: list):
    # Lazy-load the ASR model only when this mode is actually used.
    asr = load_asr_pipeline()

    with st.sidebar:
        st.markdown("### Select Range")

        default_from_idx = 0
        if st.session_state.jump_from_surah is not None:
            default_from_idx = surah_numbers.index(st.session_state.jump_from_surah)
            st.session_state.jump_from_surah = None  # consume the one-time jump

        from_label = st.selectbox(
            "From surah",
            list(surah_options.keys()),
            index=default_from_idx,
            disabled=st.session_state.session_live,
        )
        to_label = st.selectbox(
            "To surah",
            list(surah_options.keys()),
            index=default_from_idx,
            disabled=st.session_state.session_live,
        )

        from_surah = surah_options[from_label]
        to_surah   = surah_options[to_label]

        st.markdown("---")
        st.session_state.hifz_focus = st.toggle(
            "🧠 Hifz Focus Mode",
            value=st.session_state.hifz_focus,
            help="Hides the starting-word hint, so you rely purely on memory rather than a visual cue.",
            disabled=st.session_state.session_live,
        )

        st.markdown("---")

        if not st.session_state.session_live:
            if st.button("▶  Start session", use_container_width=True):
                seq = build_sequence(quran, from_surah, to_surah)
                if seq:
                    start_pos = random.randint(0, max(0, len(seq) - 1))
                    st.session_state.sequence     = seq
                    st.session_state.position     = start_pos
                    st.session_state.results      = []
                    st.session_state.session_live = True
                    st.rerun()
                else:
                    st.error("No ayahs found in that range.")
        else:
            if st.button("⏹  End session", use_container_width=True):
                st.session_state.session_live = False
                st.rerun()

        st.markdown("---")
        st.markdown(
            f'<p style="font-size:0.7rem;color:#8b95a3;">'
            f'Model: <code style="color:#b8935a;">{ASR_MODEL_ID.split("/")[-1]}</code><br>'
            f'Device: <code style="color:#b8935a;">{DEVICE.upper()}</code><br>'
            f'Pass threshold: <code style="color:#b8935a;">{int(PASS_THRESHOLD * 100)}%</code>'
            f"</p>",
            unsafe_allow_html=True,
        )

    if not st.session_state.session_live:
        st.info(
            "Select a surah range in the sidebar and press **▶ Start session** "
            "to begin. You will be given a starting ayah — recite from memory "
            "for as long as you like, then submit."
        )
        return

    seq      = st.session_state.sequence
    position = st.session_state.position

    if position >= len(seq):
        st.success("🎉 You have reached the end of the selected range!")
        st.session_state.session_live = False
        return

    cur_surah, cur_ayah, cur_text = seq[position]
    total = len(seq)
    done  = position

    st.markdown(
        f'<p style="font-size:0.75rem;color:#8b95a3;">'
        f'Ayah {done + 1} of {total} &nbsp;·&nbsp; '
        f'Surah {cur_surah} ({quran[cur_surah]["name"]}), Ayah {cur_ayah}'
        f"</p>",
        unsafe_allow_html=True,
    )

    if st.session_state.hifz_focus:
        st.markdown(
            '<div class="result-panel" style="font-family:Amiri,serif;'
            'direction:rtl;text-align:right;font-size:1.6rem;line-height:1.8;'
            'color:#4a5568;">'
            "🧠 Hifz Focus Mode — recite from memory, no hint shown"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        hint_words = " ".join(cur_text.split()[:3])
        st.markdown(
            f'<div class="result-panel" style="font-family:Amiri,serif;'
            f'direction:rtl;text-align:right;font-size:1.6rem;line-height:1.8;">'
            f"{hint_words} …"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.caption("Complete this ayah (and continue reciting as many as you like).")

    st.markdown("#### 🎙 Record your recitation")
    audio_value = st.audio_input(
        label="Press the microphone button, recite, then press Stop",
        key=f"audio_{position}",
    )

    if audio_value is not None:
        audio_bytes = audio_value.read()

        with st.spinner("Transcribing with the Quranic ASR model …"):
            try:
                transcription = transcribe_audio(audio_bytes, asr)
            except Exception as exc:
                st.error(f"Transcription failed: {exc}")
                return

        transcription = clean_asr_output(transcription)

        if not transcription.strip():
            st.warning(
                "Nothing was transcribed — the recording may have been too quiet "
                "or too short. Try again."
            )
            return

        result, new_position = grade_continuous_recitation(transcription, seq, position)

        if not result["graded"]:
            st.warning(result.get("reason", "Could not align. Try again."))
            with st.expander("Raw transcription"):
                st.write(transcription)
            return

        st.session_state.results.append(result)
        st.session_state.position = new_position
        # Stage 1 fix: previously this result only lived in st.session_state
        # and vanished on refresh/tab-close. It's now durably saved — the
        # "Practice history" section below reads it back from the database,
        # not from session_state, so it survives a refresh.
        db.save_session_result(st.session_state.user_id, result)
        # Stage 2.1: any ayah in this segment with a wrong/missing word goes
        # (or gets reset) into the SRS queue, due immediately.
        db.seed_review_from_session(st.session_state.user_id, result)

        pct = result["similarity"] * 100
        passed = result["passed"]

        st.markdown("---")
        st.markdown("#### Result")

        if passed:
            st.markdown(
                f'<p class="verdict-pass">✓ Correct &nbsp;·&nbsp; {pct:.1f}% similarity</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<p class="verdict-fail">✗ Needs review &nbsp;·&nbsp; {pct:.1f}% similarity</p>',
                unsafe_allow_html=True,
            )

        if result["start_surah"] == result["end_surah"] and result["start_ayah"] == result["end_ayah"]:
            range_str = f"Surah {result['start_surah']}, Ayah {result['start_ayah']}"
        else:
            range_str = (
                f"Surah {result['start_surah']}:{result['start_ayah']} "
                f"→ {result['end_surah']}:{result['end_ayah']}"
            )
        st.caption(f"Graded: {range_str}")

        st.markdown(legend_html(), unsafe_allow_html=True)
        for ayah in result["ayahs"]:
            st.markdown(render_ayah_block(ayah), unsafe_allow_html=True)

        with st.expander("Raw ASR transcription"):
            st.write(result["transcribed"])

        st.markdown("---")

        if new_position >= len(seq):
            st.success("🎉 You have completed the entire selected range!")
            st.session_state.session_live = False
        else:
            next_s, next_a, _ = seq[new_position]
            remaining = len(seq) - new_position
            st.info(
                f"Next: Surah {next_s} ({quran[next_s]['name']}), "
                f"Ayah {next_a} &nbsp;·&nbsp; {remaining} ayah(s) remaining."
            )
            st.button("Continue →", on_click=lambda: None, key="continue_btn")
            st.rerun()

    # ── All-time practice history ────────────────────────────────────────────
    # Stage 1 fix: this reads from the database, not st.session_state, so it
    # is still here after a refresh, a closed tab, or a new login tomorrow —
    # unlike the "this session" expander just below it, which (by design)
    # only covers segments graded since the current session started.
    st.markdown("---")

    # Stage 2.2: streak — shown unconditionally (not tucked in an expander)
    # since it's meant to be the thing that pulls someone back tomorrow.
    streak = db.get_current_streak(st.session_state.user_id)
    if streak > 0:
        st.markdown(
            f'<p style="font-size:0.85rem;color:#b8935a;">🔥 {streak}-day streak — keep it going.</p>',
            unsafe_allow_html=True,
        )

    overall = db.get_overall_stats(st.session_state.user_id)
    if overall["n_sessions"] > 0:
        with st.expander(f"📊 All-time practice history ({overall['n_sessions']} segment(s) ever graded)"):
            col1, col2 = st.columns(2)
            col1.metric("Overall similarity (all-time)", f"{overall['avg_similarity'] * 100:.1f}%")
            col2.metric("Pass rate (all-time)", f"{overall['pass_rate'] * 100:.1f}%")

            trend = db.get_daily_activity(st.session_state.user_id, days=30)
            if len(trend) >= 2:
                st.caption("Similarity trend (last 30 active days):")
                st.line_chart(
                    {row["day"]: row["avg_similarity"] * 100 for row in trend}
                )

            weak = db.get_weak_ayahs(st.session_state.user_id, limit=10)
            if weak:
                st.caption("Most-missed ayahs across all your sessions:")
                for w in weak:
                    name = quran.get(w["surah"], {}).get("name", f"Surah {w['surah']}")
                    st.markdown(
                        f"- **{name} {w['surah']}:{w['ayah']}** — missed in "
                        f"{w['miss_count']} word instance(s)"
                    )

            recent = db.get_session_history(st.session_state.user_id, limit=20)
            st.caption("Recent sessions:")
            for r in recent:
                label = "✓" if r["passed"] else "✗"
                st.markdown(
                    f"{label} {r['start_surah']}:{r['start_ayah']} → "
                    f"{r['end_surah']}:{r['end_ayah']} — "
                    f"{r['similarity'] * 100:.1f}% — {r['created_at'][:10]}"
                )

    if st.session_state.results:
        with st.expander(f"Session history ({len(st.session_state.results)} segment(s))"):
            total_scored = sum(
                len([w for a in r["ayahs"] for w in a["words"]
                     if w["status"] != "not_recited"])
                for r in st.session_state.results
            )
            total_correct = sum(
                len([w for a in r["ayahs"] for w in a["words"]
                     if w["status"] in ("correct", "close")])
                for r in st.session_state.results
            )
            overall_pct = (total_correct / total_scored * 100) if total_scored else 0
            st.metric("Overall similarity (this session)", f"{overall_pct:.1f}%")

            for i, r in enumerate(reversed(st.session_state.results), 1):
                pct = r["similarity"] * 100
                label = "✓" if r["passed"] else "✗"
                with st.expander(
                    f"{label} Segment {len(st.session_state.results) - i + 1} — "
                    f"{r['start_surah']}:{r['start_ayah']} → "
                    f"{r['end_surah']}:{r['end_ayah']} — {pct:.1f}%"
                ):
                    for ayah in r["ayahs"]:
                        st.markdown(render_ayah_block(ayah), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# EXAM SIMULATOR MODE
# ════════════════════════════════════════════════════════════════════════════

def render_exam_simulator_mode(quran: dict, surah_options: dict, surah_numbers: list):
    # Lazy-load ASR model
    asr = load_asr_pipeline()

    with st.sidebar:
        st.markdown("### 📝 Exam Setup")

        default_from_idx = 0
        if st.session_state.jump_from_surah is not None:
            if st.session_state.jump_from_surah in surah_numbers:
                default_from_idx = surah_numbers.index(st.session_state.jump_from_surah)
            st.session_state.jump_from_surah = None

        exam_from_label = st.selectbox(
            "From surah",
            list(surah_options.keys()),
            index=default_from_idx,
            disabled=st.session_state.exam_live or (st.session_state.exam_q_idx == 10 and len(st.session_state.exam_scores) == 10),
            key="exam_from_sel"
        )
        exam_to_label = st.selectbox(
            "To surah",
            list(surah_options.keys()),
            index=default_from_idx,
            disabled=st.session_state.exam_live or (st.session_state.exam_q_idx == 10 and len(st.session_state.exam_scores) == 10),
            key="exam_to_sel"
        )

        exam_from_surah = surah_options[exam_from_label]
        exam_to_surah   = surah_options[exam_to_label]

        st.markdown("---")
        if not st.session_state.exam_live:
            if st.button("🚀 Start 10-Question Exam", use_container_width=True):
                seq = build_sequence(quran, exam_from_surah, exam_to_surah)
                if not seq or len(seq) < 10:
                    st.error("Selected range is too short for a 10-question exam (needs at least 10 ayahs).")
                else:
                    max_start = len(seq) - 8
                    if max_start < 1:
                        max_start = len(seq)
                    sampled_indices = sorted(random.sample(range(0, max_start), min(10, max_start)))
                    while len(sampled_indices) < 10 and len(sampled_indices) < len(seq):
                        r = random.randint(0, len(seq) - 1)
                        if r not in sampled_indices:
                            sampled_indices.append(r)
                    sampled_indices = sorted(sampled_indices[:10])

                    st.session_state.sequence = seq
                    st.session_state.exam_questions = sampled_indices
                    st.session_state.exam_q_idx = 0
                    st.session_state.exam_scores = []
                    st.session_state.exam_details = []
                    st.session_state.exam_live = True
                    st.rerun()
        else:
            if st.button("⏹ Abort Exam", use_container_width=True):
                st.session_state.exam_live = False
                st.rerun()

        st.markdown("---")
        st.markdown(
            '<p style="font-size:0.7rem;color:#8b95a3;">'
            "<strong>Format:</strong> 10 Spot Identification questions.<br>"
            "<strong>Scoring:</strong> 10% per question. Binary pass/fail based on strict accuracy within half-page target."
            "</p>",
            unsafe_allow_html=True,
        )

    # Ensure sequence exists in session state before rendering active exam steps
    if not st.session_state.get("sequence") and not st.session_state.exam_live:
        st.markdown("## 📝 Oral Hifz Exam Simulator")
        st.info(
            "Configure your target Surah range in the sidebar and click **🚀 Start 10-Question Exam**.\n\n"
            "**Rules:**\n"
            "- Exactly 10 spot-identification questions.\n"
            "- Each question requires continuing recitation for roughly a half-page span.\n"
            "- Strict binary marking (10% per question): any hesitation, word mistake, or gap results in a 0% deduction for that question."
        )
        return

    # ── Exam Completed State ──
    if st.session_state.exam_live and st.session_state.exam_q_idx >= 10:
        st.markdown("## 🎓 Official Oral Exam Results")
        total_score = sum(st.session_state.exam_scores)
        
        if total_score >= 80:
            st.markdown(f'<p class="verdict-pass" style="font-size:1.5rem;">Passed with {total_score:.0f}%</p>', unsafe_allow_html=True)
        elif total_score >= 60:
            st.markdown(f'<p style="color:#b8935a; font-family:Spectral,serif; font-size:1.5rem;">Satisfactory: {total_score:.0f}%</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="verdict-fail" style="font-size:1.5rem;">Needs Review: {total_score:.0f}%</p>', unsafe_allow_html=True)

        st.markdown(f"**Final Mark:** {total_score:.0f} / 100%")
        st.markdown("---")

        for idx, detail in enumerate(st.session_state.exam_details, 1):
            score_label = "10/10 (10%)" if detail["score"] > 0 else "0/10 (0%)"
            icon = "✓" if detail["score"] > 0 else "✗"
            with st.expander(f"{icon} Question {idx} — Score: {score_label} (Target: Surah {detail['start_surah']}:{detail['start_ayah']})"):
                st.caption(f"Evaluated length: {len(detail['ayahs'])} ayah(s)")
                for ayah in detail["ayahs"]:
                    st.markdown(render_ayah_block(ayah), unsafe_allow_html=True)

        if st.button("🔄 Start New Exam"):
            st.session_state.exam_live = False
            st.session_state.exam_q_idx = 0
            st.session_state.exam_scores = []
            st.session_state.exam_details = []
            st.rerun()
        return

    # ── Active Exam Question Flow ──
    q_num = st.session_state.exam_q_idx + 1
    seq = st.session_state.sequence
    
    if st.session_state.exam_q_idx >= len(st.session_state.exam_questions):
        st.session_state.exam_q_idx = len(st.session_state.exam_questions)
        st.rerun()
        
    target_pos = st.session_state.exam_questions[st.session_state.exam_q_idx]
    surah_num, ayah_num, ayah_text = seq[target_pos]

    st.markdown(f"### Question {q_num} of 10")
    st.markdown(
        f'<p style="font-size:0.75rem;color:#8b95a3;text-transform:uppercase;letter-spacing:0.08em;">'
        f'Spot Target &middot; Surah {surah_num} ({quran[surah_num]["name"]}), Ayah {ayah_num}'
        f'</p>',
        unsafe_allow_html=True,
    )

    hint_words = " ".join(ayah_text.split()[:3])
    st.markdown(
        f'<div class="result-panel" style="font-family:Amiri,serif;'
        f'direction:rtl;text-align:right;font-size:1.6rem;line-height:1.8;">'
        f'{hint_words} …'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.caption("Spot Identification: Continue reciting continuously for a half-page span.")

    st.markdown("#### 🎙 Record your oral response")
    audio_val = st.audio_input(
        label=f"Record recitation for Question {q_num}",
        key=f"exam_audio_{st.session_state.exam_q_idx}"
    )

    if audio_val is not None:
        audio_bytes = audio_val.read()
        with st.spinner("Evaluating oral exam response..."):
            try:
                transcription = transcribe_audio(audio_bytes, asr)
            except Exception as e:
                st.error(f"Transcription error: {e}")
                return

        transcription = clean_asr_output(transcription)
        if not transcription.strip():
            st.warning("No audio detected or transcription was empty. Please record again.")
            return

        result, _ = grade_continuous_recitation(transcription, seq, target_pos)

        if not result["graded"]:
            st.warning(result.get("reason", "Could not align recitation. Try again."))
            return

        passed_question = result["passed"] and len(result["ayahs"]) >= 3
        question_score = 10.0 if passed_question else 0.0

        st.session_state.exam_scores.append(question_score)
        st.session_state.exam_details.append({
            "score": question_score,
            "start_surah": surah_num,
            "start_ayah": ayah_num,
            "ayahs": result["ayahs"],
            "similarity": result["similarity"]
        })

        st.session_state.exam_q_idx += 1
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# STAGE 2.1 — TODAY'S REVIEW (SRS QUEUE)
#
# Reuses grade_continuous_recitation() and render_ayah_block()/legend_html()
# exactly as-is — an SRS review is graded identically to a normal segment,
# it just starts from a due ayah instead of a random one and reports back
# into review_state instead of (only) session history.
# ════════════════════════════════════════════════════════════════════════════

def render_review_queue_mode(quran: dict, asr_loader):
    due = db.get_due_reviews(st.session_state.user_id, limit=20)

    if not due:
        st.markdown("## 🔁 Today's Review")
        st.success(
            "Nothing due right now. Ayahs land here automatically whenever "
            "a **Recite & Test** segment comes back with a wrong or missing "
            "word — come back after your next practice session."
        )
        return

    if st.session_state.review_idx >= len(due):
        st.markdown("## 🔁 Today's Review")
        st.success(f"🎉 Cleared all {len(due)} review(s) due today. Nice work.")
        if st.button("Start over"):
            st.session_state.review_idx = 0
            st.rerun()
        return

    asr = asr_loader()
    item = due[st.session_state.review_idx]
    surah_num, ayah_num = item["surah"], item["ayah"]
    ayah_name = quran.get(surah_num, {}).get("name", f"Surah {surah_num}")
    ayah_text = quran.get(surah_num, {}).get("verses", {}).get(ayah_num, "")

    st.markdown("## 🔁 Today's Review")
    status_label = "Reviewed" if item["repetitions"] else "New"
    if item["last_reviewed_at"]:
        recency_label = f'missed last on {item["last_reviewed_at"][:10]}'
    else:
        recency_label = "first time up"
    st.markdown(
        f'<p style="font-size:0.75rem;color:#8b95a3;">'
        f'Review {st.session_state.review_idx + 1} of {len(due)} &nbsp;·&nbsp; '
        f'{status_label} &nbsp;·&nbsp; {recency_label}'
        f"</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="font-size:0.75rem;color:#8b95a3;text-transform:uppercase;letter-spacing:0.08em;">'
        f'Surah {surah_num} ({ayah_name}), Ayah {ayah_num}</p>',
        unsafe_allow_html=True,
    )

    hint_words = " ".join(ayah_text.split()[:3]) if ayah_text else ""
    st.markdown(
        f'<div class="result-panel" style="font-family:Amiri,serif;'
        f'direction:rtl;text-align:right;font-size:1.6rem;line-height:1.8;">'
        f'{hint_words} …</div>',
        unsafe_allow_html=True,
    )
    st.caption("Recite this ayah from memory, then submit.")

    seq = [(surah_num, ayah_num, ayah_text)]
    audio_val = st.audio_input(
        label="Record your review recitation",
        key=f"review_audio_{item['id']}_{st.session_state.review_audio_key}",
    )

    if audio_val is not None:
        audio_bytes = audio_val.read()
        with st.spinner("Grading review …"):
            try:
                transcription = clean_asr_output(transcribe_audio(audio_bytes, asr))
            except Exception as exc:
                st.error(f"Transcription failed: {exc}")
                return

        if not transcription.strip():
            st.warning("Nothing was transcribed. Try again.")
            return

        result, _ = grade_continuous_recitation(transcription, seq, 0)
        if not result["graded"]:
            st.warning(result.get("reason", "Could not align. Try again."))
            return

        db.save_session_result(st.session_state.user_id, result)

        # Derive an SM-2 quality score (0-5) from the grading result — a
        # clean pass is a 5, a pass with some 'close' words is a 4, and any
        # failure is a 1 (low but nonzero, so ease_factor still degrades
        # gracefully instead of being wiped to the floor in one review).
        words = [w for a in result["ayahs"] for w in a["words"]]
        has_wrong_or_missing = any(w["status"] in ("wrong", "missing") for w in words)
        has_close = any(w["status"] == "close" for w in words)
        if not result["passed"] or has_wrong_or_missing:
            quality = 1
        elif has_close:
            quality = 4
        else:
            quality = 5

        db.record_review_result(st.session_state.user_id, surah_num, ayah_num, quality)

        pct = result["similarity"] * 100
        st.markdown("---")
        if quality >= 4:
            st.markdown(f'<p class="verdict-pass">✓ Well done — {pct:.1f}% similarity</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="verdict-fail">✗ Still shaky — {pct:.1f}% similarity, scheduled sooner</p>', unsafe_allow_html=True)

        st.markdown(legend_html(), unsafe_allow_html=True)
        for ayah in result["ayahs"]:
            st.markdown(render_ayah_block(ayah), unsafe_allow_html=True)

        st.markdown("---")
        if st.button("Next review →"):
            st.session_state.review_idx += 1
            st.session_state.review_audio_key += 1
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# STAGE 2.3 — TEACHER DASHBOARD MVP
#
# Any account can act as a teacher — there's no separate role flag yet
# (deliberate v1 scope cut, same spirit as Stage 1's simple-auth decision).
# A student completes assignments through the *existing* Recite & Test flow;
# this mode only adds roster/assignment management and a read-only review
# queue over sessions that flow has already graded and saved.
# ════════════════════════════════════════════════════════════════════════════

def render_teacher_mode(quran: dict, surah_options: dict, surah_numbers: list):
    st.markdown("## 🎓 Teacher Dashboard")

    classes = db.get_teacher_classes(st.session_state.user_id)
    class_names = {c["name"]: c["id"] for c in classes}

    with st.expander("➕ Create a new class", expanded=not classes):
        with st.form("new_class_form"):
            new_class_name = st.text_input("Class name")
            create_submitted = st.form_submit_button("Create class")
        if create_submitted:
            if not new_class_name.strip():
                st.error("Class name can't be empty.")
            else:
                cid = db.create_class(st.session_state.user_id, new_class_name)
                st.session_state.active_class_id = cid
                st.rerun()

    if not classes:
        st.info("Create a class above to start building a roster and assigning ranges.")
        return

    label_by_id = {c["id"]: c["name"] for c in classes}
    default_id = st.session_state.active_class_id if st.session_state.active_class_id in label_by_id else classes[0]["id"]
    selected_label = st.selectbox(
        "Class", list(class_names.keys()),
        index=list(class_names.values()).index(default_id),
    )
    class_id = class_names[selected_label]
    st.session_state.active_class_id = class_id

    roster_tab, assign_tab, review_tab = st.tabs(["👥 Roster", "📋 Assignments", "🔎 Review Submissions"])

    with roster_tab:
        with st.form("add_student_form"):
            student_username = st.text_input("Add student by username")
            add_submitted = st.form_submit_button("Add to class")
        if add_submitted:
            if not student_username.strip():
                st.error("Enter a username.")
            elif db.add_student_to_class(class_id, student_username):
                st.success(f"Added {student_username.strip()}.")
                st.rerun()
            else:
                st.error("No account with that username exists.")

        roster = db.get_class_roster(class_id)
        if roster:
            st.markdown(f"**{len(roster)} student(s):**")
            for s in roster:
                st.markdown(f"- {s['username']}")
        else:
            st.caption("No students added yet.")

    with assign_tab:
        with st.form("new_assignment_form"):
            col1, col2 = st.columns(2)
            with col1:
                a_from_label = st.selectbox("From surah", list(surah_options.keys()), key="assign_from")
            with col2:
                a_to_label = st.selectbox("To surah", list(surah_options.keys()), key="assign_to")
            due = st.date_input("Due date (optional)", value=None)
            assign_submitted = st.form_submit_button("Create assignment")
        if assign_submitted:
            from_surah = surah_options[a_from_label]
            to_surah = surah_options[a_to_label]
            due_str = due.isoformat() if due else None
            db.create_assignment(class_id, from_surah, 1, to_surah, 1, due_str)
            st.success("Assignment created.")
            st.rerun()

        assignments = db.get_class_assignments(class_id)
        if assignments:
            st.markdown("**Existing assignments:**")
            for a in assignments:
                due_label = f" — due {a['due_date']}" if a["due_date"] else ""
                st.markdown(
                    f"- Surah {a['start_surah']}–{a['end_surah']}{due_label} "
                    f"(created {a['created_at'][:10]})"
                )
        else:
            st.caption("No assignments yet.")

    with review_tab:
        assignments = db.get_class_assignments(class_id)
        if not assignments:
            st.caption("Create an assignment first.")
        else:
            a_label_map = {
                f"Surah {a['start_surah']}–{a['end_surah']} (created {a['created_at'][:10]})": a["id"]
                for a in assignments
            }
            chosen = st.selectbox("Assignment", list(a_label_map.keys()))
            assignment_id = a_label_map[chosen]
            submissions = db.get_assignment_submissions(assignment_id)
            if not submissions:
                st.info("No submissions yet for this assignment.")
            for sub in submissions:
                pct = sub["similarity"] * 100
                icon = "✓" if sub["passed"] else "✗"
                with st.expander(
                    f"{icon} {sub['username']} — {sub['start_surah']}:{sub['start_ayah']} → "
                    f"{sub['end_surah']}:{sub['end_ayah']} — {pct:.1f}% ({sub['submitted_at'][:10]})"
                ):
                    words = db.get_session_words(sub["session_id"])
                    by_ayah = {}
                    for w in words:
                        key = (w["surah"], w["ayah"])
                        by_ayah.setdefault(key, []).append({"text": w["word_text"], "status": w["status"]})
                    for (s_num, a_num), wlist in by_ayah.items():
                        st.markdown(
                            render_ayah_block({"surah": s_num, "ayah": a_num, "words": wlist}),
                            unsafe_allow_html=True,
                        )

                    note = st.text_area(
                        "Teacher note", value=sub["teacher_note"] or "",
                        key=f"note_{sub['submission_id']}",
                    )
                    override = st.selectbox(
                        "Override verdict",
                        ["(none)", "Pass", "Needs review"],
                        index=(0 if not sub["teacher_override"] else
                               (1 if sub["teacher_override"] == "Pass" else 2)),
                        key=f"override_{sub['submission_id']}",
                    )
                    if st.button("Save review", key=f"save_{sub['submission_id']}"):
                        db.set_teacher_review(
                            sub["submission_id"], note,
                            None if override == "(none)" else override,
                        )
                        st.success("Saved.")
                        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# MODE DISPATCH
# ════════════════════════════════════════════════════════════════════════════

_MODE_OPTIONS = [
    "📖 Read Quran",
    "🔁 Today's Review",
    "🎙 Recite & Test",
    "📝 Exam Simulator",
    "🎓 Teacher",
]
if st.session_state.app_mode not in _MODE_OPTIONS:
    st.session_state.app_mode = _MODE_OPTIONS[0]

with st.sidebar:
    st.markdown("### Mode")
    due_count = db.get_review_queue_size(st.session_state.user_id)
    mode_labels = list(_MODE_OPTIONS)
    if due_count:
        mode_labels[1] = f"🔁 Today's Review ({due_count})"
    mode = st.radio(
        "Choose a mode",
        mode_labels,
        index=_MODE_OPTIONS.index(st.session_state.app_mode),
        label_visibility="collapsed",
    )
    mode = _MODE_OPTIONS[mode_labels.index(mode)]  # strip the "(n)" suffix back off
    if mode != st.session_state.app_mode:
        st.session_state.app_mode = mode
        st.rerun()
    st.markdown("---")

if st.session_state.app_mode == "📖 Read Quran":
    render_read_quran_mode(quran, surah_options, surah_numbers)
elif st.session_state.app_mode == "🔁 Today's Review":
    render_review_queue_mode(quran, asr_loader=load_asr_pipeline)
elif st.session_state.app_mode == "🎙 Recite & Test":
    render_recite_test_mode(quran, surah_options, surah_numbers)
elif st.session_state.app_mode == "📝 Exam Simulator":
    render_exam_simulator_mode(quran, surah_options, surah_numbers)
else:
    render_teacher_mode(quran, surah_options, surah_numbers)