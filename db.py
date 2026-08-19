"""
Persistence layer — Stage 1 foundation fix.

Replaces:
  - bookmarks.json (flat file, broke with >1 concurrent user)
  - st.session_state.results (wiped on every page refresh)

Uses SQLite (stdlib `sqlite3`, no external dependency) so this works
identically in local dev and on a small deployment without needing a
hosted database service. If/when the app outgrows SQLite's concurrent-
write limits, every function here maps 1:1 onto Postgres — swap the
connection layer, keep the call sites in streamlit_app.py unchanged.

Password hashing uses stdlib `hashlib.pbkdf2_hmac` (no bcrypt/passlib
dependency needed) with a random per-user salt and 260,000 iterations
(OWASP's current minimum recommendation for PBKDF2-SHA256, as of their
2023 cheat sheet).
"""

import hashlib
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta

DB_PATH = os.environ.get("APP_DB_PATH", "app.db")

_PBKDF2_ITERATIONS = 260_000


# ════════════════════════════════════════════════════════════════════════════
# CONNECTION / SCHEMA
# ════════════════════════════════════════════════════════════════════════════

@contextmanager
def get_conn():
    """Yields a SQLite connection with foreign keys enabled and row access
    by column name. Always used as a context manager so connections are
    never leaked across Streamlit reruns."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Creates all tables if they don't exist yet. Safe to call on every
    app startup — CREATE TABLE IF NOT EXISTS is idempotent."""
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                created_at    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS bookmarks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                surah       INTEGER NOT NULL,
                ayah        INTEGER NOT NULL,
                surah_name  TEXT NOT NULL,
                note        TEXT DEFAULT '',
                added_at    TEXT NOT NULL,
                UNIQUE(user_id, surah, ayah)
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                start_surah  INTEGER NOT NULL,
                start_ayah   INTEGER NOT NULL,
                end_surah    INTEGER NOT NULL,
                end_ayah     INTEGER NOT NULL,
                similarity   REAL NOT NULL,
                passed       INTEGER NOT NULL,
                created_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS session_words (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                surah       INTEGER NOT NULL,
                ayah        INTEGER NOT NULL,
                word_text   TEXT NOT NULL,
                status      TEXT NOT NULL,
                word_order  INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_bookmarks_user ON bookmarks(user_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_session_words_session ON session_words(session_id);
            -- Weak-ayah lookup (feeds the Stage 2 SRS queue directly off this index)
            CREATE INDEX IF NOT EXISTS idx_session_words_status ON session_words(surah, ayah, status);

            -- ═══════════════════════════════════════════════════════════════
            -- STAGE 2.1 — SRS review queue (SM-2)
            -- ═══════════════════════════════════════════════════════════════
            CREATE TABLE IF NOT EXISTS review_state (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                surah         INTEGER NOT NULL,
                ayah          INTEGER NOT NULL,
                ease_factor   REAL NOT NULL DEFAULT 2.5,
                interval_days REAL NOT NULL DEFAULT 0,
                repetitions   INTEGER NOT NULL DEFAULT 0,
                next_due_date TEXT NOT NULL,
                last_reviewed_at TEXT,
                UNIQUE(user_id, surah, ayah)
            );

            CREATE INDEX IF NOT EXISTS idx_review_state_due ON review_state(user_id, next_due_date);

            -- ═══════════════════════════════════════════════════════════════
            -- STAGE 2.3 — Teacher Dashboard MVP
            -- ═══════════════════════════════════════════════════════════════
            CREATE TABLE IF NOT EXISTS classes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                teacher_id  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                name        TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS class_students (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                class_id    INTEGER NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
                student_id  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                added_at    TEXT NOT NULL,
                UNIQUE(class_id, student_id)
            );

            CREATE TABLE IF NOT EXISTS assignments (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                class_id      INTEGER NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
                start_surah   INTEGER NOT NULL,
                start_ayah    INTEGER NOT NULL,
                end_surah     INTEGER NOT NULL,
                end_ayah      INTEGER NOT NULL,
                due_date      TEXT,
                created_at    TEXT NOT NULL
            );

            -- Links a graded session to the assignment it was submitted for.
            -- Nullable session_id note isn't needed since we insert this row
            -- at the moment a matching session is saved (see submit_assignment_result).
            CREATE TABLE IF NOT EXISTS assignment_submissions (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                assignment_id  INTEGER NOT NULL REFERENCES assignments(id) ON DELETE CASCADE,
                student_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                session_id     INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                teacher_note   TEXT DEFAULT '',
                teacher_override TEXT,
                submitted_at   TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_class_students_class ON class_students(class_id);
            CREATE INDEX IF NOT EXISTS idx_assignments_class ON assignments(class_id);
            CREATE INDEX IF NOT EXISTS idx_submissions_assignment ON assignment_submissions(assignment_id);
            """
        )


# ════════════════════════════════════════════════════════════════════════════
# AUTH
# ════════════════════════════════════════════════════════════════════════════

def _hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    """Returns (hash_hex, salt_hex). Generates a new random salt if none given."""
    if salt is None:
        salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return digest.hex(), salt.hex()


def create_user(username: str, password: str) -> int | None:
    """Returns the new user's id, or None if the username is already taken.
    Raises ValueError for empty username/password (caller should validate
    in the UI too, but never trust the UI layer alone)."""
    username = username.strip()
    if not username or not password:
        raise ValueError("Username and password cannot be empty.")

    password_hash, password_salt = _hash_password(password)
    with get_conn() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, password_salt, created_at) "
                "VALUES (?, ?, ?, ?)",
                (username, password_hash, password_salt, datetime.now(timezone.utc).isoformat()),
            )
            return cur.lastrowid
        except sqlite3.IntegrityError:
            return None  # username already exists


def authenticate_user(username: str, password: str) -> int | None:
    """Returns the user's id if the password is correct, else None.
    Constant-time-ish: always hashes even on username-not-found to avoid
    trivially timing whether a username exists (not perfect, but better
    than short-circuiting immediately)."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, password_hash, password_salt FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()

    if row is None:
        _hash_password(password, os.urandom(16))  # dummy hash, keeps timing similar
        return None

    computed_hash, _ = _hash_password(password, bytes.fromhex(row["password_salt"]))
    if computed_hash == row["password_hash"]:
        return row["id"]
    return None


# ════════════════════════════════════════════════════════════════════════════
# BOOKMARKS  (user-scoped — this is the fix for the concurrent-user bug)
# ════════════════════════════════════════════════════════════════════════════

def load_bookmarks(user_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT surah, ayah, surah_name, note, added_at FROM bookmarks "
            "WHERE user_id = ? ORDER BY surah, ayah",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def add_bookmark(user_id: int, surah: int, ayah: int, surah_name: str, note: str = "") -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO bookmarks (user_id, surah, ayah, surah_name, note, added_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, surah, ayah, surah_name, note, datetime.now(timezone.utc).isoformat()),
        )


def remove_bookmark(user_id: int, surah: int, ayah: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM bookmarks WHERE user_id = ? AND surah = ? AND ayah = ?",
            (user_id, surah, ayah),
        )


def is_bookmarked(surah: int, ayah: int, bookmarks: list[dict]) -> bool:
    """Pure helper, unchanged signature from the old flat-file version —
    still just checks membership in an already-loaded list."""
    return any(b["surah"] == surah and b["ayah"] == ayah for b in bookmarks)


# ════════════════════════════════════════════════════════════════════════════
# SESSION / GRADING HISTORY  (this is the "no data loss on refresh" fix)
# ════════════════════════════════════════════════════════════════════════════

def save_session_result(user_id: int, result: dict) -> int:
    """Persists one grade_continuous_recitation() result dict.

    Expects the same shape the grading function already produces:
      result = {
        "start_surah": int, "start_ayah": int,
        "end_surah": int, "end_ayah": int,
        "similarity": float, "passed": bool,
        "ayahs": [ {"surah": int, "ayah": int,
                     "words": [ {"text": str, "status": str}, ... ] }, ... ],
        ...
      }

    Returns the new session's row id.
    """
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO sessions "
            "(user_id, start_surah, start_ayah, end_surah, end_ayah, similarity, passed, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                result["start_surah"], result["start_ayah"],
                result["end_surah"], result["end_ayah"],
                result["similarity"], int(bool(result["passed"])),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        session_id = cur.lastrowid

        word_rows = []
        order = 0
        for ayah in result["ayahs"]:
            for w in ayah["words"]:
                word_rows.append((session_id, ayah["surah"], ayah["ayah"], w["text"], w["status"], order))
                order += 1
        if word_rows:
            conn.executemany(
                "INSERT INTO session_words (session_id, surah, ayah, word_text, status, word_order) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                word_rows,
            )
        return session_id


def get_session_history(user_id: int, limit: int | None = 50) -> list[dict]:
    """Returns past sessions, newest first, WITHOUT per-word detail (cheap —
    use get_session_words() for a specific session if the UI needs the
    word-level breakdown, e.g. when a user expands one session)."""
    query = (
        "SELECT id, start_surah, start_ayah, end_surah, end_ayah, similarity, passed, created_at "
        "FROM sessions WHERE user_id = ? ORDER BY created_at DESC"
    )
    params: tuple = (user_id,)
    if limit is not None:
        query += " LIMIT ?"
        params = (user_id, limit)

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_session_words(session_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT surah, ayah, word_text, status FROM session_words "
            "WHERE session_id = ? ORDER BY word_order",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_overall_stats(user_id: int) -> dict:
    """Aggregate accuracy across ALL persisted sessions (survives refresh —
    this is the number that used to reset to zero every time the tab closed)."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n_sessions, AVG(similarity) AS avg_similarity, "
            "SUM(passed) AS n_passed FROM sessions WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    n_sessions = row["n_sessions"] or 0
    return {
        "n_sessions": n_sessions,
        "avg_similarity": (row["avg_similarity"] or 0.0),
        "n_passed": row["n_passed"] or 0,
        "pass_rate": (row["n_passed"] / n_sessions) if n_sessions else 0.0,
    }


# ════════════════════════════════════════════════════════════════════════════
# STAGE 2.1 — SRS REVIEW QUEUE (SM-2)
#
# Seeding: any ayah with a 'wrong'/'missing' word in a graded session gets a
# row here (or has its schedule reset) via seed_review_from_session(), which
# streamlit_app.py calls right after db.save_session_result(). An ayah with
# a *clean* segment (no wrong/missing words) is NOT auto-added — the queue
# is for ayahs that need reinforcement, not a log of everything ever recited.
#
# Scheduling: textbook SM-2, deliberately not over-engineered for v1.
#   quality 0-2 (fail)   -> repetitions reset to 0, interval = 1 day
#   quality 3-5 (pass)   -> repetitions += 1, interval grows by ease_factor
#   ease_factor is nudged by quality every review, floored at 1.3
# ════════════════════════════════════════════════════════════════════════════

_SM2_MIN_EASE = 1.3


def _sm2_next(ease_factor: float, interval_days: float, repetitions: int, quality: int):
    """Pure SM-2 step. quality is 0-5 (Anki-style: 0-2 fail, 3-5 pass)."""
    if quality < 3:
        repetitions = 0
        interval_days = 1.0
    else:
        repetitions += 1
        if repetitions == 1:
            interval_days = 1.0
        elif repetitions == 2:
            interval_days = 6.0
        else:
            interval_days = round(interval_days * ease_factor, 2)

    ease_factor = ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    ease_factor = max(_SM2_MIN_EASE, round(ease_factor, 2))
    return ease_factor, interval_days, repetitions


def seed_review_from_session(user_id: int, result: dict) -> None:
    """Call right after save_session_result(). Any ayah in this result that
    has at least one 'wrong' or 'missing' word gets scheduled for review
    (due immediately) — inserted fresh, or reset if it already existed."""
    today = datetime.now(timezone.utc).date().isoformat()
    with get_conn() as conn:
        for ayah in result["ayahs"]:
            has_miss = any(w["status"] in ("wrong", "missing") for w in ayah["words"])
            if not has_miss:
                continue
            conn.execute(
                """
                INSERT INTO review_state (user_id, surah, ayah, ease_factor, interval_days,
                                           repetitions, next_due_date, last_reviewed_at)
                VALUES (?, ?, ?, 2.5, 1, 0, ?, NULL)
                ON CONFLICT(user_id, surah, ayah) DO UPDATE SET
                    next_due_date = excluded.next_due_date
                """,
                (user_id, ayah["surah"], ayah["ayah"], today),
            )


def get_due_reviews(user_id: int, limit: int = 20) -> list[dict]:
    """Ayahs due today or earlier, most-overdue first. This is what 'Today's
    Review' renders — empty list means the queue is genuinely empty, not
    that nothing has ever been tracked."""
    today = datetime.now(timezone.utc).date().isoformat()
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, surah, ayah, ease_factor, interval_days, repetitions,
                   next_due_date, last_reviewed_at
            FROM review_state
            WHERE user_id = ? AND next_due_date <= ?
            ORDER BY next_due_date ASC
            LIMIT ?
            """,
            (user_id, today, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def record_review_result(user_id: int, surah: int, ayah: int, quality: int) -> None:
    """Advances one ayah's SM-2 schedule. quality: 0-5, where the caller
    (streamlit_app.py) derives it from the grading result — e.g. passed and
    no wrong words -> 5, passed with some close words -> 4, failed -> 1."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT ease_factor, interval_days, repetitions FROM review_state "
            "WHERE user_id = ? AND surah = ? AND ayah = ?",
            (user_id, surah, ayah),
        ).fetchone()
        if row is None:
            return  # nothing to advance — ayah was never queued

        ease, interval, reps = row["ease_factor"], row["interval_days"], row["repetitions"]
        new_ease, new_interval, new_reps = _sm2_next(ease, interval, reps, quality)

        now = datetime.now(timezone.utc)
        due = (now + timedelta(days=new_interval)).date().isoformat()

        conn.execute(
            """
            UPDATE review_state
            SET ease_factor = ?, interval_days = ?, repetitions = ?,
                next_due_date = ?, last_reviewed_at = ?
            WHERE user_id = ? AND surah = ? AND ayah = ?
            """,
            (new_ease, new_interval, new_reps, due, now.isoformat(), user_id, surah, ayah),
        )


def get_review_queue_size(user_id: int) -> int:
    today = datetime.now(timezone.utc).date().isoformat()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM review_state WHERE user_id = ? AND next_due_date <= ?",
            (user_id, today),
        ).fetchone()
    return row["n"] or 0


# ════════════════════════════════════════════════════════════════════════════
# STAGE 2.2 — STREAKS & TREND
# ════════════════════════════════════════════════════════════════════════════

def get_daily_activity(user_id: int, days: int = 30) -> list[dict]:
    """One row per day with any graded session, most recent last — feeds both
    the streak calculation and st.line_chart directly (date, avg_similarity,
    n_segments)."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT substr(created_at, 1, 10) AS day,
                   COUNT(*) AS n_segments,
                   AVG(similarity) AS avg_similarity
            FROM sessions
            WHERE user_id = ?
            GROUP BY day
            ORDER BY day DESC
            LIMIT ?
            """,
            (user_id, days),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]  # chronological order for the chart


def get_current_streak(user_id: int) -> int:
    """Consecutive days (ending today or yesterday) with at least one graded
    session. Breaks as soon as a gap day is found."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT substr(created_at, 1, 10) AS day FROM sessions "
            "WHERE user_id = ? ORDER BY day DESC",
            (user_id,),
        ).fetchall()
    active_days = {r["day"] for r in rows}
    if not active_days:
        return 0

    from datetime import date, timedelta
    cursor = date.today()
    # Today may not have a session yet — that's fine, start counting from
    # the most recent active day as long as it's today or yesterday.
    if cursor.isoformat() not in active_days:
        cursor -= timedelta(days=1)
        if cursor.isoformat() not in active_days:
            return 0

    streak = 0
    while cursor.isoformat() in active_days:
        streak += 1
        cursor -= timedelta(days=1)
    return streak


# ════════════════════════════════════════════════════════════════════════════
# STAGE 2.3 — TEACHER DASHBOARD MVP
# ════════════════════════════════════════════════════════════════════════════

def create_class(teacher_id: int, name: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO classes (teacher_id, name, created_at) VALUES (?, ?, ?)",
            (teacher_id, name.strip(), datetime.now(timezone.utc).isoformat()),
        )
        return cur.lastrowid


def get_teacher_classes(teacher_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, name, created_at FROM classes WHERE teacher_id = ? ORDER BY created_at DESC",
            (teacher_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def add_student_to_class(class_id: int, username: str) -> bool:
    """Returns False if no user has that username (caller should show a
    'no such user' error rather than silently no-op)."""
    with get_conn() as conn:
        user_row = conn.execute(
            "SELECT id FROM users WHERE username = ?", (username.strip(),)
        ).fetchone()
        if user_row is None:
            return False
        conn.execute(
            "INSERT OR IGNORE INTO class_students (class_id, student_id, added_at) VALUES (?, ?, ?)",
            (class_id, user_row["id"], datetime.now(timezone.utc).isoformat()),
        )
        return True


def get_class_roster(class_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT u.id AS student_id, u.username
            FROM class_students cs JOIN users u ON u.id = cs.student_id
            WHERE cs.class_id = ? ORDER BY u.username
            """,
            (class_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def create_assignment(class_id: int, start_surah: int, start_ayah: int,
                       end_surah: int, end_ayah: int, due_date: str | None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO assignments (class_id, start_surah, start_ayah, end_surah, end_ayah,
                                      due_date, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (class_id, start_surah, start_ayah, end_surah, end_ayah, due_date,
             datetime.now(timezone.utc).isoformat()),
        )
        return cur.lastrowid


def get_class_assignments(class_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM assignments WHERE class_id = ? ORDER BY created_at DESC",
            (class_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_student_assignments(student_id: int) -> list[dict]:
    """Assignments across every class the student belongs to — this is what
    the student-side UI reads to know what's outstanding."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT a.*, c.name AS class_name
            FROM assignments a
            JOIN class_students cs ON cs.class_id = a.class_id
            JOIN classes c ON c.id = a.class_id
            WHERE cs.student_id = ?
            ORDER BY a.due_date IS NULL, a.due_date ASC
            """,
            (student_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def submit_assignment_result(assignment_id: int, student_id: int, session_id: int) -> int:
    """Links an already-saved session (see save_session_result) to an
    assignment. Call this right after save_session_result() when the
    student was working through an assigned range, not a free-practice one."""
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO assignment_submissions (assignment_id, student_id, session_id, submitted_at) "
            "VALUES (?, ?, ?, ?)",
            (assignment_id, student_id, session_id, datetime.now(timezone.utc).isoformat()),
        )
        return cur.lastrowid


def get_assignment_submissions(assignment_id: int) -> list[dict]:
    """Everything a teacher needs to review submissions for one assignment,
    joined with the session's score and the student's name."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT sub.id AS submission_id, sub.teacher_note, sub.teacher_override, sub.submitted_at,
                   u.username, s.id AS session_id, s.start_surah, s.start_ayah,
                   s.end_surah, s.end_ayah, s.similarity, s.passed
            FROM assignment_submissions sub
            JOIN users u ON u.id = sub.student_id
            JOIN sessions s ON s.id = sub.session_id
            WHERE sub.assignment_id = ?
            ORDER BY sub.submitted_at DESC
            """,
            (assignment_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def set_teacher_review(submission_id: int, note: str, override: str | None) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE assignment_submissions SET teacher_note = ?, teacher_override = ? WHERE id = ?",
            (note, override, submission_id),
        )


def get_weak_ayahs(user_id: int, limit: int = 20) -> list[dict]:
    """Ayahs with the most 'wrong'/'missing' word-status hits across this
    user's history, most-problematic first. This is the exact query Stage 2's
    SRS queue will read from — the table shape already supports it, this
    function just isn't surfaced in the UI yet."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT sw.surah, sw.ayah, COUNT(*) AS miss_count
            FROM session_words sw
            JOIN sessions s ON s.id = sw.session_id
            WHERE s.user_id = ? AND sw.status IN ('wrong', 'missing')
            GROUP BY sw.surah, sw.ayah
            ORDER BY miss_count DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]
