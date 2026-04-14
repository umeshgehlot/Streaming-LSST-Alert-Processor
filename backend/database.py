import json
import os
import sqlite3
import uuid
from datetime import datetime
from hashlib import pbkdf2_hmac
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv() -> bool:
        return False

load_dotenv()


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("ASTRO_DB_PATH", str(BASE_DIR / "app.db"))).resolve()
UPLOAD_DIR = Path(os.getenv("ASTRO_UPLOAD_DIR", str(BASE_DIR.parent / "data" / "uploads"))).resolve()
RESULT_DIR = Path(os.getenv("ASTRO_RESULT_DIR", str(BASE_DIR / "results"))).resolve()

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS train_runs (
                id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                epochs INTEGER NOT NULL,
                final_loss REAL NOT NULL,
                model_path TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                anomaly_indices TEXT NOT NULL,
                scores_path TEXT NOT NULL,
                threshold REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                full_name TEXT NOT NULL,
                role TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS collaboration_rooms (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                dataset_id TEXT,
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS collaboration_comments (
                id TEXT PRIMARY KEY,
                room_id TEXT NOT NULL,
                author TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS public_labels (
                id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                point_index INTEGER NOT NULL,
                label TEXT NOT NULL,
                user_name TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alert_channels (
                id TEXT PRIMARY KEY,
                channel_type TEXT NOT NULL,
                target TEXT NOT NULL,
                min_confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS provenance_ledger (
                id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                result_id TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                previous_hash TEXT,
                block_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gamification_scores (
                user_name TEXT PRIMARY KEY,
                total_points INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS discoveries (
                id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                result_id TEXT,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                reasoning TEXT NOT NULL,
                meta TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_activities (
                id TEXT PRIMARY KEY,
                stage TEXT NOT NULL,
                message TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS expert_feedback (
                id TEXT PRIMARY KEY,
                discovery_id TEXT NOT NULL,
                researcher TEXT NOT NULL,
                reward REAL NOT NULL,
                notes TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rl_policy (
                id TEXT PRIMARY KEY,
                threshold_percentile REAL NOT NULL,
                sensitivity REAL NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rl_training_history (
                id TEXT PRIMARY KEY,
                reward REAL NOT NULL,
                precision REAL NOT NULL,
                threshold_before REAL NOT NULL,
                threshold_after REAL NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_results_dataset_created ON results (dataset_id, created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_results_dataset_model_created ON results (dataset_id, model_name, created_at DESC)")
        result_columns = {row["name"] for row in conn.execute("PRAGMA table_info(results)").fetchall()}
        if "feedback" not in result_columns:
            conn.execute("ALTER TABLE results ADD COLUMN feedback TEXT NOT NULL DEFAULT 'unreviewed'")
        seed_default_users(conn)


def hash_password(password: str, salt_hex: str) -> str:
    digest = pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), 120000)
    return digest.hex()


def seed_default_users(conn: sqlite3.Connection) -> None:
    current_count = conn.execute("SELECT COUNT(*) AS total FROM users").fetchone()["total"]
    if current_count > 0:
        return
    defaults = [
        ("admin@astro.local", "Project Admin", "admin", "admin123"),
        ("researcher@astro.local", "Research Analyst", "researcher", "research123"),
    ]
    now = datetime.utcnow().isoformat()
    for email, full_name, role, plain_password in defaults:
        salt = uuid.uuid4().hex
        password_hash = hash_password(plain_password, salt)
        conn.execute(
            """
            INSERT INTO users (id, email, full_name, role, password_hash, salt, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (str(uuid.uuid4()), email, full_name, role, password_hash, salt, now),
        )


def insert_dataset(filename: str, raw_bytes: bytes) -> str:
    dataset_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{dataset_id}_{filename}"
    with open(file_path, "wb") as file:
        file.write(raw_bytes)
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO datasets (id, filename, file_path, created_at) VALUES (?, ?, ?, ?)",
            (dataset_id, filename, str(file_path), now),
        )
    return dataset_id


def fetch_dataset(dataset_id: str):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
    if row is None:
        return None
    with open(row["file_path"], "rb") as file:
        raw_bytes = file.read()
    return {"id": row["id"], "filename": row["filename"], "raw_bytes": raw_bytes}


def create_train_run(
    dataset_id: str,
    model_name: str,
    epochs: int,
    final_loss: float,
    model_path: str,
) -> str:
    run_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO train_runs (id, dataset_id, model_name, epochs, final_loss, model_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, dataset_id, model_name, epochs, final_loss, model_path, now),
        )
    return run_id


def create_result(
    dataset_id: str,
    model_name: str,
    anomaly_indices: list[int],
    scores_path: str,
    threshold: float,
    feedback: str = "unreviewed",
) -> str:
    result_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO results (id, dataset_id, model_name, anomaly_indices, scores_path, threshold, feedback, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result_id,
                dataset_id,
                model_name,
                json.dumps(anomaly_indices),
                scores_path,
                threshold,
                feedback,
                now,
            ),
        )
    return result_id


def fetch_results(
    dataset_id: str | None,
    result_id: str | None = None,
    model_name: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    query = "SELECT * FROM results"
    params: list[str] = []
    conditions: list[str] = []
    if result_id is not None:
        conditions.append("id = ?")
        params.append(result_id)
    if dataset_id is not None:
        conditions.append("dataset_id = ?")
        params.append(dataset_id)
    if model_name is not None:
        conditions.append("model_name = ?")
        params.append(model_name)
    if len(conditions) > 0:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    mapped: list[dict] = []
    for row in rows:
        mapped.append(
            {
                "id": row["id"],
                "dataset_id": row["dataset_id"],
                "model_name": row["model_name"],
                "anomaly_indices": json.loads(row["anomaly_indices"]),
                "scores_path": row["scores_path"],
                "threshold": row["threshold"],
                "feedback": row["feedback"],
                "created_at": row["created_at"],
            }
        )
    return mapped


def update_result_feedback(result_id: str, feedback: str) -> bool:
    normalized = feedback.strip().lower()
    if normalized not in {"unreviewed", "real_discovery", "instrumental_noise"}:
        raise ValueError("feedback must be one of: unreviewed, real_discovery, instrumental_noise")
    with get_connection() as conn:
        cursor = conn.execute("UPDATE results SET feedback = ? WHERE id = ?", (normalized, result_id))
    return cursor.rowcount > 0


def count_results(dataset_id: str, model_name: str | None = None) -> int:
    query = "SELECT COUNT(*) AS total FROM results WHERE dataset_id = ?"
    params: list[str] = [dataset_id]
    if model_name is not None:
        query += " AND model_name = ?"
        params.append(model_name)
    with get_connection() as conn:
        row = conn.execute(query, params).fetchone()
    return int(row["total"])


def build_scores_path(dataset_id: str, model_name: str) -> str:
    filename = f"{dataset_id}_{model_name}_scores.csv"
    path = RESULT_DIR / filename
    return str(path)


def build_model_path(dataset_id: str, model_name: str) -> str:
    model_dir = BASE_DIR.parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return str(model_dir / f"{dataset_id}_{model_name}.pt")


def fetch_user_by_email(email: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "email": row["email"],
        "full_name": row["full_name"],
        "role": row["role"],
        "password_hash": row["password_hash"],
        "salt": row["salt"],
    }


def create_collaboration_room(name: str, dataset_id: str | None, created_by: str) -> str:
    room_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO collaboration_rooms (id, name, dataset_id, created_by, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (room_id, name, dataset_id, created_by, now),
        )
    return room_id


def list_collaboration_rooms(dataset_id: str | None = None) -> list[dict]:
    query = "SELECT * FROM collaboration_rooms"
    params: list[str] = []
    if dataset_id:
        query += " WHERE dataset_id = ?"
        params.append(dataset_id)
    query += " ORDER BY created_at DESC"
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def add_collaboration_comment(room_id: str, author: str, message: str) -> str:
    comment_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO collaboration_comments (id, room_id, author, message, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (comment_id, room_id, author, message, now),
        )
    return comment_id


def list_collaboration_comments(room_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM collaboration_comments WHERE room_id = ? ORDER BY created_at ASC",
            (room_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def add_public_label(dataset_id: str, point_index: int, label: str, user_name: str) -> str:
    label_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO public_labels (id, dataset_id, point_index, label, user_name, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (label_id, dataset_id, point_index, label, user_name, now),
        )
    add_gamification_points(user_name, 5)
    return label_id


def list_public_labels(dataset_id: str, limit: int = 200) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM public_labels WHERE dataset_id = ? ORDER BY created_at DESC LIMIT ?",
            (dataset_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def add_alert_channel(channel_type: str, target: str, min_confidence: float) -> str:
    channel_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO alert_channels (id, channel_type, target, min_confidence, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (channel_id, channel_type, target, float(min_confidence), now),
        )
    return channel_id


def list_alert_channels() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM alert_channels ORDER BY created_at DESC").fetchall()
    return [dict(row) for row in rows]


def get_last_block_hash() -> str | None:
    with get_connection() as conn:
        row = conn.execute("SELECT block_hash FROM provenance_ledger ORDER BY created_at DESC LIMIT 1").fetchone()
    return None if row is None else str(row["block_hash"])


def add_provenance_block(dataset_id: str, result_id: str, payload_hash: str, previous_hash: str | None, block_hash: str) -> str:
    block_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO provenance_ledger (id, dataset_id, result_id, payload_hash, previous_hash, block_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (block_id, dataset_id, result_id, payload_hash, previous_hash, block_hash, now),
        )
    return block_id


def list_provenance_blocks(dataset_id: str | None = None, limit: int = 100) -> list[dict]:
    query = "SELECT * FROM provenance_ledger"
    params: list[str | int] = []
    if dataset_id:
        query += " WHERE dataset_id = ?"
        params.append(dataset_id)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def add_gamification_points(user_name: str, points: int) -> None:
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        row = conn.execute("SELECT total_points FROM gamification_scores WHERE user_name = ?", (user_name,)).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO gamification_scores (user_name, total_points, updated_at) VALUES (?, ?, ?)",
                (user_name, int(points), now),
            )
            return
        new_total = int(row["total_points"]) + int(points)
        conn.execute(
            "UPDATE gamification_scores SET total_points = ?, updated_at = ? WHERE user_name = ?",
            (new_total, now, user_name),
        )


def get_gamification_leaderboard(limit: int = 20) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT user_name, total_points, updated_at FROM gamification_scores ORDER BY total_points DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def create_discovery(
    dataset_id: str,
    result_id: str | None,
    status: str,
    confidence: float,
    reasoning: str,
    meta: dict,
) -> str:
    discovery_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO discoveries (id, dataset_id, result_id, status, confidence, reasoning, meta, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                discovery_id,
                dataset_id,
                result_id,
                status,
                float(confidence),
                reasoning,
                json.dumps(meta),
                now,
                now,
            ),
        )
    return discovery_id


def update_discovery_status(discovery_id: str, status: str) -> None:
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            "UPDATE discoveries SET status = ?, updated_at = ? WHERE id = ?",
            (status, now, discovery_id),
        )


def list_discoveries(dataset_id: str | None = None, limit: int = 100) -> list[dict]:
    query = "SELECT * FROM discoveries"
    params: list[str | int] = []
    if dataset_id:
        query += " WHERE dataset_id = ?"
        params.append(dataset_id)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    mapped: list[dict] = []
    for row in rows:
        mapped.append(
            {
                "id": row["id"],
                "dataset_id": row["dataset_id"],
                "result_id": row["result_id"],
                "status": row["status"],
                "confidence": row["confidence"],
                "reasoning": row["reasoning"],
                "meta": json.loads(row["meta"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )
    return mapped


def add_agent_activity(stage: str, message: str, payload: dict) -> str:
    activity_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO agent_activities (id, stage, message, payload, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (activity_id, stage, message, json.dumps(payload), now),
        )
    return activity_id


def list_agent_activities(limit: int = 200) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM agent_activities ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "stage": row["stage"],
            "message": row["message"],
            "payload": json.loads(row["payload"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def add_expert_feedback(discovery_id: str, researcher: str, reward: float, notes: str) -> str:
    feedback_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO expert_feedback (id, discovery_id, researcher, reward, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (feedback_id, discovery_id, researcher, float(reward), notes, now),
        )
    return feedback_id


def get_feedback_summary() -> dict:
    with get_connection() as conn:
        total_row = conn.execute("SELECT COUNT(*) AS total FROM expert_feedback").fetchone()
        positive_row = conn.execute("SELECT COUNT(*) AS positive FROM expert_feedback WHERE reward > 0").fetchone()
    total = int(total_row["total"])
    positive = int(positive_row["positive"])
    precision = positive / max(1, total)
    return {"total_feedback": total, "positive_feedback": positive, "precision_proxy": precision}


def get_or_create_rl_policy() -> dict:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM rl_policy ORDER BY updated_at DESC LIMIT 1").fetchone()
        if row is not None:
            return {
                "id": row["id"],
                "threshold_percentile": float(row["threshold_percentile"]),
                "sensitivity": float(row["sensitivity"]),
                "updated_at": row["updated_at"],
            }
        policy_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        conn.execute(
            """
            INSERT INTO rl_policy (id, threshold_percentile, sensitivity, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (policy_id, 95.0, 1.0, now),
        )
    return {"id": policy_id, "threshold_percentile": 95.0, "sensitivity": 1.0, "updated_at": now}


def update_rl_policy(threshold_percentile: float, sensitivity: float) -> dict:
    policy = get_or_create_rl_policy()
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE rl_policy
            SET threshold_percentile = ?, sensitivity = ?, updated_at = ?
            WHERE id = ?
            """,
            (float(threshold_percentile), float(sensitivity), now, policy["id"]),
        )
    return {
        "id": policy["id"],
        "threshold_percentile": float(threshold_percentile),
        "sensitivity": float(sensitivity),
        "updated_at": now,
    }


def add_rl_training_event(reward: float, precision: float, threshold_before: float, threshold_after: float) -> str:
    event_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO rl_training_history (id, reward, precision, threshold_before, threshold_after, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (event_id, float(reward), float(precision), float(threshold_before), float(threshold_after), now),
        )
    return event_id


def list_rl_training_history(limit: int = 200) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM rl_training_history ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]
