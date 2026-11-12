
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

DATABASE = "signalsleuth.db"


def get_db():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    connection = get_db()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            guest_id TEXT,
            detection_type TEXT NOT NULL,
            filename TEXT,
            prediction TEXT NOT NULL,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS guest_usage (
            guest_id TEXT PRIMARY KEY,
            detection_count INTEGER NOT NULL DEFAULT 0
        )
    """)

    connection.commit()
    connection.close()


def register_user(name, email, password):
    connection = get_db()

    try:
        connection.execute(
            "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
            (name, email.lower(), generate_password_hash(password))
        )
        connection.commit()
        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        connection.close()


def login_user(email, password):
    connection = get_db()

    user = connection.execute(
        "SELECT * FROM users WHERE email = ?",
        (email.lower(),)
    ).fetchone()

    connection.close()

    if user and check_password_hash(user["password"], password):
        return user

    return None


def get_guest_count(guest_id):
    connection = get_db()

    row = connection.execute(
        "SELECT detection_count FROM guest_usage WHERE guest_id = ?",
        (guest_id,)
    ).fetchone()

    connection.close()

    return row["detection_count"] if row else 0


def save_detection(user_id, guest_id, detection_type,
                   filename, prediction, confidence):
    connection = get_db()

    connection.execute("""
        INSERT INTO detections
        (user_id, guest_id, detection_type, filename, prediction, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        user_id, guest_id, detection_type,
        filename, prediction, confidence
    ))

    if guest_id:
        connection.execute("""
            INSERT INTO guest_usage (guest_id, detection_count)
            VALUES (?, 1)
            ON CONFLICT(guest_id)
            DO UPDATE SET detection_count = detection_count + 1
        """, (guest_id,))

    connection.commit()
    connection.close()


def get_user_history(user_id):
    connection = get_db()

    records = connection.execute("""
        SELECT * FROM detections
        WHERE user_id = ?
        ORDER BY created_at DESC
    """, (user_id,)).fetchall()

    connection.close()
    return records
