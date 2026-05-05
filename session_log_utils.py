from __future__ import annotations

from datetime import datetime

from config import SESSION_LOG_FILE


def append_session_log(action: str, result: str, location: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    with open(SESSION_LOG_FILE, "a", encoding="utf-8") as file_obj:
        file_obj.write(f"[{timestamp}] {action} | {result} | {location}\n")