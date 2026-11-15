import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any


class BudgetDatabase:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with closing(self.connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS budget_profiles (
                    user_id TEXT PRIMARY KEY,
                    currency TEXT NOT NULL DEFAULT 'INR',
                    monthly_income REAL,
                    monthly_rent REAL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS savings_goals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    target_amount REAL NOT NULL,
                    due_date TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def get_profile(self, user_id: str) -> dict[str, Any] | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM budget_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
        return dict(row) if row else None

    def create_goal(
        self, user_id: str, purpose: str, amount: float, due_date: str
    ) -> int:
        if amount <= 0 or not purpose.strip() or not due_date.strip():
            raise ValueError("A positive amount, purpose, and due date are required.")
        with closing(self.connect()) as connection:
            with connection:
                cursor = connection.execute(
                    "INSERT INTO savings_goals(user_id, purpose, target_amount, due_date) "
                    "VALUES (?, ?, ?, ?)",
                    (user_id, purpose.strip(), amount, due_date.strip()),
                )
                return int(cursor.lastrowid)

    def goals(self, user_id: str) -> list[dict[str, Any]]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT id, purpose, target_amount, due_date, created_at "
                "FROM savings_goals WHERE user_id = ? ORDER BY id",
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]
