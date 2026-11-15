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
