from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VARIANT = {
    "number": 12,
    "name": "GM CHANDRA KISHORE",
    "domain": "Personal budgeting coach",
    "twist": "D. Code-mixed input",
    "window_k": 6,
    "context_window": 4096,
    "daily_users": 100_000,
}

FALLBACK_ANSWER = "I couldn't find that in the budgeting guide, so I don't want to guess."


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def kb(self) -> Path:
        return self.data / "kb"

    @property
    def memory(self) -> Path:
        return self.root / "memory"

    @property
    def database(self) -> Path:
        return self.root / "db" / "recallbot.sqlite3"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def index(self) -> Path:
        return self.root / "index"


CONFIGS = {"full", "trim", "summary", "no_rag", "hybrid", "hybrid_sem", "hybrid_mem"}
