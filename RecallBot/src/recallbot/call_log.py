import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


class CallLogger:
    def __init__(self, path: Path, run_id: str | None = None) -> None:
        self.path = path
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def start(self) -> float:
        return time.perf_counter()

    def record(
        self,
        *,
        config: str,
        user: str,
        session: str,
        turn: int,
        call_type: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        retrieved_tokens: int = 0,
        started: float,
    ) -> dict[str, Any]:
        row = {
            "ts": datetime.now().astimezone().isoformat(),
            "run_id": self.run_id,
            "config": config,
            "user": user,
            "session": session,
            "turn": turn,
            "call_type": call_type,
            "model": model,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "retrieved_tokens": int(retrieved_tokens),
            "latency_ms": max(0, round((time.perf_counter() - started) * 1000)),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row
