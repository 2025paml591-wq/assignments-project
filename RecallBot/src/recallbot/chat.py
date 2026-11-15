import json
import urllib.error
import urllib.request
from typing import Protocol

from .call_log import CallLogger
from .tokens import tokenizer


class ChatModel(Protocol):
    name: str

    def complete(
        self,
        prompt: str,
        *,
        system: str,
        temperature: float,
        logger: CallLogger,
        config: str,
        user: str,
        session: str,
        turn: int,
        call_type: str = "chat",
        retrieved_tokens: int = 0,
    ) -> str: ...


class OllamaChat:
    def __init__(
        self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"
    ) -> None:
        self.name = model
        self.base_url = base_url.rstrip("/")

    def complete(
        self,
        prompt: str,
        *,
        system: str,
        temperature: float,
        logger: CallLogger,
        config: str,
        user: str,
        session: str,
        turn: int,
        call_type: str = "chat",
        retrieved_tokens: int = 0,
    ) -> str:
        started = logger.start()
        payload = {
            "model": self.name,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": temperature},
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(
                f"Chat request failed for {self.name} at {self.base_url}: {error}"
            ) from error
        answer = result.get("message", {}).get("content")
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Ollama returned an empty or malformed chat response.")
        prompt_tokens = result.get("prompt_eval_count", tokenizer().count(system + prompt))
        answer_tokens = result.get("eval_count", tokenizer().count(answer))
        logger.record(
            config=config,
            user=user,
            session=session,
            turn=turn,
            call_type=call_type,
            model=self.name,
            input_tokens=prompt_tokens,
            output_tokens=answer_tokens,
            retrieved_tokens=retrieved_tokens,
            started=started,
        )
        return answer.strip()


class TemplateChat:
    """Deterministic offline smoke-test engine; do not use its runs as LLM evidence."""

    name = "template-smoke-test"

    def complete(
        self,
        prompt: str,
        *,
        system: str,
        temperature: float,
        logger: CallLogger,
        config: str,
        user: str,
        session: str,
        turn: int,
        call_type: str = "chat",
        retrieved_tokens: int = 0,
    ) -> str:
        started = logger.start()
        answer = "I have noted that. What would you like to review next?"
        if "Retrieved knowledge:" in prompt:
            source = prompt.split("Retrieved knowledge:", 1)[1].splitlines()
            excerpt = next((line for line in source if line.strip()), "")
            if excerpt and excerpt != "(none)":
                answer = f"From the guide: {excerpt[:500]}"
        elif "Long-term semantic memory:" in prompt:
            section = prompt.split("Long-term semantic memory:", 1)[1].split("Episodic memory:", 1)[0]
            excerpt = next((line.strip() for line in section.splitlines() if line.strip()), "")
            if excerpt and excerpt != "(none)":
                answer = f"From stored facts: {excerpt}"
        elif "Episodic memory:" in prompt:
            section = prompt.split("Episodic memory:", 1)[1].split("Database:", 1)[0]
            excerpt = next((line.strip() for line in section.splitlines() if line.strip()), "")
            if excerpt and excerpt != "(none)":
                answer = f"From earlier sessions: {excerpt}"
        elif "pending confirmation" in prompt.lower():
            answer = "Please confirm explicitly before I save this budget goal."
        elif "State:" in prompt:
            answer = "I used the current conversation state to resolve that follow-up."
        logger.record(
            config=config,
            user=user,
            session=session,
            turn=turn,
            call_type=call_type,
            model=self.name,
            input_tokens=tokenizer().count(system + prompt),
            output_tokens=tokenizer().count(answer),
            retrieved_tokens=retrieved_tokens,
            started=started,
        )
        return answer
