"""Thin Ollama wrapper (step 5). The LLM is used for routing, drafting and review ONLY.
Numerical predictions never come from here."""
from __future__ import annotations
import json
import re
from typing import Any, Dict, List

from .config import OLLAMA_EMBED_MODEL, OLLAMA_HOST, OLLAMA_MODEL


class LLMUnavailable(RuntimeError):
    pass


class OllamaLLM:
    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL,
                 embed_model: str = OLLAMA_EMBED_MODEL, force_offline: bool = False):
        self.host = host
        self.model = model
        self.embed_model = embed_model
        self.force_offline = force_offline
        self._client = None
        self._chat_ok: bool | None = None
        self._embed_ok: bool | None = None
        self.status_message = "offline mode (--no-llm)" if force_offline else ""

    # ---- connection -------------------------------------------------------
    @property
    def client(self):
        if self._client is None:
            import ollama
            self._client = ollama.Client(host=self.host, timeout=180)
        return self._client

    def _local_models(self) -> List[str]:
        resp = self.client.list()
        models = resp["models"] if isinstance(resp, dict) or hasattr(resp, "__getitem__") else resp.models
        names = []
        for m in models:
            name = m.get("model") or m.get("name") if isinstance(m, dict) else getattr(m, "model", None)
            if name:
                names.append(name)
        return names

    def chat_available(self) -> bool:
        if self.force_offline:
            return False
        if self._chat_ok is None:
            try:
                names = self._local_models()
                if self.model not in names and f"{self.model}:latest" not in names:
                    chat_models = [n for n in names if "embed" not in n.lower()]
                    if not chat_models:
                        raise RuntimeError(f"model '{self.model}' not found; run: ollama pull {self.model}")
                    self.status_message = f"'{self.model}' not installed, using '{chat_models[0]}' instead"
                    self.model = chat_models[0]
                self._chat_ok = True
            except Exception as exc:  # server down, model missing, ...
                self._chat_ok = False
                self.status_message = f"Ollama chat unavailable: {exc}"
        return self._chat_ok

    def embed_available(self) -> bool:
        if self.force_offline:
            return False
        if self._embed_ok is None:
            try:
                names = self._local_models()
                if not any(n == self.embed_model or n.startswith(self.embed_model.split(":")[0]) for n in names):
                    raise RuntimeError(f"embedding model '{self.embed_model}' not found; "
                                       f"run: ollama pull {self.embed_model}")
                self._embed_ok = True
            except Exception as exc:
                self._embed_ok = False
                if not self.status_message:
                    self.status_message = f"Ollama embeddings unavailable: {exc}"
        return self._embed_ok

    # ---- calls ------------------------------------------------------------
    def chat(self, system: str, user: str, json_mode: bool = False, temperature: float = 0.1) -> str:
        if not self.chat_available():
            raise LLMUnavailable(self.status_message or "Ollama not available")
        kwargs: Dict[str, Any] = {"options": {"temperature": temperature}}
        if json_mode:
            kwargs["format"] = "json"
        try:
            resp = self.client.chat(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                **kwargs,
            )
        except Exception as exc:
            raise LLMUnavailable(str(exc)) from exc
        return resp["message"]["content"].strip()

    def chat_json(self, system: str, user: str) -> Dict[str, Any]:
        raw = self.chat(system, user, json_mode=True, temperature=0.0)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, flags=re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
        raise LLMUnavailable(f"model did not return valid JSON: {raw[:120]!r}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self.embed_available():
            raise LLMUnavailable(self.status_message or "Ollama embeddings not available")
        try:
            resp = self.client.embed(model=self.embed_model, input=texts)
        except Exception as exc:
            raise LLMUnavailable(str(exc)) from exc
        return resp["embeddings"]
