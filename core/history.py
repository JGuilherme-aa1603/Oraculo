"""Persistência de sessões de conversa em JSON.

Cada sessão é salva em ~/.oraculo/sessions/<timestamp>.json e alimenta a coluna
"Conversas recentes" da splash. É a base para uma futura busca no histórico (Fase 4).
"""

import json
import time
from datetime import datetime
from pathlib import Path

import config


class SessionHistory:
    """Grava a sessão atual em disco a cada turno."""

    def __init__(self, sessions_dir: Path = config.SESSIONS_DIR):
        self.dir = Path(sessions_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.started = datetime.now()
        self.path = self.dir / f"{self.started:%Y%m%d-%H%M%S}.json"
        self._messages: list[dict] = []

    def record(self, role: str, content: str) -> None:
        """Registra uma mensagem (role: 'user' ou 'assistant') e persiste."""
        self._messages.append({"role": role, "content": content, "ts": time.time()})
        self._save()

    def _title(self) -> str:
        for m in self._messages:
            if m["role"] == "user" and m["content"].strip():
                return m["content"].strip()
        return "(sem título)"

    def _save(self) -> None:
        if not self._messages:
            return
        data = {
            "started": self.started.isoformat(),
            "title": self._title(),
            "messages": self._messages,
        }
        try:
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        except OSError:
            # Persistência é best-effort; nunca deve derrubar a conversa.
            pass


def _ago(iso: str | None) -> str:
    """Formata um timestamp ISO como tempo relativo em pt-BR."""
    if not iso:
        return ""
    try:
        then = datetime.fromisoformat(iso)
    except ValueError:
        return ""
    secs = (datetime.now() - then).total_seconds()
    if secs < 60:
        return "agora"
    mins = secs / 60
    if mins < 60:
        return f"há {int(mins)} min"
    hours = mins / 60
    if hours < 24:
        h = int(hours)
        return f"há {h} hora" + ("s" if h > 1 else "")
    days = int(hours / 24)
    if days == 1:
        return "ontem"
    return f"há {days} dias"


def load_recent(limit: int = config.RECENT_SESSIONS_ON_SPLASH,
                sessions_dir: Path = config.SESSIONS_DIR) -> list[dict]:
    """Lê as sessões mais recentes do disco para exibir na splash.

    Retorna: [{"title": str, "ago": str, "messages": int}, ...]
    """
    directory = Path(sessions_dir)
    if not directory.exists():
        return []
    files = sorted(directory.glob("*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    recent: list[dict] = []
    for f in files[:limit]:
        try:
            data = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        recent.append({
            "title": data.get("title") or "(sem título)",
            "ago": _ago(data.get("started")),
            "messages": len(data.get("messages", [])),
        })
    return recent
