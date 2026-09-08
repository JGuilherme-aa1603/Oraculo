"""Persistência de sessões de conversa em JSON.

Cada sessão é salva em ~/.oraculo/sessions/<timestamp>.json, alimenta a coluna
"Conversas recentes" da splash e pode ser retomada com `/retomar`. É a base para
uma futura busca no histórico (Fase 4).

Retomar continua escrevendo no **mesmo arquivo**, em vez de abrir um novo: a
conversa é uma só, e partir em dois deixaria dois registros pela metade na lista
de recentes — cada um parecendo uma conversa que morreu cedo.
"""

import contextlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import config


class SessionHistory:
    """Grava a sessão atual em disco a cada turno."""

    def __init__(self, sessions_dir: Path = config.SESSIONS_DIR):
        self.dir = Path(sessions_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.started = datetime.now()
        self.path = self._caminho_livre()
        self._messages: list[dict] = []

    def _caminho_livre(self) -> Path:
        """Um nome que ainda não existe no diretório.

        O carimbo tem resolução de segundo, então duas sessões abertas dentro do
        mesmo segundo — fechar e reabrir, ou dois terminais — caíam no mesmo
        arquivo e a segunda apagava a primeira sem dizer nada. Passava
        despercebido enquanto a sessão gravada era só enfeite da splash; com o
        `/retomar` ela virou algo que se escolhe numa lista e se continua, e aí
        uma conversa engolida pela seguinte é perda de dado.
        """
        base = f"{self.started:%Y%m%d-%H%M%S}"
        caminho = self.dir / f"{base}.json"
        n = 2
        while caminho.exists():
            caminho = self.dir / f"{base}-{n}.json"
            n += 1
        return caminho

    def record(self, role: str, content: str) -> None:
        """Registra uma mensagem (role: 'user' ou 'assistant') e persiste."""
        self._messages.append({"role": role, "content": content, "ts": time.time()})
        self._save()

    def retomar(self, path: Path) -> list[dict]:
        """Passa a continuar a sessão gravada em `path`, devolvendo as mensagens.

        A partir daqui os turnos novos são anexados ao arquivo antigo. A sessão
        recém-criada que estava no lugar não deixa rastro: `_save` só escreve
        quando há mensagem, então um arquivo vazio nunca chegou ao disco.
        """
        dados = json.loads(Path(path).read_text(encoding="utf-8"))
        mensagens = dados.get("messages")
        if not isinstance(mensagens, list):
            raise ValueError("sessão sem lista de mensagens")
        self.path = Path(path)
        self._messages = [m for m in mensagens if isinstance(m, dict)]
        with contextlib.suppress(KeyError, TypeError, ValueError):
            self.started = datetime.fromisoformat(dados["started"])
        return list(self._messages)

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


def _arquivos(sessions_dir: Path) -> list[Path]:
    """Sessões do diretório, da mais recente para a mais antiga.

    Só arquivos regulares: um link simbólico plantado aqui apontaria para fora
    do diretório, e quem apaga por idade seguiria o link.
    """
    directory = Path(sessions_dir)
    if not directory.is_dir():
        return []
    arquivos = [p for p in directory.glob("*.json")
                if p.is_file() and not p.is_symlink()]
    with contextlib.suppress(OSError):
        arquivos.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return arquivos


def load_recent(limit: int = config.RECENT_SESSIONS_ON_SPLASH,
                sessions_dir: Path = config.SESSIONS_DIR) -> list[dict]:
    """Lê as sessões mais recentes do disco para exibir na splash e retomar.

    Retorna: [{"title", "ago", "messages", "path"}, ...] — o `path` é o que o
    `/retomar` usa; a splash ignora.
    """
    recent: list[dict] = []
    for f in _arquivos(sessions_dir)[:limit]:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        recent.append({
            "title": data.get("title") or "(sem título)",
            "ago": _ago(data.get("started")),
            "messages": len(data.get("messages", [])),
            "path": f,
        })
    return recent


def limpar_antigas(sessions_dir: Path = config.SESSIONS_DIR) -> int:
    """Apaga sessões velhas demais. Devolve quantas saíram.

    Duas travas, porque isto apaga dado do usuário:

    - `SESSIONS_MAX_AGE_DAYS = 0` desliga a limpeza por inteiro;
    - as `SESSIONS_KEEP_MIN` mais recentes ficam SEMPRE, por mais antigas que
      sejam. Sem esse piso, voltar de uma temporada longe encontraria o
      histórico inteiro varrido — exatamente quando ele mais serve.

    A idade vem do `mtime` (último turno gravado), não do nome do arquivo: uma
    sessão de março retomada ontem está viva, e o nome diria o contrário.
    """
    dias = config.SESSIONS_MAX_AGE_DAYS
    if not dias or dias <= 0:
        return 0

    arquivos = _arquivos(sessions_dir)
    candidatos = arquivos[max(0, config.SESSIONS_KEEP_MIN):]
    if not candidatos:
        return 0

    limite = (datetime.now() - timedelta(days=dias)).timestamp()
    removidas = 0
    for path in candidatos:
        try:
            if path.stat().st_mtime >= limite:
                continue
            path.unlink()
            removidas += 1
        except OSError:
            continue          # permissão, corrida com outro processo: ignora
    return removidas
