"""Splash screen do Oráculo (v2) — layout de duas colunas estilo Claude Code.

Identidade à esquerda, comandos + conversas recentes à direita, dentro de uma
moldura ciano. As conversas recentes vêm de core.history.load_recent().
"""

import getpass
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import config

console = Console()

_RULE = "─" * 22
_SYMBOLS = "◈    ⟁    ◈"


def _user_name() -> str:
    if config.USER_NAME:
        return config.USER_NAME
    return getpass.getuser().capitalize()


def _cwd_display() -> str:
    cwd = Path.cwd()
    home = Path.home()
    try:
        return "~/" + str(cwd.relative_to(home))
    except ValueError:
        return str(cwd)


def _truncate(text: str, width: int = 46) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= width else text[: width - 1].rstrip() + "…"


def _join(lines: list[Text]) -> Text:
    out = Text()
    for i, line in enumerate(lines):
        if i:
            out.append("\n")
        out.append_text(line)
    return out


def _identity(model: str, memory_active: bool) -> list[Text]:
    memoria = "ativa" if memory_active else "inativa"
    return [
        Text(""),
        Text(f"Bem-vindo, {_user_name()}!", style="bold white"),
        Text(""),
        Text(_SYMBOLS, style="bold bright_cyan"),
        Text(""),
        Text(f"{model} · offline · local", style="grey70"),
        Text(f"memória {memoria} · {config.DEVICE_LABEL}", style="grey70"),
        Text(_cwd_display(), style="grey50"),
        Text(""),
    ]


def _info(recent_sessions: list[dict]) -> list[Text]:
    lines: list[Text] = [
        Text("Comandos disponíveis", style="bold cyan"),
        Text(_RULE, style="cyan dim"),
    ]

    def cmd_line(prefix: str, command: str, suffix: str) -> Text:
        t = Text(prefix, style="grey85")
        t.append(command, style="bright_cyan")
        t.append(suffix, style="grey85")
        return t

    lines.append(cmd_line("Digite ", "/ajuda", " para ver todos os comandos"))
    lines.append(cmd_line("Use ", "/modelo", " para trocar o modelo ativo"))
    lines.append(cmd_line("Use ", "/sair", " para encerrar o Oráculo"))
    lines.append(Text(""))
    lines.append(Text("Conversas recentes", style="bold cyan"))
    lines.append(Text(_RULE, style="cyan dim"))

    if not recent_sessions:
        lines.append(Text("nenhuma conversa ainda", style="grey50"))
        return lines

    for s in recent_sessions:
        lines.append(Text(_truncate(s.get("title", "")), style="grey85"))
        meta = Text("  ")
        meta.append(s.get("ago", ""), style="cyan dim")
        meta.append(f" · {s.get('messages', 0)} mensagens", style="grey50")
        lines.append(meta)
    return lines


def show_splash(model: str, recent_sessions: list[dict] | None = None,
                memory_active: bool = True) -> None:
    """Renderiza a splash de duas colunas."""
    recent_sessions = recent_sessions or []

    left = _identity(model, memory_active)
    right = _info(recent_sessions)

    height = max(len(left), len(right))
    # Centraliza verticalmente a coluna da esquerda em relação à direita.
    pad_top = (height - len(left)) // 2
    left = [Text("")] * pad_top + left
    left += [Text("")] * (height - len(left))
    right += [Text("")] * (height - len(right))

    divider = _join([Text("│", style="cyan dim") for _ in range(height)])

    grid = Table.grid(expand=True, padding=(0, 2))
    # no_wrap + elipse: garante que nenhuma célula ganhe linhas extras por quebra
    # de texto, o que desalinharia o divisor vertical.
    grid.add_column(justify="center", ratio=5, no_wrap=True, overflow="ellipsis")
    grid.add_column(justify="center", width=1, no_wrap=True)
    grid.add_column(justify="left", ratio=6, no_wrap=True, overflow="ellipsis")
    grid.add_row(_join(left), divider, _join(right))

    title = Text.assemble((f"{config.ASSISTANT_NAME} ", "bold cyan"),
                          (f"v{config.APP_VERSION}", "white"))

    console.print()
    console.print(Panel(grid, title=title, title_align="left",
                        border_style="cyan", box=box.SQUARE, padding=(1, 2)))
    console.print()
