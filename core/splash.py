"""Splash screen minimalista do Oráculo (rich, estilo ciano)."""

import time

from rich.console import Console
from rich.text import Text

console = Console()

_BORDER = "░" * 33
_PAD = "    "      # recuo das bordas
_INNER = "          "  # recuo do conteúdo interno


def _line(text: str = "", style: str = "") -> Text:
    return Text(text, style=style)


def show_splash(model: str, memory_active: bool = True) -> None:
    """Renderiza a splash e a animação de status antes de liberar o prompt."""
    memoria = "ativa" if memory_active else "inativa"

    console.print()
    console.print(_line(_PAD + _BORDER, style="cyan"))
    console.print()

    titulo = Text(_INNER, style="")
    titulo.append("·  ", style="cyan")
    titulo.append("O R Á C U L O", style="bold bright_cyan")
    titulo.append("  ·", style="cyan")
    console.print(titulo)

    console.print(_line(_INNER + "   ─────────────", style="cyan"))
    console.print(_line(_INNER + "assistente local · offline", style="cyan"))
    console.print()

    # Bloco de status
    def info(label: str, value: str) -> None:
        t = Text(_INNER, style="")
        t.append(f"{label:<8}→  ", style="cyan")
        t.append(value, style="bright_white")
        console.print(t)

    info("modelo", model)
    info("memória", memoria)

    # Linha de status com animação de digitação
    status_prefix = Text(_INNER, style="")
    status_prefix.append(f"{'status':<8}→  ", style="cyan")
    status_prefix.append("pronto", style="bright_white")

    console.print(status_prefix, end="")
    for _ in range(3):
        time.sleep(0.35)
        console.print(Text(".", style="bright_white"), end="")
    console.print()

    console.print()
    console.print(_line(_PAD + _BORDER, style="cyan"))
    console.print()
