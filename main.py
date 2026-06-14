"""Oráculo — assistente de voz local (Fase 1: chat de terminal).

Entry point: loop de conversa no terminal com memória e Ollama via LangChain.
"""

import sys

from rich.console import Console

import config
from core.chain import OraculoChain
from core.splash import show_splash

console = Console()


def main() -> None:
    show_splash(config.OLLAMA_MODEL, memory_active=True)

    try:
        chain = OraculoChain()
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"[bold red]Falha ao iniciar o {config.ASSISTANT_NAME}:[/] {exc}\n"
            "[yellow]O Ollama está rodando? Tente: ollama serve[/]"
        )
        sys.exit(1)

    while True:
        try:
            user_input = console.input("[bold cyan]Você:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[cyan]Encerrando...[/]")
            break

        if not user_input:
            continue

        if user_input.lower() in config.EXIT_COMMANDS:
            console.print("[cyan]Encerrando...[/]")
            break

        console.print(f"[bold bright_cyan]{config.ASSISTANT_NAME}:[/] ", end="")
        try:
            for token in chain.stream(user_input):
                console.print(token, end="", style="bright_white")
            console.print()
        except KeyboardInterrupt:
            console.print("\n[yellow](resposta interrompida)[/]")
        except Exception as exc:  # noqa: BLE001
            console.print(f"\n[bold red]Erro ao responder:[/] {exc}")


if __name__ == "__main__":
    main()
