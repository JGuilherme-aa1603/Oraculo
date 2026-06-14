"""Oráculo — assistente de voz local.

Fase 1: chat de terminal com memória e Ollama.
Fase 2: entrada/saída de voz opcional (Whisper STT + Piper TTS), comandos e
        persistência de sessões. O modo texto continua sendo o padrão.
"""

import sys

from rich.console import Console

import config
from core import commands, history as history_mod
from core.chain import OraculoChain
from core.splash import show_splash

console = Console()


def _listen(ctx: dict) -> str | None:
    """Captura uma fala no modo voz (push-to-talk). Primeiro Enter inicia a
    gravação, segundo Enter encerra; texto digitado é usado diretamente como
    escape. Retorna o texto ou None se nada foi captado."""
    from core import audio, stt

    typed = console.input(
        "[dim][voz] Enter para começar a gravar (ou digite e Enter):[/] "
    ).strip()
    if typed:
        return typed

    try:
        console.print("[dim](gravando... Enter para parar)[/]")
        path = audio.record_ptt()
        console.print("[dim](transcrevendo...)[/]")
        text = stt.transcribe(path)
    except RuntimeError as exc:
        console.print(f"[yellow]{exc}[/]")
        console.print("[yellow]Voltando ao modo texto.[/]")
        ctx["voice_mode"] = False
        return None

    if not text:
        console.print("[dim](não entendi nada — tente de novo)[/]")
        return None

    console.print(f"[dim](você disse: {text})[/]")
    return text


def _speak(text: str) -> None:
    """Sintetiza e reproduz a resposta. Falhas não derrubam a conversa."""
    from core import audio, tts

    try:
        wav = tts.speak(text)
        audio.play(wav)
    except (RuntimeError, Exception) as exc:  # noqa: BLE001
        console.print(f"[yellow](voz indisponível: {exc})[/]")


def main() -> None:
    try:
        chain = OraculoChain()
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"[bold red]Falha ao iniciar o {config.ASSISTANT_NAME}:[/] {exc}\n"
            "[yellow]O Ollama está rodando? Tente: ollama serve[/]"
        )
        sys.exit(1)

    show_splash(chain.model_name, recent_sessions=history_mod.load_recent())

    history = history_mod.SessionHistory()
    ctx = {
        "console": console,
        "chain": chain,
        "running": True,
        "voice_mode": config.VOICE_MODE_DEFAULT,
    }

    while ctx["running"]:
        try:
            if ctx["voice_mode"]:
                user_input = _listen(ctx)
                if not user_input:
                    continue
            else:
                user_input = console.input("[bold cyan]Você:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[cyan]Encerrando...[/]")
            break

        if not user_input:
            continue

        if commands.handle(user_input, ctx):
            continue

        history.record("user", user_input)
        console.print(f"[bold bright_cyan]{config.ASSISTANT_NAME}:[/] ", end="")
        try:
            chunks: list[str] = []
            for token in chain.stream(user_input):
                chunks.append(token)
                console.print(token, end="", style="bright_white")
            console.print()
            response = "".join(chunks)
            history.record("assistant", response)
            if ctx["voice_mode"] and response:
                _speak(response)
        except KeyboardInterrupt:
            console.print("\n[yellow](resposta interrompida)[/]")
        except Exception as exc:  # noqa: BLE001
            console.print(f"\n[bold red]Erro ao responder:[/] {exc}")


if __name__ == "__main__":
    main()
