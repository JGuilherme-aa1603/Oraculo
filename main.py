"""Oráculo — assistente de voz local.

Fase 1: chat de terminal com memória e Ollama.
Fase 2: entrada/saída de voz opcional (Whisper STT + Piper TTS), comandos e
        persistência de sessões. O modo texto continua sendo o padrão.
"""

import sys

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

import config
from core import commands, history as history_mod, speaker as speaker_mod
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
        console.print(f"[bold bright_cyan]{config.ASSISTANT_NAME}:[/]")
        # No modo voz, a fala é sintetizada frase a frase JÁ DURANTE a geração,
        # sobreposta à escrita — não espera a resposta inteira terminar.
        speaker = speaker_mod.StreamSpeaker() if ctx["voice_mode"] else None
        try:
            chunks: list[str] = []
            # Live + Markdown: renderiza a resposta formatada enquanto chega,
            # em vez de despejar os símbolos crus (*, #, ```) no terminal.
            with Live(console=console, refresh_per_second=12,
                      vertical_overflow="visible") as live:
                for token in chain.stream(user_input):
                    chunks.append(token)
                    if speaker:
                        speaker.feed(token)
                    live.update(Markdown("".join(chunks)))
            response = "".join(chunks)
            history.record("assistant", response)
        except KeyboardInterrupt:
            console.print("\n[yellow](resposta interrompida)[/]")
        except Exception as exc:  # noqa: BLE001
            console.print(f"\n[bold red]Erro ao responder:[/] {exc}")
        finally:
            if speaker:
                err = speaker.close()
                if err:
                    console.print(f"[yellow](voz indisponível: {err})[/]")


if __name__ == "__main__":
    main()
