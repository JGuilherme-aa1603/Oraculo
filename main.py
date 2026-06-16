"""Oráculo — assistente de voz local.

Fase 1: chat de terminal com memória e Ollama.
Fase 2: entrada/saída de voz opcional (Whisper STT + Piper TTS), comandos e
        persistência de sessões. O modo texto continua sendo o padrão.
"""

import sys
import time

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

import config
from core import commands, history as history_mod, speaker as speaker_mod
from core.chain import OraculoChain
from core.splash import show_splash

console = Console()

# Intervalo mínimo entre repaints do streaming (s). Casa com o refresh_per_second
# do Live; segura o uso da iGPU (compositor) durante a escrita da resposta.
_REFRESH_INTERVAL = 1 / 6


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
            # Preview ao vivo enquanto a resposta chega, depois render final.
            # transient=True + vertical_overflow="crop": o Live mostra só a
            # última tela e redesenha NO LUGAR (sem isso, resposta mais alta que
            # o terminal faz o Live reemitir tudo a cada frame e o texto repete
            # em cascata). Ao sair, o preview se apaga e imprimimos o Markdown
            # completo uma única vez — rola naturalmente, sem repetição.
            # O throttle (_REFRESH_INTERVAL) evita reparsear o Markdown a cada
            # token; menos repaints = menos uso da iGPU (compositor).
            last_render = 0.0
            with Live(console=console, refresh_per_second=6, transient=True,
                      vertical_overflow="crop") as live:
                for token in chain.stream(user_input):
                    chunks.append(token)
                    if speaker:
                        speaker.feed(token)
                    now = time.monotonic()
                    if now - last_render >= _REFRESH_INTERVAL:
                        live.update(Markdown("".join(chunks)))
                        last_render = now
            response = "".join(chunks)
            console.print(Markdown(response))
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
