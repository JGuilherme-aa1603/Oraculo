"""Oráculo — assistente de voz local.

Fase 1: chat de terminal com memória e Ollama.
Fase 2: entrada/saída de voz opcional (Whisper STT + Piper TTS), comandos e
        persistência de sessões. O modo texto continua sendo o padrão.
"""

import contextlib
import json
import sys
import threading
import time
import urllib.request

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.text import Text

import config
from core import (
    commands,
    history as history_mod,
    keyboard,
    speaker as speaker_mod,
    telemetry,
)
from core.chain import OraculoChain
from core.splash import show_splash

console = Console()

# Intervalo mínimo entre repaints do streaming (s). Casa com o refresh_per_second
# do Live; segura o uso da iGPU (compositor) durante a escrita da resposta.
_REFRESH_INTERVAL = 1 / 6


def _model_is_loaded(model_name: str) -> bool:
    """Consulta o Ollama (/api/ps) para saber se o modelo já está na VRAM."""
    try:
        with urllib.request.urlopen(
            f"{config.OLLAMA_BASE_URL}/api/ps", timeout=1
        ) as resp:
            data = json.loads(resp.read())
    except Exception:  # noqa: BLE001 — Ollama lento/indisponível: assume não carregado
        return False
    for m in data.get("models", []):
        loaded = m.get("model") or m.get("name") or ""
        if model_name in loaded or loaded in model_name:
            return True
    return False


class _ThinkingStatus:
    """Spinner de espera até o 1º token chegar.

    Decide o rótulo pelo estado real do modelo no Ollama:
      - já carregado na VRAM → "Pensando..." (ciano);
      - ainda não carregado  → "Carregando modelo..." (amarelo) e, em background,
        verifica o /api/ps até o modelo subir, então troca para "Pensando...".
    `first_token()` encerra a espera — daí o streaming do texto assume o Live.
    """

    def __init__(self, live: Live, model_name: str) -> None:
        self._live = live
        self._model = model_name
        self._done = threading.Event()

    def start(self) -> None:
        if _model_is_loaded(self._model):
            self._show("Pensando...", "cyan")
        else:
            self._show("Carregando modelo...", "yellow")
            threading.Thread(target=self._poll, daemon=True).start()

    def _poll(self) -> None:
        while not self._done.wait(timeout=0.4):
            if _model_is_loaded(self._model):
                if not self._done.is_set():
                    self._show("Pensando...", "cyan")
                return

    def _show(self, label: str, color: str) -> None:
        with contextlib.suppress(Exception):
            self._live.update(Spinner("dots", text=Text(label, style=f"dim {color}")))

    def first_token(self) -> None:
        self._done.set()


def _speak_until_done(speaker: speaker_mod.StreamSpeaker, ctx: dict) -> Exception | None:
    """Aguarda a fala terminar permitindo barge-in: Esc interrompe na hora e
    devolve o controle para a próxima mensagem. Sem TTY, só aguarda o fim."""
    con = ctx["console"]
    if sys.stdin.isatty() and not ctx.get("esc_hint_shown"):
        con.print("[dim](Esc interrompe a fala)[/]")
        ctx["esc_hint_shown"] = True

    interrupted = threading.Event()

    def _on_esc() -> None:
        interrupted.set()
        speaker.stop()

    with keyboard.watch_key(keyboard.ESC, _on_esc):
        err = speaker.close()
    if interrupted.is_set():
        con.print("[dim](fala interrompida)[/]")
    return err


def _listen(ctx: dict) -> str | None:
    """Captura uma fala no modo voz (push-to-talk). Primeiro Enter inicia a
    gravação, segundo Enter encerra; texto digitado é usado diretamente como
    escape. Retorna o texto ou None se nada foi captado."""
    from core import audio, stt

    ctx["last_stt_seconds"] = None
    typed = console.input(
        "[dim][voz] Enter para começar a gravar (ou digite e Enter):[/] "
    ).strip()
    if typed:
        return typed

    try:
        console.print("[dim](gravando... Enter para parar)[/]")
        path = audio.record_ptt()
        console.print("[dim](transcrevendo...)[/]")
        _stt_t0 = time.monotonic()
        text = stt.transcribe(path)
        ctx["last_stt_seconds"] = time.monotonic() - _stt_t0
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
        stt_seconds = ctx.pop("last_stt_seconds", None)
        console.print(f"[bold bright_cyan]{config.ASSISTANT_NAME}:[/]")
        # No modo voz, a fala é sintetizada frase a frase JÁ DURANTE a geração,
        # sobreposta à escrita — não espera a resposta inteira terminar.
        speaker = speaker_mod.StreamSpeaker() if ctx["voice_mode"] else None
        # Telemetria do turno: t0 começa aqui (pós-STT) — o STT entra como estágio
        # próprio. Marcações e métricas são best-effort e não alteram o turno.
        tel = telemetry.TurnTelemetry()
        tel.mode = "voz" if speaker else "texto"
        try:
            chunks: list[str] = []
            first_token = False
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
                # Spinner de espera: "Carregando modelo..." vira "Pensando..."
                # quando o modelo sobe; o 1º token o substitui pelo texto.
                status = _ThinkingStatus(live, chain.model_name)
                status.start()
                for token in chain.stream(user_input):
                    if not first_token:
                        first_token = True
                        status.first_token()
                        tel.mark("first_token")
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
            tel.set_llm(**chain.last_usage)
        except KeyboardInterrupt:
            console.print("\n[yellow](resposta interrompida)[/]")
        except Exception as exc:  # noqa: BLE001
            console.print(f"\n[bold red]Erro ao responder:[/] {exc}")
        finally:
            if speaker:
                err = _speak_until_done(speaker, ctx)
                if err:
                    console.print(f"[yellow](voz indisponível: {err})[/]")
                tel.mark_at("first_audio", speaker.first_audio_at)
            # Telemetria nunca quebra o turno: tudo em try/except próprio.
            try:
                tel.set_stage("stt", stt_seconds)
                telemetry.log_turn(tel.finish())
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    main()
